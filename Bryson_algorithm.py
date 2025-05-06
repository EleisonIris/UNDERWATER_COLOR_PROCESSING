import os
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from PIL import Image
import math
import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.linalg import cho_factor, cho_solve
from dataclasses import dataclass, field
from typing import List, Tuple
import json
import re
import sys
sys.path.append('/Users/Sarah/Desktop/sea-thru')
import calib_luca as param_calib
from utils import r0_rd
from range_map import compute_intersection
from face_dans_im_OPTI import generate_view_matrix, generate_view_vector
from backprojection import back_projeter

def import_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        print("Mesh is invalid or empty!")
    return mesh

def load_images(image_folder: str, image_prefix: str, right=False):
    image_files = [f for f in os.listdir(image_folder) if re.match(f'{image_prefix}_[lr]_\d{{4}}\.jpeg', f)]
    image_files.sort(key=lambda x: int(re.search(r'(\d{4})\.jpeg', x).group(1)))
    left_images = []
    right_images = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
        else:
            print(f"Found image at {image_path}!!")
            if 'l_' in image_file:
                left_images.append(image)
            elif right and 'r_' in image_file:
                right_images.append(image)
    return left_images + right_images

def load_transforms_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            transforms_data = json.load(f)
        transforms = []
        count = 0
        for key in sorted(transforms_data.keys(), key=int):
            for sub_key in sorted(transforms_data[key].keys(), key=int):
                transform = transforms_data[key][sub_key]
                r_vec = np.array(transform[0], dtype=np.float64)
                t_vec = np.array(transform[1], dtype=np.float64)
                rot_matrix, _ = cv2.Rodrigues(r_vec)  # Convert Rodrigues vector to rotation matrix
                transforms.append((rot_matrix, t_vec))
                if count < 5:
                    print(f"Transform {count + 1} for key {key}, sub_key {sub_key}:")
                    print(f"Rotation (matrix):\n{rot_matrix}")
                    print(f"Translation: {t_vec}\n")
                    count += 1
        return transforms
    except Exception as e:
        print(f"Error loading transforms from {file_path}: {e}")
        return []

def create_scene(mesh):
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)
    return scene

def select_random_triangles(mesh, Nt):
    faces =  np.asarray(mesh.triangles)
    vertices =  np.asarray(mesh.vertices)
    num_faces = len(faces)
    if Nt > num_faces:
        raise ValueError(f"Nt ({Nt}) is larger than the total number of triangles ({num_faces}) in the mesh!")
    selected_indices = np.random.choice(num_faces, Nt, replace=False)
    selected_faces = faces[selected_indices]
    unique_vertex_indices, inverse_indices =  np.unique(selected_faces.flatten(), return_inverse=True)
    selected_vertices = vertices[unique_vertex_indices]
    remapped_faces = inverse_indices.reshape(selected_faces.shape)
    selected_mesh = o3d.geometry.TriangleMesh()
    selected_mesh.vertices = o3d.utility.Vector3dVector(np.array(selected_vertices))
    selected_mesh.triangles = o3d.utility.Vector3iVector(np.array(remapped_faces))
    return selected_mesh, selected_indices

def find_images(selected_mesh, Nt, transforms, h, w, subset, right=False):
    l_view_matrix = generate_view_matrix(selected_mesh, Nt, transforms, h, w, type_camera="l", subset=subset)
    if right:
        r_view_matrix = generate_view_matrix(selected_mesh, Nt, transforms, h, w, type_camera="r", subset=subset)
    else:
        r_view_matrix = []
    views_should_be = 0
    for tri in range(Nt):
        for view in range(len(transforms)):
            if l_view_matrix[tri, view] == 1:
                views_should_be += 1
            if right and r_view_matrix[tri, view] == 1:
                views_should_be += 1
    return l_view_matrix, r_view_matrix, views_should_be

def compute_cam_pos(transforms, right=False):
    cam_pos_l = []
    if right:
        Rot_DG = param_calib.R_DG
        t_DG = param_calib.t_DG
        R_DG, _ = cv2.Rodrigues(Rot_DG)
        cam_pos_r = []
    for R, t in transforms:
        cam_pos_l.append(-R.T @ t) 
        if right:
            cam_pos_r.append(-R.T @ (t + t_DG))
    cam_pos_l = np.array(cam_pos_l)
    if right:
        cam_pos_r = np.array(cam_pos_r)
        cam_pos = np.concatenate((cam_pos_l, cam_pos_r), axis=0)
    else:
        cam_pos = cam_pos_l
    return cam_pos

def compute_lamp_pos(transforms, lamp_transforms):
    num_lamps = len(lamp_transforms)
    num_views = len(transforms)
    lamp_pos = np.zeros((num_views, num_lamps, 3))  
    for i, (R, t) in enumerate(transforms):
        for j, lamp_offset in enumerate(lamp_transforms):
            lamp_pos[i, j] = R.T @ (lamp_offset - t)
    return lamp_pos

def compute_centerlines(transforms, ang_x=0, ang_y=0, Nl=4):
    ang_x = np.radians(ang_x)  
    ang_y = np.radians(ang_y)
    centerlines = np.zeros((len(transforms), Nl, 3))
    for i in range(Nl):  
        base_dir = np.array([0, 0, 1])
        ax = ang_x[i] 
        ay = ang_y[i]  
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(ax), -np.sin(ax)],
                        [0, np.sin(ax), np.cos(ax)]])
        
        R_y = np.array([[np.cos(ay), 0, np.sin(ay)],
                        [0, 1, 0],
                        [-np.sin(ay), 0, np.cos(ay)]])
        R = np.dot(R_y, R_x)
        rotated_dir = np.dot(R, base_dir)
        rotated_dir /= np.linalg.norm(rotated_dir)  
        for j, (Rot, _) in enumerate(transforms):
            rotated_dir_view = Rot.T @ rotated_dir
            centerlines[j, i] = rotated_dir_view  
    return centerlines

def triangle_midpoints(selected_mesh):
    faces =  np.asarray(selected_mesh.triangles)
    vertices =  np.asarray(selected_mesh.vertices)
    midpoints = []
    for face in faces:
        v1, v2, v3 = vertices[face]
        midpoint = (v1 + v2 + v3) / 3 #TODO: coor barycentriques
        midpoints.append(midpoint)
    t_midpoints =  np.array(midpoints)
    return t_midpoints

def compute_triangle_normals(selected_mesh):
    selected_mesh.compute_triangle_normals()
    t_normals =  np.asarray(selected_mesh.triangle_normals)
    return t_normals

def compute_lamp_vectors(t_midpoints, lamp_pos):
    l_vectors = t_midpoints[:, None, None, :] - lamp_pos[None, :, :, :]
    return l_vectors

def compute_single_view_lamp_vectors(t_midpoints, lamp_pos):
    l_vectors = t_midpoints[:, None, :] - lamp_pos[None, :, :]
    return l_vectors

def compute_angles(t_normals, l_vectors, centerlines, Nt, Nl, images, right=False):
    if right:
        div = 2
    else:
        div = 1
    Nv = len(images) // div
    Phi = np.zeros((Nt, Nv, Nl))
    Theta = np.zeros((Nt, Nv, Nl))
    for i in range(Nt):
        for j in range(Nv):
            for k in range(Nl):
                L_set = l_vectors[i, j, k, :]  
                L_norms = L_set / np.linalg.norm(L_set, axis=0)  
                N = t_normals[i]  
                cos_Theta_vals = np.clip(np.dot(L_norms, -N), -1.0, 1.0)  
                Theta[i, j, k] = np.arccos(cos_Theta_vals)
                lamp_dir = centerlines[j, k, :]  
                cos_Phi_vals = np.clip(np.dot(L_norms, lamp_dir), -1.0, 1.0)  
                Phi[i, j, k] = np.arccos(cos_Phi_vals)
    return Phi, Theta

def compute_alpha(t_midpoints, cam_positions, transforms, right=False):
    Nt = len(t_midpoints)
    Nv = len(cam_positions) 
    full_rotations = []
    for R, _ in transforms:
        full_rotations.append(R)  
    if right:      
        Rot_DG = param_calib.R_DG 
        for R, _ in transforms:
            full_rotations.append(R @ Rot_DG) 
    alpha_values = np.zeros((Nt, Nv))
    vec_to_camera = np.zeros((Nt, Nv, 3))
    camera_axis_world = np.zeros((Nv, 3))
    for i in range(Nt):
        midpoint = t_midpoints[i]
        for j in range(Nv):
            cam_pos = cam_positions[j]
            R = full_rotations[j]
            cam_axis = R.T @ np.array([0, 0, 1])  
            vec_to_cam = cam_pos - midpoint
            vec_to_cam /= np.linalg.norm(vec_to_cam)
            alpha = np.arccos(np.clip(np.dot(vec_to_cam, -cam_axis), -1.0, 1.0))
            alpha_values[i, j] = alpha
            vec_to_camera[i , j] = vec_to_cam
            camera_axis_world[j] = cam_axis
    return camera_axis_world, vec_to_camera, alpha_values

def compute_single_view_angles(t_normals, l_vectors, centerlines, Ntot, Nl, right=False):
    Phi = np.zeros((Ntot, Nl))
    Theta = np.zeros((Ntot, Nl))
    for i in range(Ntot):
        for k in range(Nl):
            L_set = l_vectors[i, k, :]  
            L_norms = L_set / np.linalg.norm(L_set, axis=0)  
            N = t_normals[i]  
            cos_Theta_vals = np.clip(np.dot(L_norms, -N), -1.0, 1.0)  
            Theta[i, k] = np.arccos(cos_Theta_vals)
            lamp_dir = centerlines[k, :]  
            cos_Phi_vals = np.clip(np.dot(L_norms, lamp_dir), -1.0, 1.0)  
            Phi[i, k] = np.arccos(cos_Phi_vals)
    return Phi, Theta

def compute_single_view_alpha(t_midpoints, Ntot, cam_positions, transforms, images, image_idx, right=False):
    full_rotations = []
    full_rotations.append(transforms[0])  
    if right:      
        Rot_DG = param_calib.R_DG 
        full_rotations.append(transforms[0] @ Rot_DG) 
    alpha_values = np.zeros(Ntot)
    for i in range(Ntot):
        midpoint = t_midpoints[i]
        cam_pos = cam_positions
        if not right or image_idx < len(images)//2:
            R = full_rotations[0]
        elif right and image_idx >= len(images)//2:
            R = full_rotations[1]
        cam_axis_world = R.T @ np.array([0, 0, 1])  
        vec_to_camera = cam_pos - midpoint
        vec_to_camera /= np.linalg.norm(vec_to_camera)
        alpha = np.arccos(np.clip(np.dot(vec_to_camera, -cam_axis_world), -1.0, 1.0))
        alpha_values[i] = alpha
    return alpha_values


def compute_ranges(t_midpoints, camera_pos, lamp_pos, Nt, Nl, l_view_matrix, r_view_matrix, right=False):
    if right:
        div = 2
    else:
        div = 1
    N_views = len(camera_pos)
    rc = np.full((Nt, N_views), np.nan)
    rl = np.full((Nt, N_views, Nl), np.nan)
    invalid_range_values = 0
    for i in range(Nt):
        for j in range(N_views):
            if j < N_views // div:  
                cam_l = camera_pos[j]
                if l_view_matrix[i, j] == 1:
                    rc_ij = np.linalg.norm(t_midpoints[i] - cam_l)
                    rc[i, j] = rc_ij
                    if np.isnan(rc[i, j]):
                        print("Careful, invalid left value for rc.")
                        invalid_range_values += 1
                    for k in range(Nl):
                        lamp_pos_jk = lamp_pos[j, k]
                        rl_ijk = np.linalg.norm(t_midpoints[i] - lamp_pos_jk)
                        rl[i, j, k] = rl_ijk
                        if np.isnan(rl[i, j, k]):
                            print(f"Careful, invalid value for rl at lamp {k}.")
                            invalid_range_values += 1
            elif right and j >= N_views // div: 
                cam_r = camera_pos[j]
                if r_view_matrix[i, j - N_views // 2] == 1:
                    rc_ij = np.linalg.norm(t_midpoints[i] - cam_r)
                    rc[i, j] = rc_ij
                    if np.isnan(rc[i, j]):
                        print("Careful, invalid right value for rc.")
                        invalid_range_values += 1
                    for k in range(Nl):  
                        lamp_pos_jk = lamp_pos[j - N_views // 2, k]  
                        rl_ijk = np.linalg.norm(t_midpoints[i] - lamp_pos_jk)
                        rl[i, j, k] = rl_ijk
                        if np.isnan(rl[i, j, k]):
                            print(f"Careful, invalid value for rl at lamp {k} (right view).")
                            invalid_range_values += 1
    print("Number of invalid range values: ", invalid_range_values)
    return rc / 1000, rl / 1000

def compute_single_view_ranges(t_midpoints, camera_pos, lamp_pos, visible_indices, Nl):
    rc = np.full(len(visible_indices), np.nan)
    rl = np.full((len(visible_indices), Nl), np.nan)
    invalid_range_values = 0
    for i in range(len(visible_indices)): 
        rc[i] = np.linalg.norm(t_midpoints[i] - camera_pos)
        if np.isnan(rc[i]):
            print("Careful, invalid left value for rc.")
            invalid_range_values += 1
        for k in range(Nl):
            rl[i, k] = np.linalg.norm(t_midpoints[i] - lamp_pos[k])
            if np.isnan(rl[i, k]):
                print(f"Careful, invalid value for rl at lamp {k}.")
                invalid_range_values += 1
    print("Number of invalid range values: ", invalid_range_values)
    return rc / 1000, rl / 1000

def get_triangle_vertices_as_pixels(mesh, triangle_idx, transforms, image_idx, cam="l", right=False):
    if right:
        div = 2
    else:
        div = 1
    triangle = mesh.triangles[triangle_idx]
    vertices = np.array(mesh.vertices)
    if image_idx < len(images)//div: 
        rot, t = transforms[image_idx]
    if right and len(images)//div <= image_idx:
        rot, t = transforms[image_idx - len(images)//div]
    vert1= back_projeter(vertices[triangle[0]], rot, t, cam)[0]
    vert2 = back_projeter(vertices[triangle[1]], rot, t, cam)[0]
    vert3 = back_projeter(vertices[triangle[2]], rot, t, cam)[0]
    return vert1, vert2, vert3

def is_within_bounds(px, image_shape):
    return 0 <= px[0] < image_shape[1] and 0 <= px[1] < image_shape[0]

def get_triangle_intensity_per_channel(px1, px2, px3, image):
    px1, px2, px3 = np.array(px1, dtype=np.int32), np.array(px2, dtype=np.int32), np.array(px3, dtype=np.int32)
    triangle = np.array([px1, px2, px3], dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [triangle], 255)
    triangle_pixels = cv2.bitwise_and(image, image, mask=mask)
    r_intensity = triangle_pixels[mask == 255, 0]
    g_intensity = triangle_pixels[mask == 255, 1]
    b_intensity = triangle_pixels[mask == 255, 2]
    r_mean = r_intensity.mean() if len(r_intensity) > 0 else 0.0
    g_mean = g_intensity.mean() if len(g_intensity) > 0 else 0.0
    b_mean = b_intensity.mean() if len(b_intensity) > 0 else 0.0
    return r_mean, g_mean, b_mean

def mean_triangle_intensities_over_views(mesh, transforms, images, Nt, l_view_matrix, r_view_matrix, right=False):
    if right:
        div = 2
    else:
        div = 1
    channel_intensities = []
    r_intensities = np.zeros(Nt)
    g_intensities = np.zeros(Nt)
    b_intensities = np.zeros(Nt)
    l_invalid_projection = 0
    r_invalid_projection = 0
    for trg_idx, trg in enumerate(mesh.triangles):
        views = 0
        for img_idx, img in enumerate(images):
            if img_idx < len(images)//div-1 and l_view_matrix[trg_idx, img_idx] == 1:
                vert1, vert2, vert3 = get_triangle_vertices_as_pixels(mesh, trg_idx, transforms, img_idx, cam="l", right=right)
                for px in [vert1, vert2, vert3]:
                    if not is_within_bounds(px, img.shape):
                        l_invalid_projection += 1
                r_mean, g_mean, b_mean = get_triangle_intensity_per_channel(vert1, vert2, vert3, img)
                r_intensities[trg_idx] += r_mean
                g_intensities[trg_idx] += g_mean
                b_intensities[trg_idx] += b_mean
                views += 1
            if right and len(images)//div-1 < img_idx:
                r_idx = img_idx - len(images)//div
                if r_idx < r_view_matrix.shape[1] and r_view_matrix[trg_idx, r_idx] == 1:
                    vert1, vert2, vert3 = get_triangle_vertices_as_pixels(mesh, trg_idx, transforms, img_idx, cam="r", right=right)
                    for px in [vert1, vert2, vert3]:
                        if not is_within_bounds(px, img.shape):
                            r_invalid_projection += 1
                    r_mean, g_mean, b_mean = get_triangle_intensity_per_channel(vert1, vert2, vert3, img)
                    r_intensities[trg_idx] += r_mean
                    g_intensities[trg_idx] += g_mean
                    b_intensities[trg_idx] += b_mean
                    views += 1
        if views > 0:
            r_intensities[trg_idx] /= views
            g_intensities[trg_idx] /= views
            b_intensities[trg_idx] /= views
        else:
            r_intensities[trg_idx] = 0.0
            g_intensities[trg_idx] = 0.0
            b_intensities[trg_idx] = 0.0
    print("Invalid projections (left/right): ", l_invalid_projection, r_invalid_projection)
    channel_intensities = r_intensities, g_intensities, b_intensities
    return channel_intensities

def C(C_alpha, alpha):
    return 1 + C_alpha[0] * alpha**2 + C_alpha[1] * alpha**4 + C_alpha[2] * alpha**6

def sigma(phi_50):
    phi_50_rad = np.deg2rad(phi_50)
    return (phi_50_rad ** 2) / (-2 *  np.log(0.5))

def P(phi, sigma_l, P0_wavelength):
    P_r, P_g, P_b = P0_wavelength *  np.exp(-0.5*(phi ** 2) / sigma_l)
    return P_r, P_g, P_b

def B(b_wavelength, beta_wavelength, rc, triangle, view):
    rc_tri = rc[triangle, view]
    B = (beta_wavelength / b_wavelength) * (1 -  np.exp(-b_wavelength * rc_tri))
    return B

def single_view_B(b_wavelength, beta_wavelength, rc, triangle):
    rc_tri = rc[triangle]
    B = (beta_wavelength / b_wavelength) * (1 -  np.exp(-b_wavelength * rc_tri))
    return B

def intialize_x(mesh, transforms, images, Nt, l_view_matrix, r_view_matrix, right=False):
    r_attenuation, g_attenuation, b_attenuation = [0.01] * 3 
    r_scatter, g_scatter, b_scatter = [0.01] * 3
    C_alpha = [0.01] * 3
    r_albedo, g_albedo, b_albedo = mean_triangle_intensities_over_views(mesh, transforms, images, Nt, l_view_matrix, r_view_matrix, right=right)
    x = np.array([r_attenuation, g_attenuation, b_attenuation,
        r_scatter, g_scatter, b_scatter,
        *C_alpha], dtype=np.float64)
    x = np.concatenate((x, r_albedo / 255.0, g_albedo / 255.0, b_albedo / 255.0))
    return x

def observation(mesh, transforms, images, Nt, l_view_matrix, r_view_matrix, right=False):
    if right:
        div = 2
    else:
        div = 1
    z = []
    l_invalid_projection = 0
    r_invalid_projection = 0
    for trg_idx in range(Nt):
        for img_idx, img in enumerate(images):
            if img_idx < len(images)//div and l_view_matrix[trg_idx, img_idx] == 1:
                vert1, vert2, vert3 = get_triangle_vertices_as_pixels(mesh, trg_idx, transforms, img_idx, cam="l", right=right)
                if any(not is_within_bounds(px, img.shape) for px in [vert1, vert2, vert3]):
                    l_invalid_projection += 1
                    continue
                mean = get_triangle_intensity_per_channel(vert1, vert2, vert3, img)
                z.append(mean)
            elif right and len(images)//div <= img_idx and r_view_matrix[trg_idx, img_idx-len(images)//div] == 1: 
                vert1, vert2, vert3 = get_triangle_vertices_as_pixels(mesh, trg_idx, transforms, img_idx, cam="r", right=right)
                if any(not is_within_bounds(px, img.shape) for px in [vert1, vert2, vert3]):
                    r_invalid_projection += 1
                    continue
                mean = get_triangle_intensity_per_channel(vert1, vert2, vert3, img)
                z.append(mean)
            else:
                continue
    print("Invalid projections (left/right):", l_invalid_projection, r_invalid_projection)
    return np.array(z).T / 255.0

def single_observation(mesh, transforms, images, img_idx, Nt, view_matrix, right, isRight=False):
    z = []
    l_invalid_projection = 0
    r_invalid_projection = 0
    img = images[img_idx]
    for trg_idx in range(Nt):
        if not isRight and view_matrix[trg_idx] == 1 and img_idx < len(images)//2:
            vert1, vert2, vert3 = get_triangle_vertices_as_pixels(mesh, trg_idx, transforms, img_idx, cam="l", right=right)
            if any(not is_within_bounds(px, img.shape) for px in [vert1, vert2, vert3]):
                l_invalid_projection += 1
                continue
            mean = get_triangle_intensity_per_channel(vert1, vert2, vert3, img)
            z.append(mean)
        elif isRight and view_matrix[trg_idx] == 1 and img_idx >= len(images)//2: 
            vert1, vert2, vert3 = get_triangle_vertices_as_pixels(mesh, trg_idx, transforms, img_idx, cam="r", right=right)
            if any(not is_within_bounds(px, img.shape) for px in [vert1, vert2, vert3]):
                r_invalid_projection += 1
                continue
            mean = get_triangle_intensity_per_channel(vert1, vert2, vert3, img)
            z.append(mean)
        else:
            continue
    print("Invalid projections (left/right):", l_invalid_projection, r_invalid_projection)
    return np.array(z) / 255.0

def single_lamp_irradiance(x, Nt, phi_50, P0_wavelength, alpha_values, triangle, view, lamp, Theta, Phi, rc, rl, right=False):
    if right:
        div = 2
    else:
        div = 1
    attenuation_values = x[0:3]
    C_alpha = x[6:9]
    albedo_values = np.array([
        x[9 + triangle],
        x[9 + Nt + triangle],
        x[9 + 2*Nt + triangle]
    ])  
    alpha = alpha_values[triangle, view]
    C_val = C(C_alpha, alpha)
    sigma_l = sigma(phi_50)
    phi_val = Phi[triangle, view//div, lamp]
    P_values = P(phi_val, sigma_l, P0_wavelength)
    if view < len(images)//div:  
        if l_view_matrix[triangle, view] == 1:  
            theta_val = Theta[triangle, view, lamp]
            rl_val = rl[triangle, view, lamp]
    elif right and view >= len(images)//div:  
        if r_view_matrix[triangle, view - len(images)//div] == 1:  
            theta_val = Theta[triangle, view - len(images)//div, lamp]
            rl_val = rl[triangle, view, lamp]
            if np.isnan(rl_val):
                print("WARNING: accessed nan right lamp range at ", triangle, view, ".")
    rc_val = rc[triangle, view]
    attenuation_term = attenuation_values * (rc_val + rl_val)
    exp_term = np.exp(-attenuation_term)
    irradiance = C_val * albedo_values * P_values * np.cos(theta_val) * exp_term
    return irradiance

def total_lamp_irradiance_per_triangle(x, phi_50, P0_wavelength, alpha_values, triangle, view, Nl, Nt, Theta, Phi, rc, rl, right=False):
    all_irradiance = [0, 0, 0]
    for lamp in range(Nl):
        all_irradiance += single_lamp_irradiance(x, Nt, phi_50, P0_wavelength, alpha_values, triangle, view, lamp, Theta, Phi, rc, rl, right=right)
    return all_irradiance

def h(x, alpha_values, rc, rl, Phi, Theta, Nt, Nl, phi_50, P0_wavelength, k=1, right=False):
    if right:
        div = 2
    else:
        div = 1
    attenuation_values = x[0:3]  
    scatter_values = x[3:6]  
    irradiance = [] 
    for i in range(Nt):
        for j in range(len(images)):
            try:
                if (j < len(images)//div and l_view_matrix[i, j] == 1) or (right and j >= len(images)//div and r_view_matrix[i, j - len(images)//2] == 1):
                    irradiance_values = total_lamp_irradiance_per_triangle(
                        x, phi_50, P0_wavelength, alpha_values, i, j, Nl, Nt, Theta, Phi, rc, rl, right=right)
                    if np.isnan(irradiance_values).any():
                        print(f"NaN in irradiance_values at triangle={i}, view={j}")
                        continue
                    B_vals = B(attenuation_values, scatter_values, rc, triangle=i, view=j)
                    if np.isnan(B_vals).any():
                        print(f"NaN in B_vals at triangle={i}, view={j}")
                        continue
                    z_model = k * (np.array(irradiance_values) + np.array(B_vals))
                    if np.isnan(z_model).any():
                        print(f"NaN in z_model at triangle={i}, view={j}")
                        continue
                    irradiance.append(z_model)
            except Exception as e:
                print(f"Error at triangle={i}, view={j}: {e}")
    result = np.stack(irradiance, axis=1) if irradiance else np.array([])
    return result

def residuals(x, z, Alpha, rc, rl, Phi, Theta, Nt, Nl, phi_50, P0_wavelength, k, right=False, excluded_indices=None):
    pred = h(x, Alpha, rc, rl, Phi, Theta, Nt, Nl, phi_50, P0_wavelength, k, right=right)
    pred_flat = pred.flatten()
    z_flat = z.flatten()

    if excluded_indices:
        mask = np.ones_like(z_flat, dtype=bool)
        mask[list(excluded_indices)] = False
        pred_flat = pred_flat[mask]
        z_flat = z_flat[mask]

    return pred_flat - z_flat


def define_bounds(Nt):
    lower = np.array(
        [1e-5] * 6 +      # attenuations and scatterings
        [-10.0] * 3 +     # C_alpha
        [0.0] * 3 * Nt    # albedo
    )
    upper = np.array(
        [1.0] * 6 +       # attenuations and scatterings
        [10.0] * 3 +      # C_alpha
        [1.0] * 3 * Nt    # albedo
    )
    return (lower, upper)

def compute_K(x, alpha, tri, phi_50, Nl, P0_wavelength, Phi, Theta, rc, rl):
    attenuation_values = x[0:3] 
    C_alpha = x[6:9]
    C_val = C(C_alpha, alpha)
    print(C_val.shape)
    sigma_l = sigma(phi_50)
    rc_val = rc[tri]
    K = 0
    for lamp in range(Nl):
        phi_val = Phi[tri, lamp]
        P_val = np.array(P(phi_val, sigma_l, P0_wavelength))
        angle = Theta[tri, lamp]
        rl_val = rl[tri, lamp]
        single_lamp_K = C_val[tri] * P_val * np.cos(angle) * np.exp(-attenuation_values * (rc_val + rl_val))
        K += single_lamp_K
    return K

def single_view_matrix_and_obs(mesh, transforms, images, h, w, image_idx, right=False, subset=None):
    Ntot = len(mesh.triangles)
    if not right:
        view_matrix = generate_view_vector(mesh, image_idx, Ntot, transforms, h, w, type_camera="l", subset=subset)
        z = single_observation(mesh, transforms, images, image_idx, Ntot, view_matrix, right=right, isRight=False)
    elif image_idx < len(images)//2:
        view_matrix = generate_view_vector(mesh, image_idx, Ntot, transforms, h, w, type_camera="l", subset=subset)
        z = single_observation(mesh, transforms, images, image_idx, Ntot, view_matrix, right=right, isRight=False)
    else:
        view_matrix = generate_view_vector(mesh, image_idx - len(images)//2, Ntot, transforms, h, w, images, type_camera="r", subset=subset)
        z = single_observation(mesh, transforms, images, image_idx, Ntot, view_matrix, right=right, isRight=True)
    print("Observation and view matrix computed. Moving on!")
    return view_matrix, z

def collect_visible_mesh_and_indices(mesh, view_matrix):
    visible_indices = [i for i in range(len(mesh.triangles)) if view_matrix[i]]
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    visible_triangles = triangles[visible_indices]
    unique_vertex_indices, inverse_indices = np.unique(visible_triangles.flatten(), return_inverse=True)
    visible_vertices = vertices[unique_vertex_indices]
    remapped_triangles = inverse_indices.reshape((-1, 3))
    visible_mesh = o3d.geometry.TriangleMesh()
    visible_mesh.vertices = o3d.utility.Vector3dVector(visible_vertices)
    visible_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
    return visible_mesh, visible_indices

def compute_albedo(x, full_z, visible_indices, Alpha, phi_50, Nl, P0_wavelength, k, rc, rl, Phi, Theta):
    Ntot = len(visible_indices)
    b_wavelength = x[0:3] 
    beta_wavelength = x[3:6]  
    full_albedo = np.zeros((Ntot, 3))
    for tri in range(len(visible_indices)): 
        K = compute_K(x, Alpha, tri, phi_50, Nl, P0_wavelength, Phi, Theta, rc, rl) 
        B_val = single_view_B(b_wavelength, beta_wavelength, rc, tri) 
        print(K, B_val, full_z[tri])
        albedo = (full_z[tri] / k - B_val) / K  
        full_albedo[tri] = albedo  
        print("Albedo computed for triangle N°", visible_indices[tri])
    return full_albedo.T

def air_intensities(albedo, view, t_normals, visible_indices, right=False):
    I_air = []
    normals = []
    for tri in range(len(visible_indices)):
        if not right or view < len(images)//2:
            normals.append(t_normals[tri])
        elif right and view >= len(images)//2:
            normals.append(t_normals[tri])
        else:
            normals.append(np.nan)
    cos_theta_z = normals @ np.array([0, 1, 0])
    I_air = albedo * cos_theta_z * 255.0
    return I_air.T

def recover_image(visible_mesh, images, image_idx, I_air, transforms, right=False):
    image_shape = images[image_idx].shape
    recovered_image = np.zeros_like(images[image_idx])
    modified_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for tri, triangle in enumerate(visible_mesh.triangles):
        triangle_indices = visible_mesh.triangles[tri]
        vert1 = visible_mesh.vertices[triangle_indices[0]]
        vert2 = visible_mesh.vertices[triangle_indices[1]]
        vert3 = visible_mesh.vertices[triangle_indices[2]]
        cam = "r" if right and image_idx >= len(images)//2 else "l"
        px1 = back_projeter(vert1, transforms[0], transforms[1], cam=cam)[0]
        px2 = back_projeter(vert2, transforms[0], transforms[1], cam=cam)[0]
        px3 = back_projeter(vert3, transforms[0], transforms[1], cam=cam)[0]
        if all(is_within_bounds(px, image_shape) for px in [px1, px2, px3]):
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            triangle = np.array([px1, px2, px3], dtype=np.int32)
            cv2.fillPoly(mask, [triangle], 1)
            modified_mask[mask == 1] = 1
            for c in range(3): 
                recovered_image[:, :, c][mask == 1] = I_air[tri, c]
        else:
            print("Triangle vertices out of bounds, skipping triangle.")
    uncovered_mask = modified_mask == 0
    for c in range(3):
        recovered_image[:, :, c][uncovered_mask] = images[image_idx][:, :, c][uncovered_mask]
    return recovered_image

def display_visible_mesh_with_camera(mesh, cam_pos):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)
    o3d_mesh.compute_vertex_normals()
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
    camera_sphere.translate(cam_pos)
    camera_sphere.paint_uniform_color([1, 0, 0]) 
    o3d.visualization.draw_geometries([o3d_mesh, camera_sphere])

@dataclass
class PipelineArgs:
    right: bool = True
    Nt: int = 1000
    Nl: int = 4
    k: int = 1
    Tr: float = 0.95  
    Tg: float = 1.0   
    Tb: float = 0.9  
    Phi_50: np.ndarray = field(default_factory=lambda: np.array([67.5, 67.5, 67.5]))
    P0_wavelength: np.ndarray = field(default_factory=lambda: np.array([3.75, 6.0, 5.25])) # adds up to 15W for a color spectrum at approx. 5700K
    ang_x: List[float] = field(default_factory=lambda: [
        10, 10, -10, -10
    ])
    ang_y: List[float] = field(default_factory=lambda: [
        10, -10, 10, -10
    ])
    lamp_transforms: List[np.ndarray] = field(default_factory=lambda: [
        np.array([20, -20, 0]),
        np.array([-20, -20, 0]),
        np.array([20, 20, 0]),
        np.array([-20, 20, 0])
    ])
    image_height: int = 2000
    image_width: int = 3000
    image_idx: int = 0 # index of the image to reconstruct
    mesh_path: str = "/Users/Sarah/Desktop/sea-thru/cailloux_high_res.ply"
    image_path_prefix: str = "scene"

if __name__ == "__main__":
    print("=== IMPORTING SCENE DATA ===")
    args = PipelineArgs() 
    print("=== Step 1: Importing mesh ===")
    mesh = import_mesh(args.mesh_path)
    scene = create_scene(mesh)
    print("Mesh loaded. Number of triangles:", len(mesh.triangles))
    
    transforms_file_path = '/Users/Sarah/Desktop/sea-thru/absolute_transforms_luca.json'
    transforms = load_transforms_from_file(transforms_file_path)
    args.transforms = transforms

    image_folder = '/Users/Sarah/Desktop/sea-thru/downsampled'
    images = load_images(image_folder, args.image_path_prefix, args.right)
    if images:
        print(f"Loaded {len(images)} images.")
    
    if transforms:    
        print(f"Loaded {len(transforms)} transforms.") 

    print("=== OPTIMIZATION : LET'S RECOVER PARAMETERS FROM SCENE GEOMETRY ===")
    random_mesh, random_indices = select_random_triangles(mesh, args.Nt)
    if len(random_indices) > 0:
        print(f"Selected {len(random_indices)} random triangles.")

    l_view_matrix, r_view_matrix, views_should_be = find_images(
        mesh, args.Nt, transforms, args.image_height, args.image_width, random_indices, args.right
    )

    cam_pos = compute_cam_pos(transforms, args.right)
    print("cam positions:", cam_pos.shape)

    lamp_pos = compute_lamp_pos(transforms, args.lamp_transforms)
    print("lamp positions:", lamp_pos.shape)

    centerlines = compute_centerlines(transforms, args.ang_x, args.ang_y, args.Nl)
    print("centerlines:", centerlines.shape)

    t_midpoints = triangle_midpoints(random_mesh)
    t_normals = compute_triangle_normals(random_mesh)
    print("Midpoints and normals computed.")

    l_vectors = compute_lamp_vectors(t_midpoints, lamp_pos)
    print("lamp vectors:", l_vectors.shape)

    Phi, Theta = compute_angles(t_normals, l_vectors, centerlines, args.Nt, args.Nl, images, args.right)
    _, _, Alpha = compute_alpha(t_midpoints, cam_pos, transforms, args.right)
    print("Phi/Theta/Alpha:", Phi.shape, Theta.shape, Alpha.shape)

    #for i in range(args.Nt-1):
    #   for j in range(len(images)//2-1):
    #        if l_view_matrix[i,j] == 1:
    #            print("Theta: ", np.degrees(Theta[i,j]), "Phi: " , np.degrees(Phi[i,j]), "Alpha: ", np.degrees(Alpha[i,j]))
    #raise SystemExit

    rc, rl = compute_ranges(t_midpoints, cam_pos, lamp_pos, args.Nt, args.Nl, l_view_matrix, r_view_matrix, args.right)
    #print("rc/rl:", rc.shape, rl.shape)
    #print(rc, rl)
    #raise SystemExit

    x0 = intialize_x(random_mesh, transforms, images, args.Nt, l_view_matrix, r_view_matrix, args.right)
    z = observation(random_mesh, transforms, images, args.Nt, l_view_matrix, r_view_matrix, args.right)
    print("x0/z:", x0, z.shape)

    print("Views should be:", views_should_be)
    print("Parameters and observation initialized, ready for optimization.")

    arguments = (z, Alpha, rc, rl, Phi, Theta, args.Nt, args.Nl, args.Phi_50, args.P0_wavelength, args.k, args.right)
    bounds = define_bounds(args.Nt)

    # First pass optimization (no exclusions yet)
    excluded_indices = set()
    x_optim = least_squares(residuals, x0, args=(*arguments, excluded_indices), bounds=bounds)
    print("Optimization data (pass 1):", x_optim)
    print("Optimized vector (pass 1):", x_optim.x)

    for pass_num in range(2, 6):  # 5 passes (starting from pass 2 to 5)
        print(f"=== Pass {pass_num} ===")

        # Residuals after current optimization
        res = residuals(x_optim.x, *arguments, excluded_indices)
        res_flat = res.flatten()

        # Outlier detection
        mean_res = np.mean(res_flat)
        std_res = np.std(res_flat)
        threshold = mean_res + 2 * std_res

        new_outliers = np.where(np.abs(res_flat) > threshold)[0]
        print(f"Outlier threshold (pass {pass_num}): {threshold:.4f}")
        print(f"Found {len(new_outliers)} outliers out of {len(res_flat)} residuals.")

        if len(new_outliers) > 0:
            print(f"Outlier indices and residuals (pass {pass_num}):")
            for idx in new_outliers:
                print(f"Index {idx}: Residual = {res_flat[idx]:.4f}")

        # Update the cumulative set of excluded indices
        excluded_indices.update(new_outliers)

        # Re-run optimization using updated outlier exclusion set
        print(f"Re-running optimization excluding outliers (pass {pass_num}, max 5 iterations).")
        x_optim = least_squares(residuals, x_optim.x, args=(*arguments, excluded_indices), bounds=bounds, max_nfev=5)

        print(f"Optimization data (pass {pass_num}):", x_optim)
        print(f"Optimized vector (pass {pass_num}):", x_optim.x)

    print("Final optimized vector after 5 passes:", x_optim.x)

    print("=== OPTIMIZATION DONE : RECONSTRUCTING COLOR CORRECTED IMAGE ===")
    view_matrix, full_z = single_view_matrix_and_obs(mesh, transforms, images, args.image_height, args.image_width, args.image_idx, args.right, subset=None)
    ##view_matrix = l_view_matrix[args.image_idx] # alternate value for quick test until full_intensity
    ##full_z = z.T # alternate value for quick test until full_intensity
    print("Observation: ", full_z)
    visible_mesh, visible_indices =  collect_visible_mesh_and_indices(mesh, view_matrix)
    ##visible_mesh, visible_indices = random_mesh, random_indices # alternate value for quick test until full_intensity
    print("Visible triangles:", len(visible_indices))

    # Compute geometry for visible triangles
    view_cam_pos = cam_pos[args.image_idx]
    view_Nl = 4

    if not args.right or args.image_idx < len(images)//2:   
        view_lamp_pos = lamp_pos[args.image_idx]
        view_centerlines = centerlines[args.image_idx]
        view_transform = transforms[args.image_idx]
    else:
        view_lamp_pos = lamp_pos[args.image_idx - len(images)//2]
        view_centerlines = centerlines[args.image_idx - len(images)//2]
        view_transform = transforms[args.image_idx - len(images)//2]
        print("Warning, lamp, centerlines and transforms computed for right view. ")
    vis_t_midpoints = triangle_midpoints(visible_mesh)
    vis_t_normals = compute_triangle_normals(visible_mesh)
    print("Midpoints and normals computed for visible triangles.")
    vis_l_vectors = compute_single_view_lamp_vectors(vis_t_midpoints, view_lamp_pos)
    print("Lamp vectors computed for visible triangles. ")
    visPhi, visTheta = compute_single_view_angles(vis_t_normals, vis_l_vectors, view_centerlines, len(visible_indices), view_Nl, args.right)
    visAlpha = compute_single_view_alpha(vis_t_midpoints, len(visible_indices), view_cam_pos, view_transform, images, args.image_idx, args.right)
    print("Phi/Theta/Alpha computed for visible triangles.")
    vis_rc, vis_rl = compute_single_view_ranges(vis_t_midpoints, view_cam_pos, view_lamp_pos, visible_indices, view_Nl)
    print("Ranges computed for visible triangles.")

    full_albedo = compute_albedo(x_optim.x, full_z, visible_indices, visAlpha, args.Phi_50, view_Nl, args.P0_wavelength, args.k, vis_rc, vis_rl, visPhi, visTheta)
    print("Full albedo: ", full_albedo)
    full_air_intensities = air_intensities(full_albedo, args.image_idx, vis_t_normals, visible_indices, args.right)
    nan_count = np.sum(np.isnan(full_air_intensities))
    total_count = full_air_intensities.size
    print(f"NaN values in full_air_intensities: {nan_count}/{total_count}")
    recovered_image = recover_image(visible_mesh, images, args.image_idx, full_air_intensities, view_transform, args.right)
    cv2.imwrite("/Users/Sarah/Desktop/recovered_image.png", recovered_image)
    plt.imshow(recovered_image)
    plt.title("Recovered Image")
    plt.axis("off")
    plt.show()
