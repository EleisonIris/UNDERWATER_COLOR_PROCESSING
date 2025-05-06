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
sys.path.append('/Users/Sarah/Desktop')
from bryson_ambidextre import import_mesh, create_scene, compute_cam_pos, compute_lamp_pos, compute_centerlines, compute_lamp_vectors, compute_angles, compute_alpha, triangle_midpoints, compute_triangle_normals, select_random_triangles, find_images, load_transforms_from_file, load_images

@dataclass
class PipelineArgs:
    right: bool = True
    Nt: int = 1
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
        np.array([235, -20, 0]), #20 partout pour un close set autour de la caméra gauche (plus de visibilité)
        np.array([-20, -20, 0]), #235 en (1,1) et en (3,1) pour un cadre élargi autour des deux caméras (plus réaliste)
        np.array([235, 20, 0]),
        np.array([-20, 20, 0])
    ])
    image_height: int = 2000
    image_width: int = 3000
    image_idx: int = 0 # index of the image to reconstruct
    mesh_path: str = "/Users/Sarah/Desktop/sea-thru/mesh_cailloux_luca.ply"
    image_path_prefix: str = "scene"

if __name__ == "__main__":
    args = PipelineArgs() 

    print("=== Step 1: Importing mesh ===")
    mesh = import_mesh(args.mesh_path)
    scene = create_scene(mesh)
    print("Mesh loaded. Number of triangles:", len(mesh.triangles))

    # Load transforms and images
    transforms_file_path = '/Users/Sarah/Desktop/sea-thru/absolute_transforms_luca.json'
    transforms = load_transforms_from_file(transforms_file_path)
    args.transforms = transforms

    image_folder = '/Users/Sarah/Desktop/sea-thru/downsampled'
    images = load_images(image_folder, args.image_path_prefix, args.right)
    if images:
        print(f"Loaded {len(images)} images.")
    if transforms:
        print(f"Loaded {len(transforms)} transforms.") 

    random_mesh, random_indices = select_random_triangles(mesh, args.Nt)
    l_view_matrix, r_view_matrix, views_should_be = find_images(mesh, args.Nt, transforms, args.image_height, args.image_width, random_indices, args.right)

    # === Compute camera positions ===
    cam_positions = compute_cam_pos(transforms, args.right)
    N = cam_positions.shape[0] // 2
    left_cams = cam_positions[:N]
    right_cams = cam_positions[N:]
    def make_colored_pcd(points, base_color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = np.tile(base_color, (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    pcd_left = make_colored_pcd(left_cams, [1, 0.2, 0.2])  
    pcd_right = make_colored_pcd(right_cams, [0.2, 0.2, 1])
    lines = []
    colors = []
    for i in range(N):
        start_point = left_cams[i]
        end_point = right_cams[i]
        lines.append([i, i + N]) 
        colors.append([0, 1, 0])  
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.concatenate((left_cams, right_cams), axis=0)),
        lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)


    # === Compute lamp positions ===
    print("\n=== Step 2: Computing lamp positions ===")
    lamp_pos = compute_lamp_pos(transforms, args.lamp_transforms)

    lamp_spheres = []
    for i in range(len(transforms)):
        for j in range(args.Nl):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.8)
            sphere.translate(lamp_pos[i, j])
            sphere.paint_uniform_color([0.2, 0.3, 1]) 
            lamp_spheres.append(sphere)

    # === Compute centerlines ===
    print("\n=== Step 3: Computing centerlines ===")
    centerlines = compute_centerlines(transforms, ang_x=args.ang_x, ang_y=args.ang_y, Nl=args.Nl)

    ray_points = []
    ray_lines = []
    ray_colors = []
    idx = 0

    for cam_idx, (R, t) in enumerate(transforms):
        for lamp_idx in range(args.Nl):
            lamp_pos_current = lamp_pos[cam_idx, lamp_idx]  
            direction = centerlines[cam_idx, lamp_idx]  
            ray_end = lamp_pos_current + 100 * direction  
            ray_points.append(lamp_pos_current)
            ray_points.append(ray_end)
            ray_lines.append([idx, idx + 1])
            ray_colors.append([0, 1, 0])  
            idx += 2

    centerline_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(ray_points),
        lines=o3d.utility.Vector2iVector(ray_lines),
    )
    centerline_set.colors = o3d.utility.Vector3dVector(ray_colors)

    # === Compute triangle normals ===
    print("\n=== Step 4: Computing triangle normals ===")
    t_midpoints = triangle_midpoints(random_mesh)
    t_normals = compute_triangle_normals(random_mesh)

    # Visualize triangle normals as lines
    normal_lines = []
    normal_colors = []
    normal_points = []
    normal_scale = 1200  # Scale for visualization of normals

    for i, (midpoint, normal) in enumerate(zip(t_midpoints, t_normals)):
        start_point = midpoint
        end_point = midpoint + normal * normal_scale
        normal_points.extend([start_point, end_point])
        normal_lines.append([2 * i, 2 * i + 1])  
        normal_colors.append([1, 0, 0.5])  

    # Create line set for triangle normals
    normal_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(normal_points)),
        lines=o3d.utility.Vector2iVector(normal_lines)
    )
    normal_line_set.colors = o3d.utility.Vector3dVector(np.array(normal_colors))

    print("\n=== Step 5: Visualizing lamp vectors ===")
    lamp_vector_lines = []
    lamp_vector_colors = []
    lamp_vector_points = []
    index = 0

    for tri_idx, midpoint in enumerate(t_midpoints):
        for view_idx in range(len(transforms)):
            for lamp_idx in range(args.Nl):
                lamp_position = lamp_pos[view_idx, lamp_idx]
                lamp_vector_points.append(midpoint)
                lamp_vector_points.append(lamp_position)
                lamp_vector_lines.append([index, index + 1])
                lamp_vector_colors.append([1, 1, 0])  
                index += 2

    lamp_vector_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(lamp_vector_points)),
        lines=o3d.utility.Vector2iVector(np.array(lamp_vector_lines))
    )
    lamp_vector_set.colors = o3d.utility.Vector3dVector(np.array(lamp_vector_colors))

    print("\n=== Step 6: DEBUGGING Theta and Alpha ===")
    print("Computing lamp vectors...")
    lamp_vectors = compute_lamp_vectors(t_midpoints, lamp_pos)
    print("Computing angles Phi and Theta...")
    Phi, Theta = compute_angles(t_normals, lamp_vectors, centerlines, args.Nt, args.Nl, images, args.right)
    print("Computing Alpha (camera-to-triangle angles)...")
    world_axis, vec_to_camera, Alpha = compute_alpha(t_midpoints, cam_positions, transforms, args.right)
    print("\n--- DEBUG VALUES ---")
    print(f"Theta shape: {Theta.shape}, Alpha shape: {Alpha.shape}")
    print("Sample Theta values (degrees):", np.degrees(Theta[0, :3, :]))
    print("Sample Alpha values (degrees):", np.degrees(Alpha[0, :6]))
    print("\n--- INVALID VALUES CHECK ---")
    print("NaNs in Theta:", np.isnan(Theta).sum())
    print("NaNs in Alpha:", np.isnan(Alpha).sum())
    print("Theta out-of-range:", ((Theta < 0) | (Theta > np.pi/2)).sum())
    print("Alpha out-of-range:", ((Alpha < 0) | (Alpha > np.pi/2)).sum())
    Theta_deg = np.degrees(Theta.flatten())
    Alpha_deg = np.degrees(Alpha.flatten())
    Theta_deg = Theta_deg[~np.isnan(Theta_deg)]
    Alpha_deg = Alpha_deg[~np.isnan(Alpha_deg)]
    # Plot Theta distribution
    plt.figure(figsize=(10, 4))
    plt.hist(Theta_deg, bins=100, color='orange', alpha=0.7)
    plt.axvline(90, color='red', linestyle='--', label='π/2 (90°)')
    plt.title('Distribution of Theta Angles (degrees)')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Plot Alpha distribution
    plt.figure(figsize=(10, 4))
    plt.hist(Alpha_deg, bins=100, color='skyblue', alpha=0.7)
    plt.axvline(90, color='red', linestyle='--', label='π/2 (90°)')
    plt.title('Distribution of Alpha Angles (degrees)')
    plt.xlabel('Alpha (degrees)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Step 7: Visualize Theta vectors ===
    print("\n=== Step 7: Visualizing Theta values for lamp vectors ===")
    theta_arrow_lines = []
    theta_arrow_colors = []
    theta_arrow_points = []
    idx = 0
    for i, midpoint in enumerate(t_midpoints):
        for j in range(len(transforms)):  
            for k in range(args.Nl):  
                direction = lamp_vectors[i, j, k]
                norm_dir = direction / np.linalg.norm(direction)
                length = 1000 
                color_intensity = np.clip(Theta[i, j, k] / np.pi, 0, 1)
                color = [color_intensity, 0, 1 - color_intensity] 
                theta_arrow_points.append(midpoint)
                theta_arrow_points.append(midpoint - norm_dir * length)
                theta_arrow_lines.append([idx, idx + 1])
                theta_arrow_colors.append(color)
                idx += 2
    theta_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(theta_arrow_points)),
        lines=o3d.utility.Vector2iVector(np.array(theta_arrow_lines)),
    )
    theta_line_set.colors = o3d.utility.Vector3dVector(np.array(theta_arrow_colors))

    # === Step 8: Compute and visualize camera vectors ===
    print("\n=== Step 8: Visualizing camera vectors ===")
    cam_axis_world, cam_vectors, alpha_angles = compute_alpha(t_midpoints, cam_positions, transforms, args.right)
    print("Shape of cam_vectors:", cam_vectors.shape)
    print("Shape of camera axis world:", cam_axis_world.shape)
    cam_vec_lines = []
    cam_vec_colors = []
    cam_vec_points = []
    idx = 0
    vector_scale = 800
    for i, midpoint in enumerate(t_midpoints):
        for j, cam_pos in enumerate(cam_positions):
            cam_vec = cam_vectors[i, j]  
            cam_vec /= np.linalg.norm(cam_vec) 
            end_point = midpoint + cam_vec * vector_scale
            cam_vec_points.extend([midpoint, end_point])
            cam_vec_lines.append([idx, idx + 1])
            alpha_value = alpha_angles[i, j]
            intensity = np.clip(alpha_value / np.pi, 0, 1) 
            color = [intensity, 0, 1 - intensity]
            cam_vec_colors.append(color)
            idx += 2

    camera_vector_lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(cam_vec_points)),
        lines=o3d.utility.Vector2iVector(np.array(cam_vec_lines))
    )
    camera_vector_lines.colors = o3d.utility.Vector3dVector(np.array(cam_vec_colors))

    # === Visualize camera axes (in world space) ===
    axis_scale = 1000
    axis_lines = []
    axis_colors = []
    axis_points = []
    axis_idx = 0
    for i, (cam_pos, axis_dir) in enumerate(zip(cam_positions, cam_axis_world)):
        axis_dir = axis_dir / np.linalg.norm(axis_dir)  
        end_point = cam_pos + axis_dir * axis_scale
        axis_points.extend([cam_pos, end_point])
        axis_lines.append([axis_idx, axis_idx + 1])
        axis_colors.append([0, 0.2, 1])  
        axis_idx += 2
    camera_axis_lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(axis_points)),
        lines=o3d.utility.Vector2iVector(np.array(axis_lines))
    )
    camera_axis_lines.colors = o3d.utility.Vector3dVector(np.array(axis_colors))

    # === Final Visualization ===
    print("\n=== Displaying everything in one scene ===")
    o3d.visualization.draw_geometries([
    mesh,
    pcd_left, 
    pcd_right, line_set,
    *lamp_spheres,
    centerline_set,
    #lamp_vector_set,
    theta_line_set, normal_line_set,
    #camera_vector_lines,
    camera_axis_lines
    ])
