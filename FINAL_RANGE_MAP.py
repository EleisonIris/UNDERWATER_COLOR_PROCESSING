import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import json
from utils import r0_rd

def load_camera_params(json_path, scene_id):
    print("Loading camera parameters...")
    with open(json_path) as f:
        data = json.load(f)
    t_rot = np.array(data["0"][str(scene_id)]).flatten()
    rot, _ = cv2.Rodrigues(np.array(t_rot[:3], dtype=np.float64))
    t = np.array(t_rot[3:], dtype=np.float64)
    print("Camera parameters loaded.")
    return rot, t

def load_mesh_for_raytracing(mesh_path):
    print("Loading mesh...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    print("Mesh loaded and added to raycasting scene.")
    return scene, mesh_t

def compute_intersection(Y, rot, t, scene, cam):
    r0, rd = r0_rd(Y, rot, t, cam=cam)
    ray_start = r0
    ray_direction = -rd
    rays = o3c.Tensor(np.concatenate([ray_start.astype(np.float32), ray_direction.astype(np.float32)]), dtype=o3c.float32)
    rays = rays.reshape(o3c.SizeVector([1, 6]))
    intersection = scene.cast_rays(rays)
    t_hit = intersection["t_hit"].item()

    if np.isfinite(t_hit):
        hit_point = ray_start + t_hit * ray_direction
        distance = r0[2] - hit_point[2]
        return True, hit_point, distance, ray_start, ray_direction
    return False, None, None, ray_start, ray_direction

def generate_range_map(image_path, rot, t, scene, step=10, cam="l"):
    print("Loading image for range map...")
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError("Failed to load image at path: " + image_path)
    print("Image loaded.")

    height, width, _ = image.shape
    range_map = np.full((height, width), np.nan, dtype=np.float32)

    print("Starting range map computation...")
    for y in range(0, height, step):
        if y % 250 == 0:
            print(f"Processing row {y}/{height}")
        for x in range(0, width, step):
            Y = np.array([x, y], dtype=np.float64)
            hit, _, distance, *_ = compute_intersection(Y, rot, t, scene, cam=cam)
            if hit:
                range_map[y, x] = distance

    print("Range map computation finished.")
    return range_map

def save_range_map(range_map, output_path=None):
    print("Normalizing and saving range map as .txt...")
    valid = np.isfinite(range_map)
    normalized_range_map = np.copy(range_map)
    if np.any(valid):
        min_val = np.nanmin(normalized_range_map)
        max_val = np.nanmax(normalized_range_map)
        normalized_range_map[valid] = (
            255 - ((normalized_range_map[valid] - min_val) / (max_val - min_val) * 255)
        )
    if output_path:
        if not output_path.endswith(".txt"):
            output_path += ".txt"
        np.savetxt(output_path, normalized_range_map, fmt="%.4f")
        print(f"Range map saved as text at: {output_path}")


def save_rgbz_tiff(image_path, range_map, output_path):
    print("Preparing RGBZ TIFF...")
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError("Could not load image for RGBZ export.")
    height, width, _ = image.shape
    depth_normalized = np.nan_to_num(range_map, nan=0.0)
    depth_scaled = (depth_normalized).astype(np.uint16)
    if depth_scaled.shape != image.shape[:2]:
        raise RuntimeError("Image and depth map shapes do not match.")
    if image.dtype != np.uint8:
            image = image.astype(np.uint8)
    rgbz = np.zeros((height, width, 4), dtype=np.uint16)
    rgbz[:, :, :3] = image
    rgbz[:, :, 3] = depth_scaled
    success = cv2.imwrite(output_path, rgbz)
    if success:
        print(f"Saved RGBZ .tiff at {output_path}")
    else:
        print("Failed to save RGBZ TIFF.")

def rangemap_script(img_path, img_id, mesh_path, json_path, rangemap_path, tiff_path):
    print("Script started")
    
    rot, t = load_camera_params(json_path, img_id)
    scene, mesh_t = load_mesh_for_raytracing(mesh_path)
    
    range_map = generate_range_map(img_path, rot, t, scene, step=1)
    save_range_map(range_map, rangemap_path)
    save_rgbz_tiff(img_path, range_map, tiff_path)

    rgb_img = cv2.imread(img_path)
    if rgb_img is None:
        raise RuntimeError("Failed to load RGB image.")

