import open3d as o3d
import numpy as np
import cv2
#from data.param_calib import n_l, n_r
from calib_luca import n_l, n_r
from backprojection import back_projeter
from utils import get_image_data, r0_rd

def is_face_in_the_camera_direction(normale_face, normale_image, cos_theta_max:float=0.5) :
    """Renvoie True si la face est dans le meme sens que la camera depuis la vue donnee, False sinon""" 
    return np.dot(normale_face, normale_image) < cos_theta_max

def get_visible_faces(mesh,                         # mesh 3D
                      h, w,                         # taille de l'image
                      rot, t,                       # position du rig par rapport au repere monde
                      type_camera="l",              # type de camera ("l" ou "r")
                      cos_theta_max_pre_filtrage=0  # cos minimal de l'angle relatif entre les normales de la camera et de la face
                      ) :
    """
    Utilise le raycasting pour determiner les faces visibles depuis une vue
    Renvoie un np array de 0 et de 1 ou chaque case correspond a la visibilite d'une face du meme indice 
    /!\ Ne regarde que si le centre de chaque triangle est visible
    """
    n_cam = n_l if type_camera=="l" else n_r
    n_monde = rot.T @ n_cam 

    # chargement des faces
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices) # tableau de sommets (points dans R^3) : shape (N_ver, 3)
    triangles = np.asarray(mesh.triangles) # tableau de tableau de taille 3 : shape (N_tri, 3)
    # Chaque tableau [v1, v2, v3] represente un triangle, de sommets d'indices v1, v2 et v3 dans vertices
    N_tri = len(triangles)
    triangle_normals = np.asarray(mesh.triangle_normals) # tableau de vecteurs : shape (N_tri, 3)
    are_triangles_visible = np.zeros(N_tri, dtype=int)
   
    # -- Etape 1 -- filtration par extraction des faces dans la bonne direction
    # -- Etape 2 -- Retro-projection du centre de chaque triangle dans la bonne direction
    rays_to_faces = np.zeros((N_tri, 6), dtype=np.float64) # pour chaque triangle, r0 et rd seront stockes
    # rays_to_faces contiendra r0=0,rd=0 pour les triangles mal orientes
    for i in range(N_tri) :
        # filtration par extraction des faces dans la bonne direction
        normal = triangle_normals[i, :]
        if is_face_in_the_camera_direction(normal, n_monde, cos_theta_max_pre_filtrage) : 
            tri = triangles[i]
            X_center = vertices[tri].mean(axis=0) # centre de la face
            _, r0, rd = back_projeter(X_center,rot, t, type_camera) # retro-projection
            rays_to_faces[i, :] = np.concatenate([np.array(r0), np.array(rd)])  

    # -- Etape 3 -- Ray-tracing de chaque rayon visant une face
    # creation de la scene de raytracing
    raycasting_scene = o3d.t.geometry.RaycastingScene()
    vertices_tensor = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    triangles_tensor = o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32)
    raycasting_scene.add_triangles(vertices_tensor, triangles_tensor)

    # mask pour ne garder que les triangles bien orientes
    ray_norms = np.linalg.norm(rays_to_faces, axis=1)  # norme globale du vecteur [r0|rd]
    valid_rays_mask = ray_norms > 0
    valid_rays = rays_to_faces[valid_rays_mask]
    valid_face_indices = np.nonzero(valid_rays_mask)[0]  # pour remapper apres
    rays_tensor = o3d.core.Tensor(valid_rays, dtype=o3d.core.Dtype.Float32)
    hits = raycasting_scene.cast_rays(rays_tensor)

    # -- Etape 4 -- Garder les faces visibles d'apres le raycasting
    for i, hit in enumerate(hits["primitive_ids"].numpy()): # hits["primitive_ids"] : le tableau des faces touchees en premier de chaque hit
        original_face_idx = valid_face_indices[i]
        if hit == original_face_idx:
            are_triangles_visible[original_face_idx] = 1

    return are_triangles_visible
    
def get_selected_visible_faces(mesh,
                      h, w,
                      rot, t,
                      type_camera="l",
                      cos_theta_max_pre_filtrage=0,
                      triangle_indices_subset=None
                      ):
    """
    Utilise le raycasting pour determiner la visibilite des faces d’un sous-ensemble depuis une vue.
    Renvoie un tableau de 0 et 1 correspondant uniquement aux indices du sous-ensemble.
    """
    n_cam = n_l if type_camera == "l" else n_r
    n_monde = rot.T @ n_cam

    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)
    N_tri = len(triangles)

    # Handle full triangle list vs subset
    if triangle_indices_subset is None:
        triangle_indices_subset = np.arange(N_tri)
    else:
        triangle_indices_subset = np.array(triangle_indices_subset)

    Nt_subset = len(triangle_indices_subset)
    are_triangles_visible_subset = np.zeros(Nt_subset, dtype=int)

    # Precompute retro-projection rays for subset
    rays_to_faces = np.zeros((Nt_subset, 6), dtype=np.float64)
    for i, tri_idx in enumerate(triangle_indices_subset):
        normal = triangle_normals[tri_idx]
        if is_face_in_the_camera_direction(normal, n_monde, cos_theta_max_pre_filtrage):
            tri = triangles[tri_idx]
            X_center = vertices[tri].mean(axis=0)
            _, r0, rd = back_projeter(X_center, rot, t, type_camera)
            rays_to_faces[i, :] = np.concatenate([np.array(r0), np.array(rd)])

    # Raycasting setup (uses full mesh)
    raycasting_scene = o3d.t.geometry.RaycastingScene()
    vertices_tensor = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    triangles_tensor = o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32)
    raycasting_scene.add_triangles(vertices_tensor, triangles_tensor)

    # Select valid rays only
    ray_norms = np.linalg.norm(rays_to_faces, axis=1)
    valid_rays_mask = ray_norms > 0
    valid_rays = rays_to_faces[valid_rays_mask]
    valid_subset_indices = np.nonzero(valid_rays_mask)[0]
    triangle_ids_in_subset = triangle_indices_subset[valid_subset_indices]

    rays_tensor = o3d.core.Tensor(valid_rays, dtype=o3d.core.Dtype.Float32)
    hits = raycasting_scene.cast_rays(rays_tensor)

    # Check if the ray hits its intended triangle
    for i, hit in enumerate(hits["primitive_ids"].numpy()):
        if hit == triangle_ids_in_subset[i]:
            are_triangles_visible_subset[valid_subset_indices[i]] = 1

    return are_triangles_visible_subset


def reconstruct_visible_mesh(original_mesh,
                             are_triangles_visible) :
    """
    Reconstruit a partrir du tableau des faces visibles, le mesh contenant uniquement les faces visibles 
    Utile pour la visualisation
    """
    triangles = np.asarray(original_mesh.triangles)
    vertices = np.asarray(original_mesh.vertices)
    visible_faces_indices = np.where(are_triangles_visible == 1)[0]
    visible_triangles = triangles[visible_faces_indices]

    used_vertices_indices = np.unique(visible_triangles)
    new_vertices = vertices[used_vertices_indices]

    index_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices_indices)}
    remapped_triangles = np.vectorize(index_remap.get)(visible_triangles)

    visible_mesh = o3d.geometry.TriangleMesh()
    visible_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    visible_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
    visible_mesh.compute_vertex_normals()

    return visible_mesh

def generate_view_matrix(mesh, Nt, transforms, h, w, type_camera, subset) :
    """
    Genere la matrice Mij ou i represente la face, et j la vue
    Mij = 1 si la face i est visible sur la vue j, 0 sinon
    **INPUT** : 
    - mesh : le mesh 3D contenant les triangles
    - transforms : le tableau des transformations des vues : transforms[j-1] = (rot_j, t_j)
    - h, w : les dimensions d'une image
    - type_camera : "l" pour la camera de gauche, "r" pour la camera de droite
    """
    triangles = np.asarray(mesh.triangles)
    N_tri = Nt
    N_views = len(transforms)
    Mij = np.zeros((N_tri, N_views))
    for j in range(N_views) :
        rot, t = transforms[j]
        Mij[:, j] = get_selected_visible_faces(mesh, h, w, rot, t, type_camera, triangle_indices_subset=subset)
        print(f"Vue {j+1}/{N_views} terminee")
    return Mij

def generate_view_vector(mesh, image_idx, Nt, transforms, h, w, type_camera, subset) :
    triangles = np.asarray(mesh.triangles)
    N_tri = Nt
    Mi = np.zeros(N_tri)
    if image_idx < len(transforms):
        rot, t = transforms[image_idx]
    else:
        rot, t = transforms[image_idx//2]
    Mi = get_selected_visible_faces(mesh, h, w, rot, t, type_camera, triangle_indices_subset=subset)
    print(f"View vector computed for your view. ")
    return Mi
