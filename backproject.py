from utils import r0_rd, distance_X_to_D_r0_rd, get_image_data
import open3d as o3d
import numpy as np
import cv2

from scipy.optimize import least_squares
import time
import json
#import data.param_calib as param
import calib_luca
import matplotlib.pyplot as plt

def back_projeter(X,
                  R_cam=np.eye(3),      # rotation de la camera
                  t_cam=np.zeros((3,)), # translation de la camera
                  cam="l",              # type de camera (gauche ou droite)
                  max_cost=None) :
    """
    Renvoie Y, la projection inverse sur l'image d'un point X sur le mesh 3D
    **Input :**
        - X : numpy array [x, y, z], coordonnees dans $\mathbb{R^3}$
        - image_height, image_width : format de l'image
        - max_cost : cout maximal au dessus duquel rien ne sera renvoye (point hors de l'image)
    **Output :**
        - res.x : numpy array [x, y], coordonnees en pixel du point sur l'image
        - r0 et rd, le rayon qui emane de Y
    """
    r0, rd = None, None
    def f(Y) :     # fonction a minimiser
        nonlocal r0, rd
        r0, rd = r0_rd(Y, R_cam, t_cam, cam)
        cost = distance_X_to_D_r0_rd(X, r0, rd)
        return cost
    #Y0 = np.array([image_width // 2, image_height // 2], dtype=np.float64)  # point de depart
    Y0 = np.array([0., 0.], dtype=np.float64)
    # minimisation
    res = least_squares(f, Y0, loss="linear", verbose=0, ftol=1e-4, xtol=1e-2)
    if max_cost:
        if res.fun < max_cost :
            return res.x, r0, rd
        else : 
            return None
    else : 
        #print(f"erreur finale pour l'image {cam} : {res.fun}")
        return res.x, r0, rd
    
# --------------------------

if __name__ == "__main__" :
    mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux_luca_LOW.ply")
    # mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux_luca_high.ply")
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    mesh.compute_vertex_normals()

    flip = False
    image_id = 10
    rot,t = get_image_data(image_id)
    print(f"t = {t}, \n rot = {rot}")

    rot, t = rot, t

    image_path_l = f"downsampled/scene_l_00{str(image_id)}.jpeg"
    image_path_r = f"downsampled/scene_r_00{str(image_id)}.jpeg"
    image_l = cv2.imread(image_path_l)
    image_r = cv2.imread(image_path_r)
    h, w = image_l.shape[:2]

    points = np.asarray(mesh.vertices)

    N = 1
    indices = [np.random.randint(len(points)) for i in range(N)]
    colors = [[np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)] for i in range(N)]
    points_X = [points[idx] for idx in indices]

    print(f"Points : {points_X}")

    spheres = []
    for i in range(N) :
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
        sphere.translate(points_X[i])
        sphere.paint_uniform_color(colors[i])
        spheres.append(sphere)


    # gauche et droite
    Y_bests = []
    for i in range(N) :
        X_test = points_X[i]
        Y_best_l, r0_l, rd_l = back_projeter(X_test, rot, t, "l")
        Y_best_l = tuple(map(int, Y_best_l))  
        Y_best_r, r0_r, rd_r = back_projeter(X_test, rot, t, "r")
        Y_best_r = tuple(map(int, Y_best_r))  
        Y_bests.append((Y_best_l, r0_l, rd_l, Y_best_r, r0_r, rd_r))
        print(Y_best_r, Y_best_l)


    lignes_l = []
    lignes_r = []
    # tracer les rayons
    for i in range(N) :
        (Y_best_l, r0_l, rd_l, Y_best_r, r0_r, rd_r) = Y_bests[i]
        ligne_l = o3d.geometry.LineSet()
        ligne_l.points = o3d.utility.Vector3dVector([r0_l+1200*rd_l, r0_l +500 * rd_l])
        ligne_l.lines = o3d.utility.Vector2iVector([[0, 1]])
        ligne_l.colors = o3d.utility.Vector3dVector([colors[i]])
        ligne_r = o3d.geometry.LineSet()
        ligne_r.points = o3d.utility.Vector3dVector([r0_r+1200*rd_r, r0_r +500 * rd_r])
        ligne_r.lines = o3d.utility.Vector2iVector([[0, 1]])
        ligne_r.colors = o3d.utility.Vector3dVector([colors[i]])
        lignes_l.append(ligne_l)
        lignes_r.append(ligne_r)

    lignes_coins_l = []
    lignes_coins_r = []
    for x in [0, 2999] :
        for y in [0, 1999] :
            Y= np.array([x, y], dtype=np.float64)
            r0_coin_l, rd_coin_l = r0_rd(Y, rot, t, "l")
            r0_coin_r, rd_coin_r = r0_rd(Y, rot, t, "r")
            ligne_coin_l = o3d.geometry.LineSet()
            ligne_coin_l.points = o3d.utility.Vector3dVector([r0_coin_l+1200*rd_coin_l, r0_coin_l +500 * rd_coin_l])
            ligne_coin_l.lines = o3d.utility.Vector2iVector([[0, 1]])
            ligne_coin_l.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
            lignes_coins_l.append(ligne_coin_l)
            ligne_coin_r = o3d.geometry.LineSet()
            ligne_coin_r.points = o3d.utility.Vector3dVector([r0_coin_r+1200*rd_coin_r, r0_coin_r +500 * rd_coin_r])
            ligne_coin_r.lines = o3d.utility.Vector2iVector([[0, 1]])
            ligne_coin_r.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            lignes_coins_r.append(ligne_coin_r)

    image_with_point_l = image_l.copy()
    image_with_point_r = image_r.copy()

    for i in range(N) :
        (Y_best_l, r0_l, rd_l, Y_best_r, r0_r, rd_r) = Y_bests[i]
        color = tuple(map(lambda x: int(x * 255), colors[i][::-1]))
        cv2.circle(image_with_point_l, Y_best_l, radius=15, color=color, thickness=-1)
        cv2.circle(image_with_point_r, Y_best_r, radius=15, color=color, thickness=-1)

    if flip :
        cv2.imwrite("fichiers_test/Projection inverse_l.jpg", cv2.flip(image_with_point_l, 1))
        cv2.imwrite("fichiers_test/Projection inverse_r.jpg", cv2.flip(image_with_point_r, 1))
    else :
        cv2.imwrite("fichiers_test/Projection inverse_l.jpg",image_with_point_l)
        cv2.imwrite("fichiers_test/Projection inverse_r.jpg", image_with_point_r)
    
    o3d.visualization.draw_geometries([mesh]+spheres+lignes_l+lignes_r+lignes_coins_l+lignes_coins_r, window_name="Mesh avec point sélectionné")














# ------------------------


# if __name__ == "__main__" :
    
#     # nuage_de_pt = o3d.io.read_point_cloud("fichiers_ply/initial_cc_0.ply")

#     mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux.ply")
#     mesh.paint_uniform_color([0.5, 0.5, 0.5])
#     mesh.compute_vertex_normals()

#     flip = False
#     image_id = 10
#     r,t = get_image_data(image_id)
#     rot, _ =cv2.Rodrigues(r)
#     print(f"r = {r}, t = {t}, \n rot = {rot}")

#     rot, t = rot, t

#     image_path_l = f"downsampled/scene_l_00{str(image_id)}.jpeg"
#     image_path_r = f"downsampled/scene_r_00{str(image_id)}.jpeg"

#     image_l = cv2.imread(image_path_l)
#     image_r = cv2.imread(image_path_r)
#     h, w = image_l.shape[:2]

#     points = np.asarray(mesh.vertices)
#     idx = np.random.randint(len(points))
#     X_test = points[idx]
#     print(idx)
#     print(f"X_test = {X_test}")
#     triangles = np.asarray(mesh.triangles)
#     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
#     sphere.translate(X_test)
#     sphere.paint_uniform_color([1, 0, 0])

#     # points = np.asarray(nuage_de_pt.points)
#     # gray = np.array([[0.5, 0.5, 0.5]])  # Gris moyen
#     # colors = np.repeat(gray, len(points), axis=0)
#     # nuage_de_pt.colors = o3d.utility.Vector3dVector(colors)

#     # idx = np.random.randint(len(points))
#     # X_test = points[idx]
#     # print(idx)
#     # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
#     # sphere.translate(X_test)
#     # sphere.paint_uniform_color([1, 0, 0])


#     # gauche et droite
#     Y_best_l, r0_l, rd_l = back_projeter(X_test, h, w, rot, t, "l")
#     Y_best_l = tuple(map(int, Y_best_l))  
#     Y_best_r, r0_r, rd_r = back_projeter(X_test, h, w, rot, t, "r")
#     Y_best_r = tuple(map(int, Y_best_r))  

#     # tracer les rayons
#     ligne_l = o3d.geometry.LineSet()
#     ligne_l.points = o3d.utility.Vector3dVector([r0_l-1200*rd_l, r0_l -500 * rd_l])
#     ligne_l.lines = o3d.utility.Vector2iVector([[0, 1]])
#     ligne_l.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
#     ligne_r = o3d.geometry.LineSet()
#     ligne_r.points = o3d.utility.Vector3dVector([r0_r-1200*rd_r, r0_r -500 * rd_r])
#     ligne_r.lines = o3d.utility.Vector2iVector([[0, 1]])
#     ligne_r.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

#     lignes_coins_l = []
#     lignes_coins_r = []
#     for x in [0, 2999] :
#         for y in [0, 1999] :
#             Y= np.array([x, y], dtype=np.float64)
#             r0_coin_l, rd_coin_l = r0_rd(Y, rot, t, "l")
#             r0_coin_r, rd_coin_r = r0_rd(Y, rot, t, "r")
#             ligne_coin_l = o3d.geometry.LineSet()
#             ligne_coin_l.points = o3d.utility.Vector3dVector([r0_coin_l-1200*rd_coin_l, r0_coin_l -500 * rd_coin_l])
#             ligne_coin_l.lines = o3d.utility.Vector2iVector([[0, 1]])
#             ligne_coin_l.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
#             lignes_coins_l.append(ligne_coin_l)
#             ligne_coin_r = o3d.geometry.LineSet()
#             ligne_coin_r.points = o3d.utility.Vector3dVector([r0_coin_r-1200*rd_coin_r, r0_coin_r -500 * rd_coin_r])
#             ligne_coin_r.lines = o3d.utility.Vector2iVector([[0, 1]])
#             ligne_coin_r.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
#             lignes_coins_r.append(ligne_coin_r)

   



# if __name__ == "__main__" :
    
#     # nuage_de_pt = o3d.io.read_point_cloud("fichiers_ply/initial_cc_0.ply")

#     mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux.ply")
#     mesh.paint_uniform_color([0.5, 0.5, 0.5])
#     mesh.compute_vertex_normals()

#     flip = False
#     image_id = 10
#     r,t = get_image_data(image_id)
#     rot, _ =cv2.Rodrigues(r)
#     print(f"r = {r}, t = {t}, \n rot = {rot}")

#     rot, t = rot, t

#     image_path_l = f"downsampled/scene_l_00{str(image_id)}.jpeg"
#     image_path_r = f"downsampled/scene_r_00{str(image_id)}.jpeg"

#     image_l = cv2.imread(image_path_l)
#     image_r = cv2.imread(image_path_r)
#     h, w = image_l.shape[:2]

#     points = np.asarray(mesh.vertices)
#     idx = np.random.randint(len(points))
#     X_test = points[idx]
#     print(idx)
#     print(f"X_test = {X_test}")
#     triangles = np.asarray(mesh.triangles)
#     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
#     sphere.translate(X_test)
#     sphere.paint_uniform_color([1, 0, 0])

#     # points = np.asarray(nuage_de_pt.points)
#     # gray = np.array([[0.5, 0.5, 0.5]])  # Gris moyen
#     # colors = np.repeat(gray, len(points), axis=0)
#     # nuage_de_pt.colors = o3d.utility.Vector3dVector(colors)

#     # idx = np.random.randint(len(points))
#     # X_test = points[idx]
#     # print(idx)
#     # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
#     # sphere.translate(X_test)
#     # sphere.paint_uniform_color([1, 0, 0])


#     # gauche et droite
#     Y_best_l, r0_l, rd_l = back_projeter(X_test, h, w, rot, t, "l")
#     Y_best_l = tuple(map(int, Y_best_l))  
#     Y_best_r, r0_r, rd_r = back_projeter(X_test, h, w, rot, t, "r")
#     Y_best_r = tuple(map(int, Y_best_r))  

#     # tracer les rayons
#     ligne_l = o3d.geometry.LineSet()
#     ligne_l.points = o3d.utility.Vector3dVector([r0_l-1200*rd_l, r0_l -500 * rd_l])
#     ligne_l.lines = o3d.utility.Vector2iVector([[0, 1]])
#     ligne_l.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
#     ligne_r = o3d.geometry.LineSet()
#     ligne_r.points = o3d.utility.Vector3dVector([r0_r-1200*rd_r, r0_r -500 * rd_r])
#     ligne_r.lines = o3d.utility.Vector2iVector([[0, 1]])
#     ligne_r.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

#     lignes_coins_l = []
#     lignes_coins_r = []
#     for x in [0, 2999] :
#         for y in [0, 1999] :
#             Y= np.array([x, y], dtype=np.float64)
#             r0_coin_l, rd_coin_l = r0_rd(Y, rot, t, "l")
#             r0_coin_r, rd_coin_r = r0_rd(Y, rot, t, "r")
#             ligne_coin_l = o3d.geometry.LineSet()
#             ligne_coin_l.points = o3d.utility.Vector3dVector([r0_coin_l-1200*rd_coin_l, r0_coin_l -500 * rd_coin_l])
#             ligne_coin_l.lines = o3d.utility.Vector2iVector([[0, 1]])
#             ligne_coin_l.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
#             lignes_coins_l.append(ligne_coin_l)
#             ligne_coin_r = o3d.geometry.LineSet()
#             ligne_coin_r.points = o3d.utility.Vector3dVector([r0_coin_r-1200*rd_coin_r, r0_coin_r -500 * rd_coin_r])
#             ligne_coin_r.lines = o3d.utility.Vector2iVector([[0, 1]])
#             ligne_coin_r.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
#             lignes_coins_r.append(ligne_coin_r)



#     image_with_point_l = image_l.copy()
#     image_with_point_r = image_r.copy()

#     cv2.circle(image_with_point_l, Y_best_l, radius=15, color=(0, 0, 255), thickness=-1)
#     cv2.circle(image_with_point_r, Y_best_r, radius=15, color=(0, 0, 255), thickness=-1)

#     # plt.imshow(image_l[...,::-1]/255)
#     # plt.scatter(Y_best_l[0], Y_best_l[1], color='red')  
#     # plt.show()

#     if flip :
#         cv2.imwrite("fichiers_test/Projection inverse_l.jpg", cv2.flip(image_with_point_l, 1))
#         cv2.imwrite("fichiers_test/Projection inverse_r.jpg", cv2.flip(image_with_point_r, 1))
#     else :
#         cv2.imwrite("fichiers_test/Projection inverse_l.jpg",image_with_point_l)
#         cv2.imwrite("fichiers_test/Projection inverse_r.jpg", image_with_point_r)

#     o3d.visualization.draw_geometries([mesh, sphere, ligne_l, ligne_r]+lignes_coins_l+lignes_coins_r, window_name="Mesh avec point sélectionné")
#     #o3d.visualization.draw_geometries([nuage_de_pt, sphere, ligne_l, ligne_r]+lignes_coins_l+lignes_coins_r, window_name="Mesh avec point sélectionné")
