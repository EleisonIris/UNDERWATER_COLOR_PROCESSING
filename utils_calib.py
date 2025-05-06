import numpy as np
from numba import njit
import cv2
import networkx as nx



def undistort_rgb(rgb, K, dist):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    r_u = cv2.undistort(r, K, dist)
    g_u = cv2.undistort(g, K, dist)
    b_u = cv2.undistort(b, K, dist)

    return np.dstack((r_u, g_u, b_u))


def leaves(graph):
    deg = nx.degree(graph)
    return {node for node, degree in deg if degree == 1}


def find_root(tree):
    tree = tree.copy()

    while len(tree.nodes()) > 1:
        lvs = leaves(tree)
        for leaf in lvs:
            tree.remove_node(leaf)
            if len(tree.nodes()) <= 1:
                break

    return list(tree.nodes())[0]


def index_slice_to_bool_slice(index_slice, array_size):
    ret = np.zeros(array_size, dtype=bool)
    ret[index_slice] = True
    return ret


@njit(error_model='numpy')
def custom_rodrigues(rvec):
    theta = np.linalg.norm(rvec)

    if theta > 0.000001:
        rvec = (rvec / theta)

    rvec = rvec.reshape((3, 1))

    R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * rvec @ rvec.T + np.sin(theta) * np.array([[0, -rvec[2, 0], rvec[1, 0]], [rvec[2, 0], 0, -rvec[0, 0]], [-rvec[1, 0], rvec[0, 0], 0]])
    return R


@njit(error_model='numpy')
def normalize(v):
    return v / np.linalg.norm(v)


@njit(error_model='numpy')
def reprojection_matrix(pt2d_l, pt2d_r, cam_mat_l, cam_mat_r):
    """"compute the matrice to triangulate pt3d from pt2d matches"""
    pt_l_x, pt_l_y = pt2d_l[0], pt2d_l[1]
    pt_r_x, pt_r_y = pt2d_r[0], pt2d_r[1]
    col_l1, col_r1 = cam_mat_l[0, :], cam_mat_r[0, :]
    col_l2, col_r2 = cam_mat_l[1, :], cam_mat_r[1, :]
    col_l3, col_r3 = cam_mat_l[2, :], cam_mat_r[2, :]

    match_mat = np.empty((4, 4))
    match_mat[0] = pt_l_x * col_l3 - col_l1
    match_mat[1] = pt_l_y * col_l3 - col_l2
    match_mat[2] = pt_r_x * col_r3 - col_r1
    match_mat[3] = pt_r_y * col_r3 - col_r2

    return match_mat


@njit(error_model='numpy')
def reproject_point(match_matrix):
    """svd to triangulate one point"""
    _, S, right_normal_mat_transposed = np.linalg.svd(match_matrix)
    return right_normal_mat_transposed.transpose()[:, -1], S[-1]


@njit(error_model='numpy')
def _reconstruct_3d_points(points1, points2, PL, PR):
    """main function to triangulate 3d points from match"""

    pts3d = np.zeros((points1.shape[0], 4))
    errors = np.empty(points1.shape[0])

    for i in range(points1.shape[0]):
        pt2d_l, pt2d_r = points1[i], points2[i]
        reproj_matx = reprojection_matrix(
                pt2d_l, pt2d_r, PL, PR
                )
        pts3d[i], errors[i] = reproject_point(reproj_matx)
        pts3d[i] /= pts3d[i, 3]

    return pts3d[:, :3].copy(), errors


def reconstruct_3d_points(points1, points2, PL, PR, return_errors=False):
    points_3d, errors = _reconstruct_3d_points(points1, points2, PL, PR)
    if return_errors:
        return points_3d, errors

    return points_3d


@njit(error_model='numpy')
def rig_projection_matrices(KL, KR, Rc, Tc):
    PL = np.zeros((3, 4))
    PL[:3, :3] = np.eye(3)
    PL = KL @ PL

    PR = np.zeros((3, 4))
    PR[:3, :3] = Rc
    PR[:3, 3] = Tc
    PR = KR @ PR

    return PL, PR


def inverse_rot_trans(rot, trans):
    rot_mat, _ = cv2.Rodrigues(rot)
    new_rot_mat = rot_mat.copy().T
    new_trans = -new_rot_mat @ trans.copy()
    new_rot, _ = cv2.Rodrigues(new_rot_mat)

    return new_rot.flatten(), new_trans


def percentile_filter_3d_points(pts3d, threshold=(5, 95)):
    ref = np.percentile(pts3d, threshold, axis=0)
    mask_up = (pts3d > ref[0]).all(axis=1)
    mask_low = (pts3d < ref[1]).all(axis=1)

    return  mask_up & mask_low


# TODO dataclass?
class RigParameters:
    def __init__(self, KL, KR, Rc, Tc):
        self.KL = KL
        self.KR = KR
        self.Rc = Rc
        self.Tc = Tc

        self.PL, self.PR = rig_projection_matrices(self.KL, self.KR, self.Rc, self.Tc)


@njit(error_model="numpy", cache=True)
def closest_point_to_two_lines(ro1, rd1, ro2, rd2):
    b = ro2 - ro1

    d1_cross_d2 = np.cross(rd1, rd2)
    cross_norm2 = d1_cross_d2[0] * d1_cross_d2[0] + d1_cross_d2[1] * d1_cross_d2[1] + d1_cross_d2[2] * d1_cross_d2[2]

    t1 = np.linalg.det(np.array([
        [b[0], rd2[0], d1_cross_d2[0]],
        [b[1], rd2[1], d1_cross_d2[1]],
        [b[2], rd2[2], d1_cross_d2[2]]
    ])) / np.maximum(0.00001, cross_norm2)

    t2 = np.linalg.det(np.array([
        [b[0], rd1[0], d1_cross_d2[0]],
        [b[1], rd1[1], d1_cross_d2[1]],
        [b[2], rd1[2], d1_cross_d2[2]]
    ])) / np.maximum(0.00001, cross_norm2)

    p1 = ro1 + t1 * rd1
    p2 = ro2 + t2 * rd2

    return (p1 + p2) / 2.0, np.linalg.norm(p2 - p1)


@njit(error_model="numpy")
def trace_refract_ray(pixel_coord, air_K, distance, normal, thickness, eta_glass, eta_water):
    """
    assumes that `pixel_coord` was already undistorted
    """
    camera_rd = np.linalg.inv(air_K) @ np.array([pixel_coord[0], pixel_coord[1], 1])
    camera_rd /= np.linalg.norm(camera_rd)

    ## XXX ONLY ONE INTERFACE (air -> water)
    #c = normal @ camera_rd
    #r = 1.0 / eta_water
    #water_rd = r * camera_rd - (r * c - np.sqrt(1.0 - r*r * (1.0 - c*c))) * normal
    #water_rd /= np.linalg.norm(water_rd)
    #water_ro = camera_rd * (((np.array([0.0, 0.0, distance])) @ normal) / (camera_rd @ normal))

    # first intersection and refract, from inside the tube (air) to inside the flat port
    c = normal @ camera_rd
    r = 1 / eta_glass
    glass_rd = r * camera_rd - (r * c - np.sqrt(1 - r*r * (1 - c*c))) * normal
    glass_rd /= np.linalg.norm(glass_rd)
    glass_ro = camera_rd * (((np.array([0, 0, distance])) @ normal) / (camera_rd @ normal))

    # second intersection and refraction, from inside the flat port towards the water
    c = normal @ glass_rd
    r = eta_glass / eta_water
    water_rd = r * glass_rd - (r * c - np.sqrt(1 - r*r * (1 - c*c))) * normal
    water_rd /= np.linalg.norm(water_rd)
    water_ro = glass_ro + glass_rd * (((np.array([0, 0, distance]) + thickness * normal - glass_ro) @ normal) / (glass_rd @ normal))

    return water_ro, water_rd


@njit(error_model="numpy", cache=True)
def reconstruct_3d_point_refract(point_l, point_r, air_K_l, air_K_r, d_l, d_r, n_l, n_r, Rc, Tc, thickness, eta_glass, eta_water):
    """
    assumes that `point_l` and `point_r` were previously undistorted
    """
    #print(Rc, n_l, n_r, d_l, d_r)
    water_ro_l, water_rd_l = trace_refract_ray(point_l, air_K_l, d_l, n_l, thickness, eta_glass, eta_water)
    water_ro_r, water_rd_r = trace_refract_ray(point_r, air_K_r, d_r, n_r, thickness, eta_glass, eta_water)

    water_ro_r = Rc.T @ water_ro_r - Rc.T @ Tc
    water_rd_r = Rc.T @ water_rd_r

    return closest_point_to_two_lines(water_ro_l, water_rd_l, water_ro_r, water_rd_r)


