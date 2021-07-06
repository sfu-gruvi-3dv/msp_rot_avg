import cv2
import numpy as np
from scipy.linalg import logm, norm, inv

def triangulate_points(P1, P2, refined_pts1, refined_pts2):
    """Reconstructs 3D points by triangulation using Direct Linear Transformation."""
    # convert to 2xN arrays
    refined_pts1 = refined_pts1.T
    refined_pts2 = refined_pts2.T

    # triangulate inliers and keep only points that are in front of both cameras
    # homog_3D is a 4xN array of reconstructed points in homogeneous coordinates, pts_3D is a Nx3 array
    homog_3D = cv2.triangulatePoints(P1, P2, refined_pts1, refined_pts2)
    homog_3D = homog_3D / homog_3D[3]
    pts_3D = np.array(homog_3D[:3]).T

    return pts_3D

def five_point_essential_estimate(p1, p2, K, epiploar_threshold=3.0):
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=epiploar_threshold)
    _, R, t, _ = cv2.recoverPose(E, p1, p2, K)

    Rt = np.zeros((3, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t.ravel()

    return Rt # world -> camera

def normalize_translation(t):
    return t / norm(t)

def rel_pose_err(rel1, rel2):
    R1, R2 = rel1[:3, :3], rel2[:3, :3]
    t1, t2 = rel1[:3, 3], rel2[:3, 3]
    t1 = t1 / norm(t1)
    t2 = t2 / norm(t2)
    dot = np.dot(t1, t2)
    dot = np.clip(dot, -1, 1)
    R_err = norm(logm(np.matmul(R1.T, R2)))
    t_err = np.arccos(dot)
    return R_err, t_err

""" Converting
"""
def fundamental_to_essential_mat(F, K1, K2):
    """ E = K2' * F * K1;
    """
    E = np.matmul(K2.T, F)
    E = np.matmul(E, K1)
    return E

def essential_to_fundamental_mat(E, K1, K2):
    """ E = K2' * F * K1;
    """
    F = np.matmul(inv(K2).T, E)
    F = np.matmul(F, inv(K1))
    assert np.linalg.matrix_rank(F) == 2
    return F

def E_from_Rt(R, t):
    x, y, z = t[0], t[1], t[2]
    tx = [0, -z, y, z, 0, -x, -y, x, 0]
    tx = np.asarray(tx).astype(t.dtype)
    tx = tx.reshape(3, 3)
    E = np.matmul(R, tx)
    return E

def FE_from_Rt(R, t, K1, K2):
    E = E_from_Rt(R, t)
    F = essential_to_fundamental_mat(E, K1, K2)
    return F, E

""" Debug
"""
import random
import matplotlib.pyplot as plt
        
def draw_correspondences(img1, img2, pts1, pts2, limite_num=None, show=True, pt_size=4):


    img1_epi = img1.copy()
    img1_w = img1.shape[1]
    img2_epi = img2.copy()
    img2_w = img2.shape[1]

    if np.max(img1_epi) <= 1.0:
        img1_epi *= 255.0
    if np.max(img2_epi) <= 1.0:
        img2_epi *= 255.0

    line_pt_set = list(zip(pts1, pts2))
    if limite_num is not None:
        random.shuffle(line_pt_set)
        line_pt_set = line_pt_set[:limite_num]

    for (pt1, pt2) in line_pt_set:

        color = tuple(np.random.uniform(0, 255, 3).tolist())

        pt1 = pt1.astype(np.int32)
        pt2 = pt2.astype(np.int32)

        cv2.circle(img1_epi, tuple(pt1), pt_size, color, -1)
        cv2.circle(img2_epi, tuple(pt2), pt_size, color, -1)

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(20, 30))

        ax[0].imshow(img1_epi)
        ax[0].set_title('img #1')
        ax[1].imshow(img2_epi)
        ax[1].set_title('img #2')

        plt.show()
    else:
        return img1_epi, img2_epi

def draw_epipolar_lines(img1, img2, pts1, pts2, F, limite_num=None, draw_lines=True, show=True, pt_size=4, line_size=2):
    # check epipolar line
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    img1_epi = img1.copy()
    img1_w = img1.shape[1]
    img2_epi = img2.copy()
    img2_w = img2.shape[1]

    if np.max(img1_epi) <= 1.0:
        img1_epi *= 255.0
    if np.max(img2_epi) <= 1.0:
        img2_epi *= 255.0

    line_pt_set = list(zip(lines1, lines2, pts1, pts2))
    if limite_num is not None:
        random.shuffle(line_pt_set)
        line_pt_set = line_pt_set[:limite_num]

    for (r1, r2, pt1, pt2) in line_pt_set:

        color = tuple(np.random.uniform(0, 255, 3).tolist())

        r1_0 = (0, int(-r1[2] / r1[1]))
        r1_1 = (img1_w, int(-(r1[2] + r1[0] * img1_w) / r1[1]))
        r2_0 = (0, int(-r2[2] / r2[1]))
        r2_1 = (img2_w, int(-(r2[2] + r2[0] * img2_w) / r2[1]))

        if draw_lines:
            cv2.line(img1_epi, r1_0, r1_1, color, line_size)
            cv2.line(img2_epi, r2_0, r2_1, color, line_size)

        pt1 = pt1.astype(np.int32)
        pt2 = pt2.astype(np.int32)

        cv2.circle(img1_epi, tuple(pt1), pt_size, color, -1)
        cv2.circle(img2_epi, tuple(pt2), pt_size, color, -1)

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(20, 30))

        ax[0].imshow(img1_epi)
        ax[0].set_title('img #1')
        ax[1].imshow(img2_epi)
        ax[1].set_title('img #2')

        plt.show()
    else:
        return img1_epi, img2_epi

def verify_essential_mat(K1, K2, E, pt1, pt2):

    K1_inv = inv(K1)
    K2_inv = inv(K2)
    residules = []

    for i in range(pt1.shape[0]):
        pt1_v = [pt1[i][0], pt1[i][1], 1.0]
        pt2_v = [pt2[i][0], pt2[i][1], 1.0]

        pt1_v = np.asarray(pt1_v).reshape(3, 1)
        pt2_v = np.asarray(pt2_v).reshape(3, 1)

        pt1_v_norm = np.matmul(K1_inv, pt1_v)
        pt2_v_norm = np.matmul(K2_inv, pt2_v)

        residule = np.matmul(pt2_v_norm.reshape(1, 3), E)
        residule = np.matmul(residule, pt1_v_norm.reshape(3, 1))
        residules.append(residule)

    plt.hist(np.asarray(residules).ravel(), bins=50)
    plt.ylabel('Freq.')

def verify_fundamental_mat(F, pts1, pts2):

    dot_ps = []
    for (pt1, pt2) in zip(pts1, pts2):
        pt1_h = np.zeros(3, dtype=F.dtype)
        pt1_h[2] = 1.0
        pt1_h[:2] = pt1

        pt2_h = np.zeros(3, dtype=F.dtype)
        pt2_h[:2] = pt2
        pt2_h[2] = 1.0

        l = np.dot(F, pt1_h)
        dot_p = np.dot(pt2_h, l)
        dot_ps.append(dot_p)

    dot_ps = np.asarray(dot_ps)
    plt.hist(np.asarray(dot_ps).ravel(), bins=50)
    plt.ylabel('Freq.')