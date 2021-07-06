# OpenSFM routines
import sys
sys.path.append('/mnt/Tango/pg/libs/OpenSfM/build/lib/')
from opensfm.reconstruction import two_view_reconstruction
from opensfm import pygeometry
from opensfm import features
from opensfm.types import PerspectiveCamera, BrownPerspectiveCamera
import numpy as np
import cv2
from core_3dv.essential_mat import triangulate_points

def build_perspective_cam(cam_id, K):
    cam = BrownPerspectiveCamera()
    cam.id = int(cam_id)
    cam.focal_x = K[0, 0]
    cam.focal_y = K[1, 1]
    cam.c_x = K[0, 2]
    cam.c_y = K[1, 2]
    cam.height = int(2*cam.c_y)
    cam.width = int(2*cam.c_x)
    cam.k1 = 0.0
    cam.k2 = 0.0
    cam.p1 = 0.0
    cam.p2 = 0.0
    cam.k3 = 0.0
    return cam

def Rt_from_E(E, pts1, pts2, K1, K2, refine_Rt=False, refine_iter=300):
    cam1 = build_perspective_cam(0, K1)
    cam2 = build_perspective_cam(1, K2)
    bts1 = cam1.pixel_bearing_many(pts1[:, :2])
    bts2 = cam2.pixel_bearing_many(pts2[:, :2])
    Rt = pygeometry.relative_pose_from_essential(E, bts1, bts2)
    if refine_Rt is True:
        Rt = pygeometry.relative_pose_refinement(Rt, bts1, bts2, refine_iter)
    return Rt

def Refine_Rt_from_E(E, pts1, pts2, K1, K2, refine_Rt=False, refine_iter=300):
    E = E.reshape(3, 3).astype(np.float64)
    pts1 = pts1.astype(np.float64)
    pts2 = pts2.astype(np.float64)

    cam1 = build_perspective_cam(0, K1)
    cam2 = build_perspective_cam(1, K2)
    bts1 = cam1.pixel_bearing_many(pts1[:, :2])
    bts2 = cam2.pixel_bearing_many(pts2[:, :2])
    Rt = pygeometry.relative_pose_from_essential(E, bts1, bts2)
    if refine_Rt is True:
        r_Rt = pygeometry.relative_pose_refinement(Rt, bts1, bts2, refine_iter)
    else:
        r_Rt = None
    return Rt, r_Rt

def decompose_E(E):
    R1, R2, t = cv2.decomposeEssentialMat(E)
    return R1, R2, t

def check_Rt(R1, R2, t, pts1, pts2, K1, K2):
    Rt_candidates = [np.hstack([R1, t]), np.hstack([R1, -t]), np.hstack([R2, t]), np.hstack([R2, -t])]
    best_depth_count = 0
    best_Rt = None
    for Rt_cand in Rt_candidates:
        # triangulate the points
        P1 = np.matmul(K1, np.eye(4)[:3, :])
        P2 = np.matmul(K2, Rt_cand)

        X_3d = triangulate_points(P1, P2, pts1[:, :2], pts2[:, :2])
        pos_depth_count = np.count_nonzero(X_3d[:, -1] > 0)
        if pos_depth_count > best_depth_count:
            best_Rt = Rt_cand
            best_depth_count = pos_depth_count

    return best_Rt