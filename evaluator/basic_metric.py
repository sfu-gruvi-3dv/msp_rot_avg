import numpy as np
from numpy.core.fromnumeric import amax, amin
import core_math.transfom as trans


def rel_R_deg(R1, R2):
    # relative pose
    rel_R = np.matmul(R1[:3, :3], R2[:3, :3].T)
    acos = np.arccos(np.clip((np.trace(rel_R) - 1) / 2, a_min=-1, a_max=1))
    R_err = np.rad2deg(acos)
    return R_err

def rel_sfm_t_err(t1, t2):
    t1 = t1 / np.linalg.norm(t1)
    t2 = t2 / np.linalg.norm(t2)
    
    dot = np.dot(t1, t2)
    dot = np.clip(dot, -1, 1)
    t_err = np.arccos(dot)
    return t_err

def rel_rot_quaternion_deg(q1, q2):
    """
    Compute relative error (deg) of two quaternion
    :param q1: quaternion 1, (w, x, y, z), dim: (4)
    :param q2: quaternion 2, (w, x, y, z), dim: (4)
    :return: relative angle in deg
    """
    return 2 * 180 * np.arccos(np.clip(np.dot(q1, q2), -1.0, 1.0)) / np.pi


def rel_rot_angle(T1, T2):
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    q1 = trans.quaternion_from_matrix(R1)
    q2 = trans.quaternion_from_matrix(R2)
    return rel_rot_quaternion_deg(q1, q2)


def rel_distance(T1, T2):
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    d = np.dot(R1.T, t1) - np.dot(R2.T, t2)
    return np.linalg.norm(d)