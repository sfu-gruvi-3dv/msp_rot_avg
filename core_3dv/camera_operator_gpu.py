import numpy as np
import torch
import torch.nn.functional as F
# from banet_track.ba_optimizer import batched_mat_inv
from core_3dv.mat_util import batched_mat_inv

""" Intrinsic Mat ------------------------------------------------------------------------------------------------------
"""
def scale_K(K:torch.Tensor, rescale_factor):
    K_mat = K.clone()
    if K_mat.dim() == 3:
        assert K_mat.shape[2] == 3
        assert K_mat.shape[1] == 3
        K_mat *= rescale_factor
        K_mat[:, 2, 2] = 1.0
    else:
        K_mat *= rescale_factor
        K_mat[2, 2] = 1.0
    return K_mat

def scale_K_xy(K:torch.Tensor, x_scale=1.0, y_scale=1.0):
    K_mat = K.clone()
    if K_mat.dim() == 3:
        assert K_mat.shape[2] == 3
        assert K_mat.shape[1] == 3
        K_mat[:, 0, 0] *= x_scale
        K_mat[:, 0, 2] *= x_scale
        K_mat[:, 1, 1] *= y_scale
        K_mat[:, 1, 2] *= y_scale
        K_mat[:, 2, 2] = 1.0
    else:
        K_mat[0, 0] *= x_scale
        K_mat[0, 2] *= x_scale
        K_mat[1, 1] *= y_scale
        K_mat[1, 2] *= y_scale
        K_mat[2, 2] = 1.0
    return K_mat

""" Camera Utilities ---------------------------------------------------------------------------------------------------
"""
def camera_center_from_Tcw(Rcw, tcw):
    """
    Compute the camera center from extrinsic matrix (world -> camera)
    :param R: Rotation matrix
    :param t: translation vector
    :return: camera center in 3D
    """
    # C = -Rcw' * t

    keep_dim_n = False
    if Rcw.dim() == 2:
        Rcw = Rcw.unsqueeze(0)
        tcw = tcw.unsqueeze(0)
    N = Rcw.shape[0]
    Rwc = torch.transpose(Rcw, 1, 2)
    C = -torch.bmm(Rwc, tcw.view(N, 3, 1))
    C = C.view(N, 3)

    if keep_dim_n:
        C = C.squeeze(0)
    return C


def translation_from_center(R, C):
    """
    convert center to translation vector, C = -R^T * t -> t = -RC
    :param R: rotation of the camera, dim (3, 3)
    :param C: center of the camera
    :return: t: translation vector
    """
    keep_dim_n = False
    if R.dim() == 2:
        R = R.unsqueeze(0)
        C = C.unsqueeze(0)
    N = R.shape[0]
    t = -torch.bmm(R, C.view(N, 3, 1))
    t = t.view(N, 3)

    if keep_dim_n:
        t = t.squeeze(0)
    return t


def camera_pose_inv(R, t):
    """
    Compute the inverse pose
    :param R: rotation matrix, dim (N, 3, 3) or (3, 3)
    :param t: translation vector, dim (N, 3) or (3)
    :return: inverse pose of [R, t]
    """
    keep_dim_n = False
    if R.dim() == 2:
        keep_dim_n = True
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)

    N = R.size(0)
    Rwc = torch.transpose(R, 1, 2)
    tw = -torch.bmm(Rwc, t.view(N, 3, 1))

    if keep_dim_n:
        Rwc = Rwc.squeeze(0)
        tw = tw.squeeze(0)

    return Rwc, tw


def transpose(R, t, X):
    """
    Pytorch batch version of computing transform of the 3D points
    :param R: rotation matrix in dimension of (N, 3, 3) or (3, 3)
    :param t: translation vector could be (N, 3, 1) or (3, 1)
    :param X: points with 3D position, a 2D array with dimension of (N, num_points, 3) or (num_points, 3)
    :return: transformed 3D points
    """
    keep_dim_n = False
    keep_dim_hw = False
    if R.dim() == 2:
        keep_dim_n = True
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
    if X.dim() == 2:
        X = X.unsqueeze(0)

    if X.dim() == 4:
        assert X.size(3) == 3
        keep_dim_hw = True
        N, H, W = X.shape[:3]
        X = X.view(N, H*W, 3)

    N = R.shape[0]
    M = X.shape[1]
    X_after_R = torch.bmm(R, torch.transpose(X, 1, 2))
    X_after_R = torch.transpose(X_after_R, 1, 2)
    trans_X = X_after_R + t.view(N, 1, 3).expand(N, M, 3)

    if keep_dim_hw:
        trans_X = trans_X.view(N, H, W, 3)
    if keep_dim_n:
        trans_X = trans_X.squeeze(0)

    return trans_X


def transform_mat44(R, t):
    """
    Concatenate the 3x4 mat [R, t] to 4x4 mat [[R, t], [0, 0, 0, 1]].
    :param R: rotation matrix, dim (N, 3, 3) or (3, 3)
    :param t: translation vector, dim (N, 3) or (3)
    :return: identical transformation matrix with dim 4x4
    """
    keep_dim_n = False
    if R.dim() == 2:
        keep_dim_n = True
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)

    N = R.shape[0]
    bot = torch.tensor([0, 0, 0, 1], dtype=torch.float).to(R.device).view((1, 1, 4)).expand(N, 1, 4)
    b = torch.cat([R, t.view(N, 3, 1)], dim=2)
    out_mat44 = torch.cat([b, bot], dim=1)
    if keep_dim_n:
        out_mat44 = out_mat44.squeeze(0)

    return out_mat44


def Rt(T):
    """
    Return the rotation matrix and the translation vector
    :param T: transform matrix with dim (N, 3, 4) or (N, 4, 4), 'N' can be ignored, dim (3, 4) or (4, 4) is acceptable
    :return: R, t
    """
    if T.dim() == 2:
        return T[:3, :3], T[:3, 3]
    elif T.dim() == 3:
        return T[:, :3, :3], T[:, :3, 3]
    else:
        raise Exception("The dim of input T should be either (N, 3, 3) or (3, 3)")


def relative_pose(R_A, t_A, R_B, t_B):
    """
    Computing the relative pose from
    :param R_A: frame A rotation matrix
    :param t_A: frame A translation vector
    :param R_B: frame B rotation matrix
    :param t_B: frame B translation vector
    :return: Nx3x3 rotation matrix, Nx3x1 translation vector that build a Nx3x4 matrix of T = [R,t]
    """
    keep_dim_n = False
    if R_A.dim() == 2 and t_A.dim() == 2:
        keep_dim_n = True
        R_A = R_A.unsqueeze(0)
        t_A = t_A.unsqueeze(0)
    if R_B.dim() == 2 and t_B.dim() == 2:
        R_B = R_B.unsqueeze(0)
        t_B = t_B.unsqueeze(0)

    N = R_A.shape[0]
    A_Tcw = transform_mat44(R_A, t_A)
    A_Twc = batched_mat_inv(A_Tcw)
    B_Tcw = transform_mat44(R_B, t_B)

    # Transformation from A to B
    T_AB = torch.bmm(B_Tcw, A_Twc)
    T_AB = T_AB[:, :3, :]

    if keep_dim_n is True:
        T_AB = T_AB.squeeze(0)

    return T_AB


""" Projections --------------------------------------------------------------------------------------------------------
"""
def pi(K, X):
    """
    Projecting the X in camera coordinates to the image plane
    :param K: camera intrinsic matrix tensor (N, 3, 3) or (3, 3)
    :param X: point position in 3D camera coordinates system, is a 3D array with dimension of (N, num_points, 3), or (num_points, 3)
    :return: N projected 2D pixel position u (N, num_points, 2) and the depth X (N, num_points, 1)
    """
    keep_dim_n = False
    keep_dim_hw = False
    if K.dim() == 2:
        keep_dim_n = True
        K = K.unsqueeze(0)      # make dim (1, 3, 3)
    if X.dim() == 2:
        X = X.unsqueeze(0)      # make dim (1, num_points, 3)
    if X.dim() == 4:
        assert X.size(3) == 3
        keep_dim_hw = True
        N, H, W = X.shape[:3]
        X = X.view(N, H*W, 3)

    assert K.size(0) == X.size(0)
    N = K.shape[0]

    X_x = X[:, :, 0:1]
    X_y = X[:, :, 1:2]
    X_z = X[:, :, 2:3]

    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    u_x = (fx * X_x + cx*X_z) / X_z
    u_y = (fy * X_y + cy*X_z) / X_z
    u = torch.cat([u_x, u_y], dim=-1)
    d = X_z

    if keep_dim_hw:
        u = u.view(N, H, W, 2)
        d = d.view(N, H, W)
    if keep_dim_n:
        u = u.squeeze(0)
        d = d.squeeze(0)

    return u, d


def pi_inv(K, x, d):
    """
    Projecting the pixel in 2D image plane and the depth to the 3D point in camera coordinate.
    :param x: 2d pixel position, a 2D array with dimension of (N, num_points, 2)
    :param d: depth at that pixel, a array with dimension of (N, num_points, 1)
    :param K: camera intrinsic matrix tensor (N, 3, 3)
    :return: 3D point in camera coordinate (N, num_points, 3)
    """
    keep_dim_n = False
    keep_dim_hw = False
    if K.dim() == 2:
        keep_dim_n = True
        K = K.unsqueeze(0)      # make dim (1, 3, 3)
    if x.dim() == 2:
        x = x.unsqueeze(0)      # make dim (1, num_points, 3)
    if d.dim() == 2:
        d = d.unsqueeze(0)      # make dim (1, num_points, 1)

    if x.dim() == 4:
        assert x.size(0) == d.size(0)
        assert x.size(1) == d.size(1)
        assert x.size(2) == d.size(2)
        assert x.size(3) == 2
        keep_dim_hw = True
        N, H, W = x.shape[:3]
        x = x.view(N, H*W, 2)
        d = d.view(N, H*W, 1)

    N = K.shape[0]
    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    X_x = d * (x[:, :, 0:1] - cx) / fx
    X_y = d * (x[:, :, 1:2] - cy) / fy
    X_z = d
    X = torch.cat([X_x, X_y, X_z], dim=-1)

    if keep_dim_hw:
        X = X.view(N, H, W, 3)
    if keep_dim_n:
        X = X.squeeze(0)

    return X


def x_2d_coords(h, w, n=None):
    N = 1 if n is None else n
    x_2d = np.zeros((N, h, w, 2), dtype=np.float32)
    for y in range(0, h):
        x_2d[:, y, :, 1] = y
    for x in range(0, w):
        x_2d[:, :, x, 0] = x
    x_2d = torch.Tensor(x_2d)
    if n is None:
        x_2d = x_2d.squeeze(0)

    return x_2d


def x_2d_normalize(h, w, x_2d):
    """
    Convert the x_2d coordinates to (-1, 1)
    :param x_2d: coordinates mapping, (N, H * W, 2)
    :return: x_2d: coordinates mapping, (N, H * W, 2), with the range from (-1, 1)
    """
    keep_dim = False
    if x_2d.dim() == 4:
        N, H, W, C = x_2d.shape
        x_2d = x_2d.view(N, H*W, C)
        keep_dim = True

    x_2d[:, :, 0] = (x_2d[:, :, 0] / (float(w) - 1.0))
    x_2d[:, :, 1] = (x_2d[:, :, 1] / (float(h) - 1.0))
    x_2d = x_2d * 2.0 - 1.0
    if keep_dim is True:
        x_2d = x_2d.view(N, H, W, C)
    return x_2d


def dense_corres_a2b(d_a, Ka, Kb, Ta, Tb, pre_cache_x2d=None, normalize=False):
    """
    Compute dense correspondence map from a to b.
    :param d_a: depth map of frame a, dim (N, H, W) or (H, W)
    :param K: camera intrinsic matrix, dim (N, 3, 3) or (3, 3)
    :param Ta: frame a camera pose, dim (N, 3, 4) or (3, 4)
    :param Tb: frame b camera pose, dim (N, 3, 4) or (3, 4)
    :param pre_cache_x2d: pre cached 2d coordinates, dim (N, H, W, 2)
    :return:
    """
    keep_dim_n = False
    if d_a.dim() == 2:
        keep_dim_n = True
        d_a = d_a.unsqueeze(0)
        Ta = Ta.unsqueeze(0)
        Ka = Ka.unsqueeze(0)
        Kb = Kb.unsqueeze(0)
    if Tb.dim() == 2:
        Tb = Tb.unsqueeze(0)

    N, H, W = d_a.shape
    d_a = d_a.view((N, H*W, 1))

    rel_Tcw = relative_pose(R_A=Ta[:, :3, :3], t_A=Ta[:, :3, 3],
                            R_B=Tb[:, :3, :3], t_B=Tb[:, :3, 3])
    if pre_cache_x2d is None:
        x_a_2d = x_2d_coords(h=H, w=W, n=N).to(d_a.device).view((N, H*W, 2))
    else:
        x_a_2d = pre_cache_x2d.view((N, H*W, 2))

    X_3d = pi_inv(K=Ka, x=x_a_2d, d=d_a.view((N, H*W, 1)))
    X_3d = transpose(R=rel_Tcw[:, :3, :3], t=rel_Tcw[:, :3, 3], X=X_3d)
    x_2d, _ = pi(K=Kb, X=X_3d)

    if normalize is True:
        x_2d = x_2d_normalize(h=H, w=W, x_2d=x_2d).view((N, H, W, 2))
    x_2d = x_2d.view((N, H, W, 2))

    if keep_dim_n is True:
        x_2d = x_2d.squeeze(0)

    return x_2d


def inv_dense_corres(dense_corres_a2b, pre_cache_x2d=None):
    raise Exception('NO IMPLEMENTATION')


def mark_out_bound_pixels(dense_corr_map, depth_map):
    """
    Mark out the out of boundary correspondence
    :param dense_corr_map: dense correspondence map, dim (N, H, W, 2) or dim (H, W, 2)
    :param depth_map: depth map, dim (N, H, W), (N, H*W), (N, H*W, 1) or dim (H, W)
    :return: 'out_area': the boolean 2d array indicates correspondence that is out of boundary, dim (N, H, W) or (H, W)
    """
    keep_dim_n = False
    if dense_corr_map.dim() == 3:
        keep_dim_n = True
        dense_corr_map = dense_corr_map.unsqueeze(0)
        depth_map = depth_map.unsqueeze(0)

    N, H, W = dense_corr_map.shape[:3]
    out_area_y = (dense_corr_map[:, :, :, 1] > H) | (dense_corr_map[:, :, :, 1] < 0)
    out_area_x = (dense_corr_map[:, :, :, 0] > W) | (dense_corr_map[:, :, :, 0] < 0)

    depth_mask = depth_map.view((N, H, W)) < 1e-5
    out_area = out_area_x | out_area_y
    out_area = out_area | depth_mask

    if keep_dim_n:
        out_area = out_area.squeeze(0)

    return out_area


def gen_overlap_mask_img(d_a, Ka, Kb, Ta, Tb, pre_cache_x2d=None):
    """
    Generate overlap mask of project a onto b
    :param d_a: depth map of frame a, dim (N, H, W) or (H, W)
    :param K: camera intrinsic matrix, dim (N, 3, 3) or (3, 3)
    :param Ta: frame a camera pose, dim (N, 3, 4) or (3, 4)
    :param Tb: frame b camera pose, dim (N, 3, 4) or (3, 4)
    :param pre_cache_x2d: pre cached 2d coordinates, dim (N, H, W, 2)
    :return: 'map':overlap mask; 'x_2d': correspondence
    """
    keep_dim_n = False
    if d_a.dim() == 2:
        keep_dim_n = True
        d_a = d_a.unsqueeze(0)
        Ta = Ta.unsqueeze(0)
        Ka = Ka.unsqueeze(0)
        Kb = Kb.unsqueeze(0)
    if Tb.dim() == 2:
        Tb = Tb.unsqueeze(0)

    N, H, W = d_a.shape
    d_a = d_a.view((N, H*W))

    rel_Tcw = relative_pose(Ta[:, :3, :3], Ta[:, :3, 3], Tb[:, :3, :3], Tb[:, :3, 3])
    if pre_cache_x2d is None:
        x_a_2d = x_2d_coords(H, W, N).to(d_a.device).view((N, H*W, 2))
    else:
        x_a_2d = pre_cache_x2d.view((N, H*W, 2))

    X_3d = pi_inv(Ka, x_a_2d, d_a.view((N, H*W, 1)))
    X_3d = transpose(rel_Tcw[:, :3, :3], rel_Tcw[:, :3, 3], X_3d)
    x_2d, corr_depth = pi(Kb, X_3d)

    x_2d = x_2d.view((N, H, W, 2))
    out_area = mark_out_bound_pixels(x_2d, d_a.view((N, H*W)))

    zeros = torch.zeros(out_area.size(), dtype=torch.float).to(d_a.device)
    ones = torch.ones(out_area.size(), dtype=torch.float).to(d_a.device)
    map = torch.where(out_area, zeros, ones)

    if keep_dim_n:
        map = map.squeeze(0)
        x_2d = x_2d.squeeze(0)

    return map.float(), x_2d

def gen_overlap_mask_img_by_corres(x_2d_a2b, d_a):
    N, H, W = d_a.shape
    x_2d = x_2d_a2b.view((N, H, W, 2))
    out_area = mark_out_bound_pixels(x_2d, d_a.view((N, H*W)))
    zeros = torch.zeros(out_area.size(), dtype=torch.float).to(d_a.device)
    ones = torch.ones(out_area.size(), dtype=torch.float).to(d_a.device)
    map = torch.where(out_area, zeros, ones)
    return map.float().detach()

def photometric_overlap(d_a, Ka, Kb, Ta, Tb, pre_cache_x2d=None):
    """
    Compute overlap ratio of project a onto b
    :param d_a: depth map of frame a, dim (N, H, W) or (H, W)
    :param K: camera intrinsic matrix, dim (N, 3, 3) or (3, 3)
    :param Ta: frame a camera pose, dim (N, 3, 4) or (3, 4)
    :param Tb: frame b camera pose, dim (N, 3, 4) or (3, 4)
    :param pre_cache_x2d: pre cached 2d coordinates, dim (N, H, W, 2)
    :return: overlap ratio, dim (N)
    """
    keep_dim_n = False
    if d_a.dim() == 2:
        keep_dim_n = True
        d_a = d_a.unsqueeze(0)
        Ta = Ta.unsqueeze(0)
        Ka = Ka.unsqueeze(0)
        Kb = Kb.unsqueeze(0)
    if Tb.dim() == 2:
        Tb = Tb.unsqueeze(0)

    N, H, W = d_a.shape
    d_a = d_a.view((N, H*W))

    rel_Tcw = relative_pose(Ta[:, :3, :3], Ta[:, :3, 3], Tb[:, :3, :3], Tb[:, :3, 3])
    if pre_cache_x2d is None:
        x_a_2d = x_2d_coords(H, W, N).to(d_a.device).view((N, H*W, 2))
    else:
        x_a_2d = pre_cache_x2d.view((N, H*W, 2))

    X_3d = pi_inv(Ka, x_a_2d, d_a.view((N, H*W, 1)))
    X_3d = transpose(rel_Tcw[:, :3, :3], rel_Tcw[:, :3, 3], X_3d)
    x_2d, corr_depth = pi(Kb, X_3d)

    x_2d = x_2d.view((N, H, W, 2))
    out_area = mark_out_bound_pixels(x_2d, d_a.view(N, H, W))

    non_zeros = torch.sum(out_area.view(N, -1), dim=1).float()
    total_valid_pixels = torch.sum(d_a > 1e-5, dim=1).float()

    ones = torch.ones_like(total_valid_pixels)
    out_ratio = torch.where(total_valid_pixels < 1e-6, ones, non_zeros / total_valid_pixels)
    in_ratio = torch.clamp(1.0 - out_ratio, 0.0, 1.0)

    if keep_dim_n is True:
        in_ratio = in_ratio.item()

    return in_ratio


def interp2d(tensor, x_2d, mode='bilinear'):
    """
    Interpolate the tensor, it will sample the pixel in input tensor by given the new coordinate (x, y) that indicates
    the position in original image.
    :param tensor: input tensor to be interpolated to a new tensor, (N, C, H, W)
    :param x_2d: new coordinates mapping, (N, H, W, 2) in (-1, 1), if out the range, it will be fill with zero
    :param mode: either 'bilinear' or 'nearest'
    :return: interpolated tensor
    """
    return F.grid_sample(tensor, x_2d, mode=mode)


def depth2scene(d, K, Rcw, tcw, pre_cache_x2d=None, in_chw_order=False):
    """
    Compute Scene Coordinate from depth and camera pose
    :param d: depth, dim=(N, H, W)
    :param K: camera intrinsic mat, dim=(N, 3, 3)
    :param Rcw: rotation mat, dim=(N, 3, 3)
    :param tcw: translation vector, dim=(N, 3)
    :param pre_cache_x2d: x coordinate grid, dim=(N, H, W, 2)
    :param in_chw_order: check if input output is (N, 3, H, W ) instead of (N, H, W, 3)
    :return:
    """
    # raise Exception("this func implementation has bug")

    keep_dim_n = False
    if d.dim() == 2:
        keep_dim_n = True
        d = d.unsqueeze(0)
        Rcw = Rcw.unsqueeze(0)
        tcw = tcw.unsqueeze(0)
        K = K.unsqueeze(0)

    N, H, W = d.shape
    if pre_cache_x2d is None:
        x_a_2d = x_2d_coords(n=N, h=H, w=W).to(d.device).view((N, H*W, 2))
    else:
        x_a_2d = pre_cache_x2d.view((N, H*W, 2))

    X_a_3d = pi_inv(K, x_a_2d, d.view((N, H * W, 1)))
    Rwc, twc = camera_pose_inv(Rcw, tcw)
    X_w_3d = transpose(Rwc, twc, X_a_3d)

    X_w_3d = X_w_3d.view(N, H, W, 3)
    if in_chw_order is True:
        X_w_3d = X_w_3d.permute(0, 3, 1, 2)

    if keep_dim_n:
        X_w_3d = X_w_3d.squeeze(0)

    return X_w_3d


def wrapping(I_b, d_a, Ka, Kb, Ta, Tb, pre_cache_x2d=None, in_chw_order=False):
    """
    Wrapping image from b to a
    :param I_b: image of frame b, dim (N, H, W, C) if 'in_chw_order=False' or dim (H, W, C)
    :param d_a: depth of frame a, dim (N, H, W) or (H, W)
    :param K: camera intrinsic matrix, dim (N, 3, 3) or (3, 3)
    :param Ta: frame a camera pose, dim (N, 3, 4) or (3, 4)
    :param Tb: frame b camera pose, dim (N, 3, 4) or (3, 4)
    :param pre_cache_x2d: pre cached 2d coordinates, dim (N, H, W, 2)
    :param in_chw_order: indicates the format order of 'I_b', either 'chw' if 'in_chw_order=False' or hwc
    :return: wrapped image from b to a, dim is identical to 'I_b'
    """
    keep_dim_n = False
    if I_b.dim() == 3:
        keep_dim_n = True
        I_b = I_b.unsqueeze(0)
        d_a = d_a.unsqueeze(0)
        Ta = Ta.unsqueeze(0)
        Tb = Tb.unsqueeze(0)
        Ka = Ka.unsqueeze(0)
        Kb = Kb.unsqueeze(0)
    if in_chw_order is False:
        I_b = I_b.permute(0, 3, 1, 2)

    N, C, H, W = I_b.shape
    d_a = d_a.view(N, H*W)

    rel_Tcw = relative_pose(Ta[:, :3, :3], Ta[:, :3, 3], Tb[:, :3, :3], Tb[:, :3, 3])
    if pre_cache_x2d is None:
        x_a_2d = x_2d_coords(H, W, N).to(d_a.device).view((N, H*W, 2))
    else:
        x_a_2d = pre_cache_x2d.view((N, H*W, 2))

    X_3d = pi_inv(Ka, x_a_2d, d_a.view((N, H*W, 1)))
    X_3d = transpose(rel_Tcw[:, :3, :3], rel_Tcw[:, :3, 3], X_3d)
    x_2d, _ = pi(Kb, X_3d)

    wrap_img_b = interp2d(I_b, x_2d_normalize(H, W, x_2d).view((N, H, W, 2)))                       # (N, H, W, 2)
    x_b_2d = x_2d.view((N, H, W, 2))

    if in_chw_order is False:
        wrap_img_b = wrap_img_b.permute(0, 2, 3, 1)

    if keep_dim_n:
        wrap_img_b = wrap_img_b.squeeze(0)
        x_b_2d = x_b_2d.squeeze(0)

    return wrap_img_b, x_b_2d

def reproj_err(K:torch.Tensor, Tcw:torch.Tensor, X_3d:torch.Tensor, x_2d:torch.Tensor):
    """
    compute re-projection error on scene coordinate maps
    Parameters
    ----------
    K: camera intrinsic matrix, dim: (N, 3, 3)
    Tcw: camera extrinsic matrix, dim: (N, 3, 4)
    X_3d: 3d points, dim (N, M, 3)
    x_2d: 2d coordinate on image plane, dim (N, M, 2)
    Returns
    -------
    reproj_err: re-projection error on 2D, dim (N, M)
    """
    assert X_3d.dim() == 3
    X_3d_C = transpose(Tcw[:, :3, :3], Tcw[:, :3, 3], X_3d)     # dim (N, M, 3)
    x_2d_c, _ = pi(K, X_3d_C)                                      # dim (N, M, 2)
    err = x_2d_c - x_2d                                         # dim (N, M, 2)
    return torch.norm(err, dim=-1)                              # dim (N, M)

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    # Example:
    #     >>> input = torch.rand(4, 3, 3)  # Nx3x4
    #     >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    # if not rotation_matrix.shape[-2:] == (3, 4):
    #     raise ValueError(
    #         "Input size must be a N x 3 x 4  tensor. Got {}".format(
    #             rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (1 - mask_d0_d1)
    mask_c2 = (1 - mask_d2) * mask_d0_nd1
    mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def rot2quaternion(R):
    """
    :param R: rotation matrix 3x3
    :return:  quaternion (w, x, y, z)
    """
    N = R.shape[0]
    diag = 1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q0 = torch.sqrt(diag) / 2.0
    q1 = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * q0)
    q2 = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * q0)
    q3 = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * q0)
    q = torch.stack([q0, q1, q2, q3], dim=1)
    q_norm = torch.sqrt(torch.sum(q*q, dim=1))
    return q / (q_norm.view(N, 1) + 1e-5)


def quaternion2rot(q):
    """
    [TESTED]
    :param q: normalized quaternion vector, dim: (N, 4)
    :return: rotation matrix, dim: (N, 3, 3)
    """
    N = q.shape[0]
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]
    return torch.stack([1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw,
                        2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw,
                        2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy
                        ], dim=1).view(N, 3, 3)