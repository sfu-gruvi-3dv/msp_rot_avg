import numpy as np
import torch
import core_3dv.camera_operator_gpu as cam_opt_gpu
import core_3dv.camera_operator as cam_opt

def compute_surface_normal(K, d, offset=1):
    H, W = d.shape
    d = d.reshape((H * W, 1))
    x_a = cam_opt.x_2d_coords(H, W).reshape((H * W, 2))
    X_3d = cam_opt.pi_inv(K, x_a, d)
    X_3d = X_3d.reshape((H, W, 3))

    dzdy = X_3d[2 * offset:, :, :] - X_3d[:-2 * offset, :, :]
    dzdx = X_3d[:, 2 * offset:, :] - X_3d[:, :-2 * offset, :]
    dxy = np.zeros((H, W, 2, 3), dtype=np.float32)
    dxy[offset:-offset, :, 1, :] = dzdy
    dxy[:, offset:-offset, 0, :] = dzdx
    n = np.zeros((H, W, 3), dtype=np.float32)
    n[offset:-offset, offset:-offset, :] = np.cross(dxy[:, :, 0, :], dxy[:, :, 1, :], axis=2)[offset:-offset,
                                           offset:-offset, :]
    norm = np.linalg.norm(n, axis=2)
    n /= norm[:, :, np.newaxis]
    return n

def extract_surf_normal(K, depth, offset=1, pre_cache_x2d=None):
    """
    Compute the surface normal in camera coordinate system from dense depth that projects to 3D point cloud
    :param K: camera intrinsic matrix, dim: (N, 3, 3) or (3, 3)
    :param depth: depth map, dim: (N, H, W) or (N, 1, H, W) or (H, W)
    :param offset: sampling offset along x-axis and y-axis
    :param pre_cache_x2d: pre-cached coordinate grid
    :return: extracted surface normal, dim: (N, H, W, 3)
    """
    keep_dim_n = False
    if depth.dim() == 3:
        N, H, W = depth.shape
    elif depth.dim() == 4:
        N, _, H, W = depth.shape
    elif depth.dim() == 2:
        keep_dim_n = True
        N = 1
        H, W = depth.shape
        K = K.unsqueeze(0)
        depth = depth.unsqueeze(0)

    if pre_cache_x2d is None:
        x_a_2d = cam_opt_gpu.x_2d_coords(H, W, N).to(depth.device).view((N, H*W, 2))
    else:
        x_a_2d = pre_cache_x2d.to(depth.device).view((N, H*W, 2))

    # compute the 3d point cloud in camera coordinate system
    depth = depth.view((N, H*W, 1))
    X_3d = cam_opt_gpu.pi_inv(K, x_a_2d, depth).view((N, H, W, 3))

    # compute changes along x-axis and y-axis
    dxy = torch.zeros((N, H, W, 2, 3)).float().to(depth.device)
    dxy[:, offset:-offset, :, 1, :] = X_3d[:, 2*offset:, :, :] - X_3d[:, :-2*offset, :, :]
    dxy[:, :, offset:-offset, 0, :] = X_3d[:, :, 2*offset:, :] - X_3d[:, :, :-2*offset, :]

    # cross-product of two changes along x-axis and y-axis, the normal should be perpendicular to
    # both vector
    n = torch.zeros((N, H, W, 3)).float().to(depth.device)
    n[:, offset:-offset, offset:-offset, :] = torch.cross(dxy[:, :, :, 0, :],
                                                          dxy[:, :, :, 1, :], dim=3)[:, offset:-offset, offset:-offset, :]

    # normalize surface normal
    norm = torch.norm(n, dim=3, keepdim=True)
    n = n / norm

    if keep_dim_n:
        n = n.squeeze(0)
    return n

def remove_nan_from_normalmap(normal_map, fill=torch.Tensor([0.0, 0.0, 1.0])):
    """
    Remove the surface normal that has nan components
    :param normal_map: surface normal map, dim: (N, H, W, 3)
    :param fill: the vec3f that fill the pixel with nan
    """
    nan_mask = torch.isnan(normal_map).any(dim=-1)
    fill = fill.to(normal_map.device)
    if (nan_mask == 1).any().item() == True:
        normal_map[nan_mask, :] = fill
    return normal_map


def azimuth_map(surface_normal):
    h = surface_normal.shape[0]
    w = surface_normal.shape[1]
    proj_norm = surface_normal[:, :, :2]
    x_axis = np.asarray((1.0, 0.0), dtype=np.float32)
    az_map = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h):
        for x in range(0, w):
            normal = proj_norm[y, x]
            az_map[y, x] = np.arccos(np.dot(normal, x_axis) / (np.linalg.norm(normal)))
    return np.rad2deg(az_map)
