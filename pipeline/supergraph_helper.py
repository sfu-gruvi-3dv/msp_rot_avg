import numpy as np
import torch, os, random
import core_3dv.camera_operator as cam_opt

from core_3dv.quaternion import *
import graph_utils.utils as graph_utils
from graph_utils.graph_node import Node
from torch_scatter import scatter

torch.manual_seed(666)

""" Functions ----------------------------------------------------------------------------------------------------------
"""
def edge_model(x, edge_index):
    row, col = edge_index
    q_ij = qmul(x[col], inv_q(x[row]))
    return q_ij

def transform_q(pred, root):
    N = pred.shape[0]
    ref_ = pred[root: root +1, :].view(1, 4).repeat(N, 1)
    q_ref_to_p = qmul(pred, inv_q(ref_))
    # q_ref_to_p = inv_q(q_p_to_ref)
    return q_ref_to_p


def edge_model_batch(x, edge_index):
    """
    Args:
        x: pose, dim: (N, K, 4)
        edge_index: (2, E)
    Returns:
    """
    row, col = edge_index
    N, K = x.shape[:2]
    E = edge_index.shape[1]
    from_ = x[row, :, :].view(E*K, 4)
    to_ = x[col, :, :].view(E*K, 4)
    q_ij = qmul(to_, inv_q(from_)).view(E, K, 4)
    return q_ij

def rel_gt(edge_indices, cam_Es):

    gt_rel_Rts = list()
    for ei in range(len(edge_indices)):
        n1 = e_node_idx[ei][0].item()
        n2 = e_node_idx[ei][1].item()
        E_n1 = cam_Es[0, n1].cpu().numpy()
        E_n2 = cam_Es[0, n2].cpu().numpy()

        rel_Rt = cam_opt.relateive_pose(E_n1[:3, :3], E_n1[:3, 3], E_n2[:3, :3], E_n2[:3, 3])
        gt_rel_Rts.append(rel_Rt.reshape(1, 3, 4))

    gt_rel_Rts = np.vstack(gt_rel_Rts)
    return gt_rel_Rts

def load_mask(base_dir, dataset_name, itr):
    txt_file_path = os.path.join(base_dir, dataset_name[0], str(itr), 'EGs_less_20.txt')
    n1n2 = np.loadtxt(txt_file_path)[:, :2].astype(np.int32)
    edge_dict = dict()
    for n_i in range(n1n2.shape[0]):
        n = n1n2[n_i]
        edge_dict['%d-%d' % (n[0], n[1])] = True
    return edge_dict

def build_mask(valid_edge_dict, sub2id, e_node_idx):
    e_mask = np.zeros(len(e_node_idx))
    for ei in range(len(e_node_idx)):
        n1_ = e_node_idx[ei][0].item()
        n2_ = e_node_idx[ei][1].item()
        n1 = sub2id[n1_].item()
        n2 = sub2id[n2_].item()
        key1 = '%d-%d' % (n1, n2)
        key2 = '%d-%d' % (n2, n1)

        e_mask[ei] = 1.0 if key1 in valid_edge_dict or key2 in valid_edge_dict else 0
    return e_mask

def refine_mask(node_num, e_mask, e_node_idx):
    nodes_dict = dict()
    for i in range(e_mask.shape[0]):
        if e_mask[i] == 1.0:
            n1 = e_node_idx[i][0].item()
            n2 = e_node_idx[i][1].item()
            nodes_dict[n1] = True
            nodes_dict[n2] = True

    # check connection
    invalid_nodes = dict()
    for n in range(node_num):
        if n not in nodes_dict:
            invalid_nodes[n] = True
    print(invalid_nodes.keys())

    # add other edges
    invalid_nodes_edges = dict()
    for i in range(e_mask.shape[0]):
        n1 = e_node_idx[i][0].item()
        n2 = e_node_idx[i][1].item()
        if n1 in invalid_nodes and n2 in nodes_dict:
            if n1 not in invalid_nodes_edges:
                invalid_nodes_edges[n1] = [i]
            else:
                invalid_nodes_edges[n1].append(i)

        elif n2 in invalid_nodes and n1 in nodes_dict:
            if n2 not in invalid_nodes_edges:
                invalid_nodes_edges[n2] = [i]
            else:
                invalid_nodes_edges[n2].append(i)

    # add to e_mask
    for node_i, edges in invalid_nodes_edges.items():
        random.shuffle(edges)

        for e in edges[:3]:
            e_mask[e] = 0.5

    return e_mask, invalid_nodes

# spt
def build_dbg_spt(node_num, e_node_idx, start_node):
    adj_list = [[] for x in range(node_num)]
    for e_i, e in enumerate(e_node_idx):
        n1, n2 = e_node_idx[e_i][0].item(), e_node_idx[e_i][1].item()
        adj_list[n1] += [n2]
        adj_list[n2] += [n1]

    root = Node(start_node)
    q = [root]

    vis = [0 for x in range(node_num)]
    vis[root.attr] = 1

    while len(q) != 0:
        u = q.pop(0)
        random.shuffle(adj_list[u.attr])
        for v in adj_list[u.attr][:20]:
            if vis[v] != 1:
                node_v = Node(v)
                u.add_child(node_v)
                q.append(node_v)
                vis[v] = 1

    return root

def read_iccv13_res_(rot_txt_path, valid_cam_list=None):
    res = np.loadtxt(rot_txt_path)
    res[np.isnan(res)] = 0.0
    R_mat = res[:, 1:].reshape(res.shape[0], 3, 3)
    cam_id_list = res[:, 0]

    if valid_cam_list is not None:
        valid_cam_list = valid_cam_list.tolist()
        valid_R_mat = []
        for cam_i in range(cam_id_list.shape[0]):
            cam_id = cam_id_list[cam_i]
            if cam_id in valid_cam_list:
                valid_R_mat.append(R_mat[cam_i])
        R_mat = np.asarray(valid_R_mat)
        cam_id_list = np.asarray(valid_cam_list)

    q = rot2quaternion(torch.from_numpy(R_mat))
    return cam_id_list, q

def read_iccv13_res(num_nodes, rot_txt_path, id2sub, ref_root):
    res = np.loadtxt(rot_txt_path)
    res[np.isnan(res)] = 0.0
    # res[res > 1000] = 0.0
    N = num_nodes

    id = res[:, 0].astype(np.int)
    R_mat = res[:, 1:].reshape(res.shape[0], 3, 3)

    # locate ref id
    ref_id = -1
    for n in range(res.shape[0]):
        if id[n] not in id2sub:
            print('Ignored %d' % id[n])
            continue

        sub_id = id2sub[id[n]]
        if sub_id == ref_root:
            ref_id = n
            break

    if ref_id == -1:
        raise Exception('Ref Node not found')

    # R_mat = torch.from_numpy(R_mat).float()
    R_mat = graph_utils.rel_ref_2_R_(ref_id, R_mat)
    q = rot2quaternion(torch.from_numpy(R_mat))

    re_order_q = torch.zeros(N, 4)
    re_order_q[:, 0] = 1.0
    for n in range(res.shape[0]):
        if id[n] not in id2sub:
            continue

        sub_id = id2sub[id[n]]
        re_order_q[sub_id, :] = q[n, :]

    return re_order_q


""" Loss Function ------------------------------------------------------------------------------------------------------
"""
def weighted_l2_loss(input, w, edge_index, size, alpha=0.05):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])
    nn = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
    # beta = torch.squeeze(beta)
    nn = nn * beta.view(*nn.shape)

    cond = nn < alpha
    loss = torch.where(cond, 0.5 * nn ** 2 / alpha, nn - 0.5 * alpha)
    loss = scatter(loss, edge_index[0], dim=0, dim_size=size(0), reduce="mean")
    # out = scatter(edge_out, edge_index[i], dim=0,dim_size=size[i], reduce=self.aggr)

    return loss.mean()

def my_smooth_l1_loss(input, beta, alpha=0.05):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])
    nn = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
    nn = nn * beta

    cond = nn < alpha
    loss = torch.where(cond, 0.5 * nn ** 2 / alpha, nn - 0.5 * alpha)
    # if size_average:
    #     return loss.mean()
    # return loss.sum()
    return loss


def smooth_l1_loss(input, beta, edge_index, N, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])

    n = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
    n = beta * n
    loss = n

    # loss = scatter(loss, edge_index[0], dim=0, dim_size=N, reduce="mean")
    if size_average:
        return loss.mean()
    return loss.sum()

def my_smooth_l1_loss_new(input, beta, edge_index, size, alpha=0.05):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])
    nn = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
    # beta = torch.squeeze(beta)
    nn = nn * beta

    cond = nn < alpha
    loss = torch.where(cond, 0.5 * nn ** 2 / alpha, nn - 0.5 * alpha)
    loss = scatter(loss, edge_index[0], dim=0, dim_size=size(0), reduce="mean")
    # out = scatter(edge_out, edge_index[i], dim=0,dim_size=size[i], reduce=self.aggr)

    return loss.mean()

def opt_init_pose_loss(init_qs, gt_q, edge_node_idx, edge_rel_q):
    """
    Optimize the initial pose
    Args:
        init_qs:
        gt_q:
        edge_node_idx:
        edge_rel_q:

    Returns:

    """
    pass