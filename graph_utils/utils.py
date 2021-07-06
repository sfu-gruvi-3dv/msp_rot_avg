from evaluator.basic_metric import rel_R_deg
from graph_utils.mstpy.node_base import node_base
from graph_utils.mstpy.edge_base import edge_base
from graph_utils.mstpy.mst_base import mst_base
from graph_utils.graph_node import Node
import core_3dv.camera_operator_gpu as cam_opt_gpu
import torch
import numpy as np
import random

def build_prob_spt(node_num, e_node_idx, probs):

    prob_dict = {}
    for e_i, e in enumerate(e_node_idx):
        n1, n2 = e_node_idx[e_i][0].item(), e_node_idx[e_i][1].item()
        prob_dict[(n1, n2)] = probs[e_i].item()

    # build inlier SPT
    start_node = choose_anchor_node(node_num, e_node_idx, probs)
    mst, node_list = gen_max_mst(e_node_idx, probs, start_node)

    adj_list = [[] for x in range(node_num)]
    for e_i, e in enumerate(mst):
        n1, n2 = e.u.item(), e.v.item()
        adj_list[n1] += [n2]
        adj_list[n2] += [n1]

    root = Node(start_node)
    q = [root]

    vis = [0 for x in range(node_num)]
    vis[root.attr] = 1

    while len(q) != 0:
        u = q.pop(0)
        for v in adj_list[u.attr]:
            if vis[v] != 1:
                node_v = Node(v)
                u.add_child(node_v)
                q.append(node_v)
                vis[v] = 1

    return root

def build_inlier_spt(node_num, e_node_idx, e_label):

    # build inlier SPT
    inlier_adj_list = [[] for x in range(node_num)]
    for e_i, e in enumerate(e_node_idx):
        n1, n2 = e_node_idx[e_i][0].item(), e_node_idx[e_i][1].item()
        if e_label[0][e_i].item() == 1:
            inlier_adj_list[n1] += [n2]
            inlier_adj_list[n2] += [n1]

    inlier_start_node = 0
    for ns in inlier_adj_list:
        adj_ns = len(ns)
        if adj_ns > inlier_start_node:
            inlier_start_node = adj_ns

    root = Node(inlier_start_node)
    q = [root]

    vis = [0 for x in range(node_num)]
    vis[root.attr] = 1

    while len(q) != 0:
        u = q.pop(0)
        for v in inlier_adj_list[u.attr]:
            if vis[v] != 1:
                node_v = Node(v)
                u.add_child(node_v)
                q.append(node_v)
                vis[v] = 1

    return root

def build_random_spt(node_num, e_node_idx, e_label):
    # build inlier SPT
    inlier_adj_list = [[] for x in range(node_num)]
    adj_list = [[] for x in range(node_num)]
    for e_i, e in enumerate(e_node_idx):
        n1, n2 = e_node_idx[e_i][0].item(), e_node_idx[e_i][1].item()
        if e_label[0][e_i].item() == 1:
            inlier_adj_list[n1] += [n2]
            inlier_adj_list[n2] += [n1]
        adj_list[n1] += [n2]
        adj_list[n2] += [n1]

    inlier_start_node = 0
    for ns in inlier_adj_list:
        adj_ns = len(ns)
        if adj_ns > inlier_start_node:
            inlier_start_node = adj_ns

    root = Node(inlier_start_node)
    q = [root]

    vis = [0 for x in range(node_num)]
    vis[root.attr] = 1

    while len(q) != 0:
        u = q.pop(0)
        random.shuffle(adj_list[u.attr])
        for v in adj_list[u.attr][:5]:
            if vis[v] != 1:
                node_v = Node(v)
                u.add_child(node_v)
                q.append(node_v)
                vis[v] = 1

    return root

def count_tree_nodes(root):
    q = [root]
    node_count = 0
    while len(q) != 0:
        u = q.pop(0)
        node_count += 1
        q += [v for v in u.childrens]
    return node_count

def non_spt_nodes_list(num_nodes, root):
    q = [root]
    valid_list = []
    while len(q) != 0:
        u = q.pop(0)
        q += [v for v in u.childrens]
        valid_list += [v.attr for v in u.childrens]

    invalid_list = []
    for i in range(num_nodes):
        if i not in valid_list:
            invalid_list.append(i)
    return invalid_list

def compute_inlier_spt(num_nodes, e_node_idx, edge_label):

    # build inlier SPT
    inlier_adj_list = [[] for x in range(num_nodes)]
    for e_i, e in enumerate(e_node_idx):
        n1, n2 = e_node_idx[e_i][0].item(), e_node_idx[e_i][1].item()
        if edge_label[0][e_i].item() == 1:
            inlier_adj_list[n1] += [n2]
            inlier_adj_list[n2] += [n1]

    inlier_start_node = 0
    for ns in inlier_adj_list:
        adj_ns = len(ns)
        if adj_ns > inlier_start_node:
            inlier_start_node = adj_ns

    root = Node(inlier_start_node)
    q = [root]

    vis = [0 for x in range(num_nodes)]
    vis[root.attr] = 1

    while len(q) != 0:
        u = q.pop(0)
        for v in inlier_adj_list[u.attr]:
            if vis[v] != 1:
                node_v = Node(v)
                u.add_child(node_v)
                q.append(node_v)
                vis[v] = 1

    return root

def rel_ref_2_R_(ref_id, cam_R_mats):
    N, _, _ = cam_R_mats.shape
    ref_cam_Rs = []
    ref_R = cam_R_mats[ref_id]
    for e_i in range(N):
        R = cam_R_mats[e_i]
        rel_ref_R = np.matmul(R, ref_R.T)
        ref_cam_Rs.append(rel_ref_R.reshape(1, 3, 3))

    ref_cam_Rs = np.concatenate(ref_cam_Rs)
    return ref_cam_Rs

def rel_ref_2_E_(ref_id, cam_Es):
    _, N, _, _ = cam_Es.shape

    ref_cam_Es = []
    ref_E = cam_Es[0:1, ref_id]
    for e_i in range(N):
        E = cam_Es[0:1, e_i]
        rel_ref_E = cam_opt_gpu.relative_pose(ref_E[:, :3, :3], ref_E[:, :3, 3], E[:, :3, :3], E[:, :3, 3])
        ref_cam_Es.append(rel_ref_E.unsqueeze(0))

    return ref_cam_Es

class edge_mst_greater(edge_base):
    def __init__(self, u, v, dist=0, value=None):
        super(edge_mst_greater, self).__init__(u,v,dist,value)
        
    def __lt__(self, other):
        """
        less operation used in sorting.
        you need to implement this function in your child edge class
        """
        # TODO: implement lt function
        return self.dist > other.dist


def gen_max_mst(edgelist, probs, start_node=0):
    mst_graph = mst_base()
    node_list = []
    maxx_node_id = 0 
    minn_node_id = 100000000
    for edge in edgelist:
        maxx_node_id = max(maxx_node_id, edge[0])
        maxx_node_id = max(maxx_node_id, edge[1])
        minn_node_id = min(minn_node_id, edge[0])
        minn_node_id = min(minn_node_id, edge[1])

    for idx in range(minn_node_id, maxx_node_id+1):
        mst_graph.node_list.append(idx)

    for idx in range(len(edgelist)):
        edge = edgelist[idx]
        prob = probs[idx]
        mst_graph.edge_list.append(edge_mst_greater(edge[0],edge[1], prob))

    mst = mst_graph.generate_mst()
    return mst, mst_graph.node_list

def pose_mul(A, B):
    A = cam_opt_gpu.transform_mat44(A)
    B = cam_opt_gpu.transform_mat44(B)
    return torch.bmm(A, B)[3:, :]

def build_rel_R_dict(e_node_idx, e_rel_Rt):
    # relative pose
    rRs_dict = dict()
    for ei, (n1, n2) in enumerate(e_node_idx):
        n1, n2 = e_node_idx[ei][0].item(), e_node_idx[ei][1].item()

        rR = e_rel_Rt[ei]
        rR_inv, _ = cam_opt_gpu.camera_pose_inv(rR[0, :3, :3], rR[0, :3, 3])

        rRs_dict[(n1, n2)] = rR.view(3, 4)[:3, :3]
        rRs_dict[(n2, n1)] = rR_inv.view(3, 3)
    return rRs_dict    

def build_init_pose(num_nodes, root:Node, rel_R_dict:dict):
    pose = [torch.eye(3, dtype=torch.float)[:3, :3].view(1, 3, 3) for x in range(num_nodes)]

    q = [root]
    while len(q) != 0:
        u = q.pop(0)
        for v in u.childrens:
            rel_Rt_uv = rel_R_dict[(u.attr, v.attr)][:3, :3]
            pose[v.attr] = torch.bmm(rel_Rt_uv.view(1, 3, 3), pose[u.attr].view(1, 3, 3))
        q += [v for v in u.childrens]
    return pose

def check_R_err(num_nodes, root, rel_R_dict, rel_Es, init_Rs):
    q = [root]
    while len(q) != 0:
        u = q.pop(0)
        for v in u.childrens:
            rel_Rt_uv = rel_R_dict[(u.attr, v.attr)][:3, :3]

            if rel_Es != None:
                E_v = rel_Es[v.attr].view(1, 3, 4)
                E_u = rel_Es[u.attr].view(1, 3, 4)
                gR_init = init_Rs[v.attr].view(3, 3).cpu().numpy()
                deg = rel_R_deg(E_v.view(3, 4).cpu().numpy(), gR_init)

                rel_gt = cam_opt_gpu.relative_pose(E_u[:, :3, :3], E_u[:, :3, 3], E_v[:, :3, :3], E_v[:, :3, 3]).view(3, 4)
                rel_deg_err = rel_R_deg(rel_Rt_uv.cpu().numpy(), rel_gt.cpu().numpy())
                print(deg, rel_deg_err)
        q += [v for v in u.childrens]

def compute_R(tree, root, rRs_dict, node_list, Es):
    gRs = [0 for x in node_list]
    vis = [0 for x in node_list]
    adj_list = [[] for x in node_list]
    for idx in range(len(tree)):
        edge = tree[idx]
        n1, n2 = edge.u, edge.v
        adj_list[n2].append(n1)
        adj_list[n1].append(n2)

    q = []
    q.append(root)
    while len(q) != 0:
        u = q.pop(0)
        # if vis[u] == 1:
        #     continue
        # vis[u] = 1
        if u == root:
            gR = torch.eye(3, dtype=torch.float)[:3, :3]
            gRs[u] = gR.view(1, 3, 3)

        for v in adj_list[u]:
            if vis[v] != 1:
                rel_Rt_uv = rRs_dict[(u, v)][:3, :3]
                gRs[v] = torch.bmm(rel_Rt_uv.view(1, 3, 3), gRs[u].view(1, 3, 3))
                q.append(v)

                if Es != None:
                    E_v = Es[v].view(1, 3, 4)
                    E_u = Es[u].view(1, 3, 4)
                    gR_init = gRs[v].view(3, 3).cpu().numpy()
                    deg = rel_R_deg(E_v.view(3, 4).cpu().numpy(), gR_init)

                    rel_gt = cam_opt_gpu.relative_pose(E_u[:, :3, :3], E_u[:, :3, 3], E_v[:, :3, :3], E_v[:, :3, 3]).view(3, 4)
                    rel_deg_err = rel_R_deg(rel_Rt_uv.cpu().numpy(), rel_gt.cpu().numpy())
                    print(deg, rel_deg_err)
                
                vis[v] = 1

    for i in range(len(gRs)):
        if not isinstance(gRs[i], torch.Tensor):
            raise Exception('Error: Invalid spt')

    return gRs

def compute_spt(edge_list, probs, start_node=0):
    mst, node_list = gen_max_mst(edge_list, probs, start_node)
    return mst

def compute_booststrap(edgelist, probs, rRs_dict, start_node=0, Es=None):
    mst, node_list = gen_max_mst(edgelist, probs, start_node)
    gRs = compute_R(mst, start_node, rRs_dict, node_list, Es)
    gRs = torch.cat(gRs, dim=0)
    return gRs

def max_indices(arr, k):
    '''
    Returns the indices of the k first largest elements of arr
    (in descending order in values)
    '''
    assert k <= arr.size, 'k should be smaller or equal to the array size'
    arr_ = arr.astype(float)  # make a copy of arr
    max_idxs = []
    for _ in range(k):
        max_element = np.max(arr_)
        if np.isinf(max_element):
            break
        else:
            idx = np.where(arr_ == max_element)
        max_idxs.append(idx)
        arr_[idx] = -np.inf
    return max_idxs

def choose_anchor_node(node_num, edge_list: torch.Tensor, probs, top_k=None):
    _, E = edge_list.shape

    scores = np.zeros(node_num)
    links = np.zeros(node_num)
    for idx in range(E):

        prob = probs.view(-1)[idx].item()
        n1, n2 = edge_list[0][idx].item(), edge_list[1][idx].item()

        links[n1] += 1
        links[n2] += 1
        scores[n1] += prob
        scores[n2] += prob
    # return np.argmax(scores / (links + 1e-5))
    if top_k == None:
        return np.argmax(scores)
    else:
        return scores.argsort()[-top_k:][::-1]

def bi_direct_edge(e_node_idx, e_node_rel_Rt):

    e_list = []
    e_rel_Rt = []
    for i, e in enumerate(e_node_idx):
        n1, n2 = e[0].item(), e[1].item()
        e_list.append(torch.Tensor([n1, n2]).int())
        e_rel_Rt.append(e_node_rel_Rt[i].view(1, 3, 4))

    for i, e in enumerate(e_node_idx):
        n1, n2 = e[0].item(), e[1].item()
        e_list.append(torch.Tensor([n2, n1]).int())
        rRt = e_node_rel_Rt[i].view(1, 3, 4)
        # rR_inv, rt_inv = cam_opt_gpu.camera_pose_inv(rRt[0, :3, :3], rRt[0, :3, 3])
        rR_inv = rRt[0, :3, :3].permute(1, 0)
        rt_inv = rRt[0, :3, 3]
        Rt_inv = torch.zeros((3, 4))
        Rt_inv[:3, :3] = rR_inv.view(3, 3)
        Rt_inv[:3, 3] = rt_inv.view(-1)
        e_rel_Rt.append(Rt_inv.view(1, 3, 4))

    e_list = torch.stack(e_list)
    e_rel_Rt = torch.stack(e_rel_Rt)
    return e_list, e_rel_Rt.squeeze(1)