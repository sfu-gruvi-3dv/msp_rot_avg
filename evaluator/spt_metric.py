import random, torch
import core_3dv.camera_operator_gpu as cam_opt_gpu
import evaluator.basic_metric as eval
import numpy as np


class Node:
    childrens = []
    parent = None

    def __init__(self, attr, level=0):
        self.attr = attr
        self.childrens = []
        self.parent = None
        self.level = 0

    def add_child(self, node):
        node.parent = self
        node.level = self.level + 1
        self.childrens.append(node)


def rel_ref_2_E(ref_id, cam_Es):
    N, _, _ = cam_Es[0].shape

    ref_cam_Es = []
    ref_E = cam_Es[0, ref_id].unsqueeze(0)
    for e_i in range(N):
        E = cam_Es[0, e_i].unsqueeze(0)
        rel_ref_E = cam_opt_gpu.relative_pose(ref_E[:, :3, :3], ref_E[:, :3, 3],
                                              E[:, :3, :3], E[:, :3, 3])
        ref_cam_Es.append(rel_ref_E.squeeze(0))

    return ref_cam_Es


def rel_R_deg(R1, R2):
    # relative pose
    rel_R = np.matmul(R1[:3, :3], R2[:3, :3].T)
    R_err = np.rad2deg(np.arccos((np.trace(rel_R) - 1) / 2))
    return R_err

def spt_statsitic_view(node_nums, out_graph, e_node_idx, cam_Es, e_rel_Rt, verbose=True, inlier=True):
    # build adj mat
    adj_mat_dict = {}
    for im, e in enumerate(e_node_idx):
        n1 = e[0].item()
        n2 = e[1].item()
        Rt = e_rel_Rt[im].squeeze(0)
        R, t = cam_opt_gpu.camera_pose_inv(Rt[:3, :3], Rt[:3, 3])
        Rt_inv = torch.zeros_like(Rt)
        Rt_inv[:3, :3] = R
        Rt_inv[:3, 3] = t.view(3)
        adj_mat_dict[(n1, n2)] = Rt
        adj_mat_dict[(n2, n1)] = Rt_inv

    # spt on inliers
    N = node_nums
    max_level = 0
    label = 1 if inlier == True else -1
    while max_level < 1:
        random_idx = random.randint(0, N - 1)
        root_node = Node(random_idx)
        stack = [root_node]

        # find the inliers
        flag = {root_node.attr: True}

        while len(stack) != 0:
            cur_node = stack.pop(0)

            if cur_node.level > max_level:
                max_level = cur_node.level

            out_graph = out_graph.view(node_nums,node_nums)
            row = out_graph[cur_node.attr]
            for j in range(row.shape[0]):
                if row[j].item() == label and j not in flag:
                    child_node = Node(j)
                    cur_node.add_child(child_node)
                    flag[j] = True
                    stack.append(child_node)

    stack = [root_node]
    poses = {root_node.attr: torch.eye(3).float()[:3, :]}

    while len(stack) != 0:
        cur_node = stack.pop(0)
        cur_R = poses[cur_node.attr]

        for child in cur_node.childrens:
            child_attr = child.attr
            rel_R = adj_mat_dict[(cur_node.attr, child_attr)]
            child_R = torch.bmm(rel_R[:3, :3].unsqueeze(0), cur_R.unsqueeze(0))
            poses[child_attr] = child_R.squeeze(0)
            stack.append(child)

    gt_poses = rel_ref_2_E(random_idx, cam_Es=cam_Es)
    gt_poses = {key: gt_poses[key] for key in poses.keys()}

    # compute pose
    stack = [root_node]
    level_errs = dict()

    while len(stack) != 0:
        cur_node = stack.pop(0)
        cur_R = poses[cur_node.attr]
        cur_gt_R = gt_poses[cur_node.attr]

        R_err = rel_R_deg(cur_R.cpu().numpy(), cur_gt_R.cpu().numpy())
        level = cur_node.level
        if level not in level_errs:
            level_errs[cur_node.level] = []
        level_errs[cur_node.level].append(R_err)

        for child in cur_node.childrens:
            stack.append(child)

    for key in level_errs.keys():
        level_errs[key] = np.asarray(level_errs[key])

    if verbose:
        for level in level_errs.keys():
            if level == 0:
                continue

            R_errs = level_errs[level]

            print('[%s Level %d] AVG:%.2f STD:%.2f MIN:%.2f MAX:%.2f' % ('Inliers' if inlier == True else 'Outliers',
                                                                         level,
                                                                         np.mean(R_errs), np.std(R_errs),
                                                                         np.min(R_errs), np.max(R_errs)))

    return level_errs