import random
import numpy as np
import torch
import core_3dv.camera_operator_gpu as cam_opt_gpu

def filtering(sample, r_thres=20):
    ds, idx, img_name, img_list, img_dims, Es, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt, e_rel_errs, n_f, e_f = sample

    E = edge_label.shape[1]
    N = out_graph.shape[1]

    # # compute rel err
    # e_rel_errs = []
    # for i in range(E):
    #     n1 = e_node_idx[i][0].item()
    #     n2 = e_node_idx[i][1].item()
    #
    #     R1 = Es[0, n1:n1 + 1].view(1, 3, 4)
    #     R2 = Es[0, n2:n2 + 1].view(1, 3, 4)
    #     rel_R_gt = cam_opt_gpu.relative_pose(R1[:, :3, :3], R1[:, :3, 3], R2[:, :3, :3], R2[:, :3, 3])[:, :3, :3].view(
    #         3, 3)
    #     rel_R = e_rel_Rt[i][0, :3, :3].view(3, 3)
    #
    #     r_err_deg = rel_R_deg(rel_R_gt.cpu().numpy(), rel_R.cpu().numpy())
    #     e_rel_errs.append(r_err_deg)
    # e_rel_errs = np.asarray(e_rel_errs)

    nodes_dict = dict()

    f_e_node_idx = []
    f_e_rel_Rt = []
    f_e_f = []
    f_e_label = []
    f_e_rel_errs = []
    for i in range(E):
        if e_rel_errs[0, i].item() < r_thres:
            n1 = e_node_idx[i][0].item()
            n2 = e_node_idx[i][1].item()
            f_e_node_idx.append(e_node_idx[i])
            f_e_rel_Rt.append(e_rel_Rt[i])
            f_e_f.append(e_f[:, i:i + 1, :])
            f_e_label.append(1.0)
            f_e_rel_errs.append(e_rel_errs[:, i])
            nodes_dict[n1] = True
            nodes_dict[n2] = True

    # f_e_f = torch.cat(f_e_f, dim=1)

    # check connection
    invalid_nodes = dict()
    for n in range(N):
        if n not in nodes_dict:
            invalid_nodes[n] = True

    # add other edges
    invalid_nodes_edges = dict()
    for e_i in range(E):
        n1 = e_node_idx[e_i][0].item()
        n2 = e_node_idx[e_i][1].item()
        if n1 in invalid_nodes and n2 in nodes_dict:
            if n1 not in invalid_nodes_edges:
                invalid_nodes_edges[n1] = [e_i]
            else:
                invalid_nodes_edges[n1].append(e_i)

        elif n2 in invalid_nodes and n1 in nodes_dict:
            if n2 not in invalid_nodes_edges:
                invalid_nodes_edges[n2] = [e_i]
            else:
                invalid_nodes_edges[n2].append(e_i)

    # add to e_mask
    for node_i, edges in invalid_nodes_edges.items():
        random.shuffle(edges)

        for e in edges[:3]:
            f_e_node_idx.append(e_node_idx[e])
            f_e_rel_Rt.append(e_rel_Rt[e])
            f_e_f.append(e_f[:, e:e + 1, :])
            f_e_label.append(0.0)
            f_e_rel_errs.append(e_rel_errs[:, e])

    f_e_f = torch.cat(f_e_f, dim=1)
    f_e_label = torch.from_numpy(np.asarray(f_e_label)).int().unsqueeze(0)
    f_e_rel_errs = torch.cat(f_e_rel_errs, dim=0)

    sample = ds, idx, img_name, img_list, img_dims, Es, Ks, out_graph, sub2id, id2sub, covis_mat, f_e_node_idx, f_e_label, e_keypt_n1, e_keypt_n2, f_e_rel_Rt, f_e_rel_errs, n_f, f_e_f
    return sample
