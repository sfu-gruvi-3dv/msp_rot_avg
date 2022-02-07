from data.ambi.ambi_parser import *
import numpy
from exp.make_dataset import make_dataset
from networkx.algorithms.clique import enumerate_all_cliques
import numpy as np
from core_math.transfom import euler_from_matrix, quaternion_from_matrix
from visualizer.visualizer_2d import *
import socket, shutil, os, pickle, sys, torch
from core_dl.train_params import TrainParameters
from data.capture.capture_dataset_rel_fast import CaptureDataset
from core_3dv.essential_mat import *
import scipy.io as sio
from core_3dv.quaternion import *

# dataset utils
from data.ambi.ambi_dataset import imgs2batchsamesize
import core_3dv.camera_operator_gpu as cam_opt_gpu
import torch
import torch.nn.functional as F
from net.adjmat2dgl_graph import build_graph, adjmat2graph, gather_edge_feat, gather_edge_label, inv_adjmat, \
    gather_edge_feat2

# add 5-point algorithm
# sys.path.append('/local-scratch7/pg/5point_alg_pybind/build')
sys.path.append('/mnt/Tango/pg/pg_akt_old/libs/5point_alg_pybind/build')
# import five_point_alg
from core_3dv.essential_mat import *
from torch.utils.data import dataloader
from data.capture.visualize_graph import GraphVisualizer
from data.ambi.ambi_mutable_dataset import AmbiLocalFeatDataset
from data.oned_sfm.local_feat_dataset_fast import OneDSFMDataset
import graph_utils.utils as graph_util
import torchgeometry as tgm

torch.manual_seed(6666)

# from exp.local_feat_gat_trainbox import LocalGlobalGATTrainBox, cluster_pool
# from exp.no_local_feat_gat_trainbox import LocalGlobalGATTrainBox
from exp.local_feat_notgat_trainbox import LocalGlobalVLADTrainBox as LocalGlobalGATTrainBox_Prior
# from exp.rot_avg_org_loss_trainbox import LocalGlobalVLADTrainBox as LocalGlobalGATTrainBox
from exp.rot_avg_refined_finenet import LocalGlobalVLADTrainBox as LocalGlobalGATTrainBox
from exp.rot_avg_refined_finenet import *

# In[1] Dataset
""" Captured Dataset
"""
cap_res_dir = '/mnt/Exp_5/pg_train'
# cap_res_dir = '/mnt/Exp_5/bottles'
# cap_res_dir = '/mnt/Exp_5/AmbiguousData_pg/AmbiguousData_pg'
valid_c_datalist = [
    # {'name': 'hall', 'bundle_prefix': 'bundle'},
    # {'name': 'cup_pred_test', 'bundle_prefix': 'bundle'}
    # {'name': 'Westerminster_colmap_manual', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_seq7', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_statue1', 'bundle_prefix': 'bundle'},
    # {'name': 'ambi_cup', 'bundle_prefix': 'bundle'},
    {'name': 'bottles_7', 'bundle_prefix': 'bundle'},
]

train_dataset = CaptureDataset(iccv_res_dir=cap_res_dir, 
                            image_dir=cap_res_dir,
                            dataset_list=valid_c_datalist,
                            img_max_dim=480,
                            sampling_num_range=[8, 8],
                            #    sample_res_cache='/tmp/furniture6.bin',
                            # sample_res_cache='/tmp/bottles7_36_new_undefined.bin',
                            # sample_res_cache='/tmp/wester_48.bin',
                            sampling_undefined_edge=True,
                            sub_graph_nodes=64)

# Ambigous dataset:


# In[0]: Network 
# load checkpoint if needed
checkpoint_dict = {
    'vlad': '/mnt/Exp_4/valid_cache/netvlad_vgg16.tar',
    # 'ckpt': '/mnt/Exp_4/valid_cache/iter_mgat_only.pth.tar'
    # 'ckpt': '/mnt/Exp_4/valid_cache/iter_mgat_local.pth.tar'
    # 'ckpt': '/mnt/Exp_4/valid_cache/iter_init.pth.tar'
    'ckpt': '/mnt/Exp_4/rot_avg_logs/Aug14_09-02-41_cs-gruvi-24-cmpt-sfu-ca_ori_finenet_rot_avg_fixed_data_inlier_spt_only/checkpoints/iter_000002.pth.tar'
}

# set train parameters
train_params = TrainParameters()
train_params.DEV_IDS = [0, 0]
train_params.VERBOSE_MODE = True

box = LocalGlobalGATTrainBox(train_params=train_params, ckpt_path_dict=checkpoint_dict)
box._prepare_eval()

# train_params = TrainParameters()
# train_params.DEV_IDS = [0, 1]
# train_params.VERBOSE_MODE = True
#
# prior_box = LocalGlobalGATTrainBox_Prior(train_params=train_params, ckpt_path_dict={
#     'vlad': '/mnt/Exp_4/valid_cache/netvlad_vgg16.tar',
#     'ckpt': '/mnt/Exp_4/valid_cache/iter_init.pth.tar'
# })
# prior_box._prepare_eval()

train_loader = dataloader.DataLoader(train_dataset, 
                                batch_size=1, shuffle=True, pin_memory=True, drop_last=True)

itr = iter(train_loader)

# In[0]: Test 
seqs = []
inlier_edges = []
accs = []

i = 0
for s_itr, sample in enumerate(itr):
    idx, img_names, img_lists, img_dims, Es, Cs, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt = sample

    # with torch.no_grad():
    #     out_edge_res = prior_box.forward_once(sample[2:])
    #     out_edge_res_ = torch.sigmoid(out_edge_res).view(-1)

    spt_init_q, pred_q, gt_q, loss, spt_root, spt_node_nums, pred_edge_prob = box.forward_once(sample[2:], build_inlier=True)

    print('I:', s_itr, 'Root', spt_root.attr, 'Idx:', idx, ' Outliers:', (edge_label == 0).sum(), 'Inliers:', (edge_label == 1).sum(), 'SPT_inlier_nodes:', spt_node_nums)

    # init
    theta = quaternion_rel_deg(spt_init_q, gt_q)
    print('[Init] mean: %0.2f, std: %0.2f, median: %0.2f' % (np.mean(theta), np.std(theta), np.median(theta)))

    # measure the residual
    theta2 = quaternion_rel_deg(pred_q, gt_q)
    print('[Opt] mean: %0.2f, std: %0.2f, median: %0.2f' % (np.mean(theta2), np.std(theta2), np.median(theta2)))

    # pred_q2 = pred_q
    # for j in range(5):
    #     pred_q2, pred_edge_prob = box.inference(pred_q2, gt_q, e_node_idx, e_rel_Rt)
    #
    #     # measure the residule
    #     pred_residual_q = qmul(inv_q(gt_q), pred_q2)
    #     pred_residual_q = F.normalize(pred_residual_q, p=2, dim=1)
    #     val2 = pred_residual_q.data.cpu().numpy()
    #     theta = 2.0 * np.arccos(np.abs(val2[:, 0])) * 180.0 / np.pi
    #     print('[Opt] mean: %0.2f, std: %0.2f, median: %0.2f' % (np.mean(theta), np.std(theta), np.median(theta)))

    pred_edge_res_ = torch.sigmoid(pred_edge_prob)
    pred_edge_res_ = pred_edge_res_.view(-1)

    """ visualize graph (initilization)
    """
    spt_init_R = quaternion2rot(spt_init_q)

    # mark the inliers
    edge_errs = dict()
    init_inlier_edges = []
    for ei, (n1, n2) in enumerate(e_node_idx):
        label = edge_label[0][ei].item()
        n1 = e_node_idx[ei][0].item()
        n2 = e_node_idx[ei][1].item()
        rel_R_obs = e_rel_Rt[ei]

        n1_R = spt_init_R[n1].cpu().numpy()
        n2_R = spt_init_R[n2].cpu().numpy()

        rel_R = np.matmul(n1_R, n2_R.T)
        err = rel_R_deg(rel_R, rel_R_obs.cpu().numpy())
        edge_errs[(n1, n2)] = err
        if err < 20:
            init_inlier_edges.append((n1, n2))

    # spt_root_errs = [edge_errs[(spt_root.attr, spt_root.childrens[i].attr)] for i in range(len(spt_root.childrens))]

    """ visualize graph
    """
    G = GraphVisualizer()
    
    # draw nodes
    for ix, img in enumerate(img_lists):
        img = img[0]
        name = img_names[ix][0].split('/')[-1].split('.jpg')[0]
        to_bgr = spt_root.attr == ix
        text = '%.2f' % (theta2[ix] - theta[ix])
        if spt_root.attr == ix:
            text += '(ROOT)'
        G.add_node(ix, img=img, name=name, unormlize_tensor=True, text=text, to_bgr=to_bgr)

    # edges
    for ei, (n1, n2) in enumerate(e_node_idx):
        label = edge_label[0][ei].item()
        prob = pred_edge_res_[ei].item()
        if prob < 0.7:
            continue

        n1 = e_node_idx[ei][0].item()
        n2 = e_node_idx[ei][1].item()
        G.add_edge(n1, n2, pred_prob=prob, gt_inlier=label, pred_prob_thres=0.8)

    for (n1, n2) in init_inlier_edges:
        G.add_mark_edge(n1, n2)

    G.draw_spring_layout(draw=False, save_fig_path='/mnt/Exp_5/vis/cup/spt_inlier_%d_finenet.pdf' % s_itr)
    G.draw_spring_layout_marker(draw=False, save_fig_path='/mnt/Exp_5/vis/cup/spt_inlier_initspt_%d_finenet.pdf' % s_itr, update_pos=False)
