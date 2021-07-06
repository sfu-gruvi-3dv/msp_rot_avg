from exp.make_dataset import make_dataset
from networkx.algorithms.clique import enumerate_all_cliques
import numpy as np
from visualizer.visualizer_2d import show_multiple_img
import socket, shutil, os, pickle, sys, torch
from core_dl.train_params import TrainParameters
from data.capture.capture_dataset_rel_fast import CaptureDataset
from core_3dv.essential_mat import *
import scipy.io as sio

# dataset utils
from data.ambi.ambi_dataset import imgs2batchsamesize
import core_3dv.camera_operator_gpu as cam_opt_gpu
import torch.nn.functional as f
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

# graph mst
from graph_utils.utils import *
torch.manual_seed(6666)

# from exp.local_feat_gat_trainbox import LocalGlobalGATTrainBox, cluster_pool
# from exp.no_local_feat_gat_trainbox import LocalGlobalGATTrainBox
# from exp.local_feat_notgat_trainbox import LocalGlobalVLADTrainBox as LocalGlobalGATTrainBox
from exp.rot_avg_org_loss_trainbox import LocalGlobalVLADTrainBox as LocalGlobalGATTrainBox

# load checkpoint if needed
checkpoint_dict = {
    'vlad': '/mnt/Exp_4/valid_cache/netvlad_vgg16.tar',
    # 'ckpt': '/mnt/Exp_4/valid_cache/iter_mgat_only.pth.tar'
    # 'ckpt': '/mnt/Exp_4/valid_cache/iter_mgat_local.pth.tar'
    # 'ckpt': '/mnt/Exp_4/valid_cache/iter_init.pth.tar'
    'ckpt': '/local-scratch5/rot_avg_logs/Aug04_23-48-37_cs-guv-gpu02_local_appear_nogat_rot_avg_ori_loss_fixed_data/checkpoints/iter_020000.pth.tar'
}

# set log dir
log_dir = None

""" Captured Dataset
"""
# cap_res_dir = '/mnt/Exp_5/pg_train'
# # cap_res_dir = '/mnt/Exp_5/bottles'
# # cap_res_dir = '/mnt/Exp_5/AmbiguousData_pg/AmbiguousData_pg'
# valid_c_datalist = [
#     # {'name': 'hall', 'bundle_prefix': 'bundle'},
#     # {'name': 'cup_pred_test', 'bundle_prefix': 'bundle'}
#     # {'name': 'Westerminster_colmap_manual', 'bundle_prefix': 'bundle'},
#     # {'name': 'sv_seq7', 'bundle_prefix': 'bundle'},
#     {'name': 'sv_statue1', 'bundle_prefix': 'bundle'},
# #     {'name': 'bottles_7', 'bundle_prefix': 'bundle'},
# ]

# train_dataset = CaptureDataset(iccv_res_dir=cap_res_dir,
#                                image_dir=cap_res_dir,
#                                dataset_list=valid_c_datalist,
#                                img_max_dim=480,
#                                sampling_num_range=[8, 8],
#                             #    sample_res_cache='/tmp/furniture6.bin',
#                                # sample_res_cache='/tmp/bottles7_36_new_undefined.bin',
#                                # sample_res_cache='/tmp/wester_48.bin',
#                                sampling_undefined_edge=False,
#                                sub_graph_nodes=31)

train_dataset, _ = make_dataset()

# set train parameters
train_params = TrainParameters()
train_params.MAX_EPOCHS = 20
train_params.START_LR = 1.0e-4
train_params.DEV_IDS = [0, 1]
train_params.LOADER_BATCH_SIZE = 1
train_params.LOADER_NUM_THREADS = 0
train_params.VALID_STEPS = 5000
train_params.MAX_VALID_BATCHES_NUM = 20
train_params.CHECKPOINT_STEPS = 6000
train_params.VERBOSE_MODE = True
train_params.NAME_TAG = 'test_gat_cluster'

box = LocalGlobalGATTrainBox(train_params=train_params,
                             ckpt_path_dict=checkpoint_dict)
box._prepare_eval()

train_loader = dataloader.DataLoader(train_dataset,
                                     batch_size=1,
                                     shuffle=True,
                                     pin_memory=True,
                                     drop_last=True)
itr = iter(train_loader)

seqs = []
inlier_edges = []
accs = []

n_spt_nodes = []
new_idx = []
iout_count = []

i = 0
for s_itr, sample in enumerate(itr):
    idx, img_names, img_lists, img_dims, Es, Cs, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt = sample    

    print('I:', s_itr, 'Idx:', idx, ' Outliers:', (edge_label == 0).sum(), 'Inliers:', (edge_label == 1).sum())
    new_idx.append((idx.item(), (edge_label == 0).sum().item(), (edge_label == 1).sum().item()))

    spt_root = build_inlier_spt(len(img_names), e_node_idx, edge_label)
    spt_nodes_num = count_tree_nodes(spt_root)
    print('Num SPT Nodes:', spt_nodes_num)
    n_spt_nodes.append(spt_nodes_num)
    continue
    # found = False
    # for img_name in img_names:
    #     if img_name[0].endswith('P1010205.jpg'):
    #         found = True
    #         break
    #
    # if found is False:
    #     continue

    pred_rot, pred_edge_res, gt_rot, _, _, init_rot = box.forward_once(sample[1:])
    pred_edge_res_ = torch.sigmoid(pred_edge_res)

    # dump for matlab ------------------------------
    mat_edge_idx = []
    mat_edge_rel_R = []
    for e_i in range(pred_edge_res_.shape[0]):
        edge_prob = pred_edge_res_[e_i].item()
        edge_rel_R = e_rel_Rt[e_i][0][:3, :3]
#         if edge_prob > 0.5:
        if edge_label[0, e_i] == 1:
            n1 = e_node_idx[e_i][0].item()
            n2 = e_node_idx[e_i][1].item()
            mat_edge_idx.append((n1, n2))
            # observed:
            R = edge_rel_R.cpu().numpy()

            #GT:
            # R = cam_opt_gpu.relative_pose(Es[:, n1, :, :3], Es[:, n1, :, 3], Es[:, n2, :, :3], Es[:, n2, :, 3])
            # R = R[0, :, :3].cpu().numpy()

            mat_edge_rel_R.append(R)

            #Bi-directional
            # mat_edge_idx.append((n2, n1))
            # mat_edge_rel_R.append(R.T)


            # print(np.linalg.det(R), np.linalg.det(R.T))

    mat_edge_idx = np.asarray(mat_edge_idx)
    # print(np.array(mat_edge_rel_R).shape)
    mat_edge_rel_R = np.vstack(mat_edge_rel_R).reshape(-1, 3, 3)
    mat_init_R = cam_opt_gpu.quaternion2rot(init_rot).detach().cpu().numpy()
    mat_pred_R = cam_opt_gpu.quaternion2rot(pred_rot).detach().cpu().numpy()
    mat_gt_R = cam_opt_gpu.quaternion2rot(gt_rot).detach().cpu().numpy()
    
    sio.savemat('/mnt/Exp_4/graph_dbg/matlab/sample_%04d.mat' % s_itr, {
        'edge_idx': mat_edge_idx,
        'edge_rel_R': mat_edge_rel_R,
        'init_R': mat_init_R,
        'pred_R': mat_pred_R,
        'gt_R': mat_gt_R
    })

np.save('/mnt/Exp_4/graph_dbg/spt_nodes_count.npy', np.asarray(n_spt_nodes))
np.save('/mnt/Exp_4/graph_dbg/dataset_random_idxmap.npy', np.asarray(new_idx))