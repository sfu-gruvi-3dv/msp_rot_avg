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
from graph_utils.utils import compute_booststrap, gen_max_mst, compute_spt
torch.manual_seed(6666)

# from exp.local_feat_gat_trainbox import LocalGlobalGATTrainBox, cluster_pool
# from exp.no_local_feat_gat_trainbox import LocalGlobalGATTrainBox
# from exp.local_feat_notgat_trainbox import LocalGlobalVLADTrainBox as LocalGlobalGATTrainBox
from logs.net_def.rot_avg_img_feat_finenet_box import LocalGlobalVLADTrainBox as LocalGlobalGATTrainBox


# load checkpoint if needed
checkpoint_dict = {
    'vlad': '/mnt/Exp_4/valid_cache/netvlad_vgg16.tar',
    # 'ckpt': '/mnt/Exp_4/valid_cache/iter_mgat_only.pth.tar'
    # 'ckpt': '/mnt/Exp_4/valid_cache/iter_mgat_local.pth.tar'
    'ckpt': '/mnt/Exp_4/valid_cache/iter_080000.pth.tar'
    # 'ckpt': '/local-scratch5/rot_avg_logs/Aug04_23-48-37_cs-guv-gpu02_local_appear_nogat_rot_avg_ori_loss_fixed_data/checkpoints/iter_020000.pth.tar'
}

# set log dir
log_dir = None
#
# iccv_res_dir = '/mnt/Exp_5/AmbiguousData_pg/AmbiguousData_pg/'
# cambridge_img_dir = '/mnt/Exp_5/AmbiguousData_pg/AmbiguousData_pg/'
# lmdb_img_cache = ['/mnt/Exp_5/AmbiguousData_pg/cache.lmdb',
#                   '/mnt/Exp_5/AmbiguousData_pg/cache_meta.bin']
# #
# dataset_list = [
#     {'name': 'cup_ori', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00064.out'},
#     # {'name': 'books', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00021.out'},
#     #     {'name': 'cereal', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00025.out'},
#     #     {'name': 'desk', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00031.out'},
#     #     {'name': 'oats', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00024.out'},
#     #     {'name': 'street', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00019.out'},
# ]
#
# # [CORE] Manual add the edges
# manual_linkage_dict = None
# # manual_linkage_dict={
# #     # key: (dataset_idx)_(subgraph_idx)
# #     '0_1':[(18, 13, 1)]     # node_n1, node_n2, label(-1:outlier, 1:inlier)
# # }
#
# train_dataset = AmbiLocalFeatDataset(iccv_res_dir=iccv_res_dir,
#                                      image_dir=cambridge_img_dir,
#                                      # lmdb_paths=lmdb_img_cache,
#                                      dataset_list=dataset_list,
#                                      downsample_scale=0.5,
#                                      sampling_num=30,
#                                      # sample_res_cache='/mnt/Exp_5/AmbiguousData_pg/temp_cache.bin',
#                                      manual_modification_dict=manual_linkage_dict,
#                                      sub_graph_nodes=32)

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
#     {'name': 'sv_statue1', 'bundle_prefix': 'bundle'},
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
                               sampling_undefined_edge=False,
                               sub_graph_nodes=41)

# train_dataset, _ = make_dataset()
# exit(0)

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

i = 0
for s_itr, sample in enumerate(itr):
    idx, img_names, img_lists, img_dims, Es, Cs, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt = sample    

    print('I:', s_itr, 'Idx:', idx, ' Outliers:', (edge_label == 0).sum(), 'Inliers:', (edge_label == 1).sum())    
    
    # for img_name in img_names:
    #     if img_name[0].endswith('P1010205.jpg'):
    #         found = True
    #         break
    #
    # if found is False:
    #     continue

    pred_rot, pred_edge_res, gt_rot, _, _, init_rot = box.forward_once(sample[2:])
    pred_edge_res_ = torch.sigmoid(pred_edge_res)

#     # dump for matlab ------------------------------
#     mat_edge_idx = []
#     mat_edge_rel_R = []
#     for e_i in range(pred_edge_res_.shape[0]):
#         edge_prob = pred_edge_res_[e_i].item()
#         edge_rel_R = e_rel_Rt[e_i][0][:3, :3]
# #         if edge_prob > 0.5:
#         if edge_label[0, e_i] == 1:
#             n1 = e_node_idx[e_i][0].item()
#             n2 = e_node_idx[e_i][1].item()
#             mat_edge_idx.append((n1, n2))
#             # observed:
#             R = edge_rel_R.cpu().numpy()

#             # GT:
#             # R = cam_opt_gpu.relative_pose(Es[:, n1, :, :3], Es[:, n1, :, 3], Es[:, n2, :, :3], Es[:, n2, :, 3])
#             # R = R[0, :, :3].cpu().numpy()

#             mat_edge_rel_R.append(R)

#             # Bi-directional
#             # mat_edge_idx.append((n2, n1))
#             # mat_edge_rel_R.append(R.T)


#             # print(np.linalg.det(R), np.linalg.det(R.T))

#     mat_edge_idx = np.asarray(mat_edge_idx)
#     # print(np.array(mat_edge_rel_R).shape)
#     mat_edge_rel_R = np.vstack(mat_edge_rel_R).reshape(-1, 3, 3)
#     mat_init_R = cam_opt_gpu.quaternion2rot(init_rot).detach().cpu().numpy()
#     mat_pred_R = cam_opt_gpu.quaternion2rot(pred_rot).detach().cpu().numpy()
#     mat_gt_R = cam_opt_gpu.quaternion2rot(gt_rot).detach().cpu().numpy()
    
#     sio.savemat('/mnt/Exp_4/graph_dbg/matlab2/sample_%04d.mat' % s_itr, {
#         'edge_idx': mat_edge_idx,
#         'edge_rel_R': mat_edge_rel_R,
#         'init_R': mat_init_R,
#         'pred_R': mat_pred_R,
#         'gt_R': mat_gt_R
#     })
    
#     if s_itr > 20:
#         break

    G = GraphVisualizer()

    # nodes
    for ix, img in enumerate(img_lists):
        name = img_names[ix][0].split('/')[-1].split('.jpg')[0]
        img = img.clone().squeeze(0)
        G.add_node(ix, img=img, name=name, unormlize_tensor=True)

    # edges
    for ei, (n1, n2) in enumerate(e_node_idx):
        label = edge_label[0][ei].item()
#         prob = 1.0 if label == 1 else 0.2
#         print(pred_edge_res_.size())
        print(pred_edge_res_)
        prob = pred_edge_res_[ei][0].item()
        n1 = e_node_idx[ei][0].item()
        n2 = e_node_idx[ei][1].item()
        G.add_edge(n1, n2, pred_prob=prob, gt_inlier=label, pred_prob_thres=0.8)
    G.draw_spring_layout(draw=False, save_fig_path='/mnt/Tango/pg/pg_akt_rot_avg/vis/spt_mpnn_iograph.pdf')

    break
    print((pred_edge_res_ > 0.8).view(-1).int())
    print((edge_label).int())
    
    edge_label_ = edge_label.view(*pred_edge_res_.shape).to(pred_edge_res_.device).float()
    acc = box.acc_func(pred_edge_res_, edge_label_, threshold=0.8)
    accs.append(acc)

    seqs.append(
        {'img_key': [img_name[0] for img_name in img_names],
         'edge_node_idx': [(n[0].item(), n[1].item()) for n in e_node_idx],
         'edge_label': edge_label_.detach().cpu().numpy(),
         'pred_prob': pred_edge_res_.detach().cpu().numpy()},
    )
    print('Total Accuracy: %f' % acc)
    pred_probs = pred_edge_res_.view(-1)

#     for e_i in range(pred_edge_res_.shape[0]):
#         edge_prob = pred_edge_res_[e_i].item()
#         if edge_prob > 0.8:
#             n1 = e_node_idx[e_i][0].item()
#             n2 = e_node_idx[e_i][1].item()
#             n1_name = img_names[n1][0].split('/')[-1]
#             n2_name = img_names[n2][0].split('/')[-1]
#             inlier_edges.append((n1_name, n2_name))

#     ## Compute mst
#     all_edge_for_mst = []
#     rRs = []
#     probs = []
#     start_node = 0
#     for ei, (n1, n2) in enumerate(e_node_idx):
#         label = edge_label[0][ei].item()
#         prob = pred_probs[ei].item()
#         n1 = e_node_idx[ei][0].item()
#         n2 = e_node_idx[ei][1].item()
#         all_edge_for_mst.append((n1, n2))
#         rRs.append(1)
#         probs.append(prob)
#     mst = compute_spt(all_edge_for_mst, probs, start_node)

#     # improve graph
#     G = GraphVisualizer()

#     # nodes
#     for ix, img in enumerate(img_lists):
#         img = img[0]
#         name = img_names[ix][0].split('/')[-1].split('.jpg')[0]
#         if name.endswith('P1010205'):
#             img = img.permute(1, 2, 0).cpu().numpy()
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             img = torch.from_numpy(img).permute(2, 0, 1)
#         G.add_node(ix, img=img, name=name, unormlize_tensor=True)

#     # edges
#     for ei, (n1, n2) in enumerate(e_node_idx):
#         label = edge_label[0][ei].item()
#         prob = pred_probs[ei].item()
#         if prob < 0.6:
#             continue

#         n1 = e_node_idx[ei][0].item()
#         n2 = e_node_idx[ei][1].item()
#         G.add_edge(n1, n2, pred_prob=prob, gt_inlier=label, pred_prob_thres=0.8)

#     # spanning tree
#     for sp_e in mst:
#         n1 = sp_e[0]
#         n2 = sp_e[1]
#         G.add_mark_edge(n1, n2)

#     G.draw_spring_layout(draw=False, save_fig_path='/mnt/Exp_5/vis/bottles/spt_%d_mpnn.pdf' % s_itr)


#     # mst, node_list = gen_max_mst(all_edge_for_mst, probs, start_node)

#     plt.close()
#     # break

#     if s_itr >= 0:
#         break

# # # save inlier pairs
# # edge_dict = dict()
# # for e_name in inlier_edges:
# #     n1_name = e_name[0]
# #     n2_name = e_name[1]
# #
# #     key1 = "%s-%s" % (n1_name, n2_name)
# #     key2 = "%s-%s" % (n2_name, n1_name)
# #
# #     if key1 not in edge_dict or key2 not in edge_dict:
# #         edge_dict[key1] = True
# #
# # with open('/mnt/Exp_5/vis/bottles_pairs.txt', 'w') as f:
# #     for key, val in edge_dict.items():
# #         tokens = key.split('-')
# #         f.write('%s %s\n' % (tokens[0], tokens[1]))

# accs = np.asarray(accs)
# print('Average: %f, Std: %f' % (accs.mean(), accs.std()))
# np.save('/mnt/Exp_5/vis/bottles/acc_bottles_mpnn.npy', accs)
# with open('/mnt/Exp_5/vis/recon_seqs_bottles_mpnn.bin', 'wb') as f:
#     pickle.dump(seqs, f)
