import numpy as np
from visualizer.visualizer_2d import show_multiple_img
import socket, shutil, os, pickle, sys, torch
from core_dl.train_params import TrainParameters
from exp.local_feat_gat_trainbox import LocalGlobalGATTrainBox, cluster_pool
from data.capture.capture_dataset_fast import CaptureDataset
from core_3dv.essential_mat import *

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
from data.capture.capture_dataset_fast import CaptureDataset
from data.oned_sfm.local_feat_dataset_fast import OneDSFMDataset

# load checkpoint if needed
checkpoint_dict = {
    'vlad': '/mnt/Exp_4/valid_cache/netvlad_vgg16.tar',
    'ckpt': '/mnt/Exp_4/valid_cache/iter_init.pth.tar'
}

# set log dir
log_dir = None

iccv_res_dir = '/mnt/Exp_5/AmbiguousData_pg/AmbiguousData_pg/'
cambridge_img_dir = '/mnt/Exp_5/AmbiguousData_pg/AmbiguousData_pg/'
lmdb_img_cache = ['/mnt/Exp_5/AmbiguousData_pg/cache.lmdb',
                  '/mnt/Exp_5/AmbiguousData_pg/cache_meta.bin']

dataset_list = [
    {'name': 'cup', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00064.out'},
    #     {'name': 'books', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00021.out'},
    #     {'name': 'cereal', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00025.out'},
    #     {'name': 'desk', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00031.out'},
    #     {'name': 'oats', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00024.out'},
    #     {'name': 'street', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00019.out'},
]

train_dataset = AmbiLocalFeatDataset(iccv_res_dir=iccv_res_dir,
                                     image_dir=cambridge_img_dir,
                                     lmdb_paths=lmdb_img_cache,
                                     dataset_list=dataset_list,
                                     downsample_scale=0.5,
                                     sampling_num=30,
                                     sample_res_cache='/mnt/Exp_5/AmbiguousData_pg/temp_cache.bin',
                                     sub_graph_nodes=24)

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

train_loader = dataloader.DataLoader(train_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     pin_memory=True,
                                     drop_last=True)
itr = iter(train_loader)

i = 0
for s_itr, sample in enumerate(itr):
    img_names, meta_dict, img_lists, img_dims, Es, Cs, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2 = sample

    found = False
    for img_name in img_names:
        if img_name[0].endswith('P1010205.jpg'):
            found = True
            break

    if found is False:
        continue

    print('Meta:', meta_dict)

    pred_edge_res = box.forward_once(sample[2:])

    pred_edge_res_ = torch.sigmoid(pred_edge_res)
    edge_label_ = edge_label.view(*pred_edge_res_.shape).to(pred_edge_res_.device).float()
    acc = box.acc_func(pred_edge_res_, edge_label_, threshold=0.8)

    print('Total Accuracy: %f' % acc)
    break

# improve graph
G = GraphVisualizer()

# nodes
for ix, img in enumerate(img_lists):
    img = img[0]
    name = img_names[ix][0]
    if name.endswith('P1010205.jpg'):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = torch.from_numpy(img).permute(2, 0, 1)
    G.add_node(ix, img=img, unormlize_tensor=True)

# edges
pred_probs = pred_edge_res_.view(-1)
for ei, (n1, n2) in enumerate(e_node_idx):
    label = edge_label[0][ei].item()
    prob = pred_probs[ei].item()

    n1 = e_node_idx[ei][0].item()
    n2 = e_node_idx[ei][1].item()
    G.add_edge(n1, n2, pred_prob=prob, gt_inlier=label, pred_prob_thres=0.8)

G.draw_spring_layout(draw=False, save_fig_path='/mnt/Exp_5/vis/dbg.pdf')
