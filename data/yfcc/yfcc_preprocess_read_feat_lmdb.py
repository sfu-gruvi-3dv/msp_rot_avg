# dataset utils
import torch, pickle
import sys
import h5py
import numpy

torch.manual_seed(6666)

# add 5-point algorithm
# sys.path.append('/local-scratch7/pg/5point_alg_pybind/build')
sys.path.append('/mnt/Tango/pg/pg_akt_old/libs/5point_alg_pybind/build')
# import five_point_alg
from data.ambi.read_helper import *
from core_io.lmdb_reader import LMDBModel
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from exp.make_yfcc_dbg_dataset import make_dataset
from exp.rot_avg_refined_finenet import *

# In[1] Dataset
out_node_edge_feat_lmdb_path = '/mnt/Exp_5/yfcc100/node_edge_feat.lmdb'
out_node_edge_feat_meta_path = out_node_edge_feat_lmdb_path.split('.lmdb')[0] + '.bin'

delete_lmdb = True

run_dev_ids = [0, 0]
max_batch_size = 30

train_set, valid_set = make_dataset()

# In[2] Network --------------------------------------------------------------------------------------------------------
from exp.same_size_2to1_local_feat_notgat_trainbox import SameSizeLocalGlobalVLADTrainBox
train_params = TrainParameters()
train_params.DEV_IDS = run_dev_ids
train_params.VERBOSE_MODE = True
prior_box = SameSizeLocalGlobalVLADTrainBox(train_params=train_params, ckpt_path_dict={
    'vlad': '/mnt/Exp_4/valid_cache/netvlad_vgg16.tar',
    'ckpt': '/mnt/Exp_4/valid_cache/edge_yfcc_init.pth.tar'
})
prior_box._prepare_eval()

""" Function
"""
def read_lmdb(dataset, lmdb, processed_edge_dict, processed_node_dict):
    train_loader = DataLoader(dataset, num_workers=0, shuffle=True)

    pbar = tqdm(total=len(dataset))
    for sample in train_loader:
        dataset_name, idx, img_names, imgs, img_ori_dim, cam_Es, cam_Ks, _, img_id2sub_id, sub_id2img_id, _, edge_subnode_idx, edge_type, edge_local_matches_n1, edge_local_matches_n2, edge_rel_Rt = sample

        for e_i, e in enumerate(edge_subnode_idx):
            sub_n1, sub_n2 = e[0].item(), e[1].item()
            n1, n2 = sub_id2img_id[sub_n1], sub_id2img_id[sub_n2]
            node_key = '%s,%d' % (dataset_name[0], sub_n1)
            edge_key = '%s,%d-%d' % (dataset_name[0], n1, n2)

            node_feat = lmdb.read_ndarray_by_key(node_key)
            edge_feat = lmdb.read_ndarray_by_key(edge_key)

        pbar.update(1)

""" Dump to lmdb
"""
# init lmdb
lmdb = LMDBModel(out_node_edge_feat_lmdb_path, read_only=True)
if os.path.exists(out_node_edge_feat_meta_path):
    with open(out_node_edge_feat_meta_path, 'rb') as f:
        o = pickle.load(f)
        processed_edge_dict, processed_node_dict = o
read_lmdb(train_set, lmdb, processed_edge_dict, processed_node_dict)
