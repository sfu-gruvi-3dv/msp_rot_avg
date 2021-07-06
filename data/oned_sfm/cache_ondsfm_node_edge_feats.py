# dataset utils
import torch, pickle
import sys, os, json
import h5py
import numpy

torch.manual_seed(6666)

# add 5-point algorithm
# sys.path.append('/local-scratch7/pg/5point_alg_pybind/build')
sys.path.append('/mnt/Tango/pg/pg_akt_old/libs/5point_alg_pybind/build')
# import five_point_alg
from data.ambi.read_helper import *
from core_io.lmdb_writer import LMDBWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from exp.make_yfcc_dataset_new import make_dataset, datasets_config_filename
from exp.rot_avg_refined_finenet import *

""" Configuration -------------------------------------------------------------------------------------------------------
"""
# todo: modify following
# json_file_path = './train_config/onedsfm_part1_80node.json'
json_file_path = './train_config/yfcc_64nodes.json'

delete_lmdb = True

run_dev_ids = [1, 1]

max_batch_size = 30

# In[1] Network --------------------------------------------------------------------------------------------------------
from exp.same_size_2label_BCE_loss_trainbox_rollback import SameSizeLocalGlobalVLADTrainBox
train_params = TrainParameters()
train_params.DEV_IDS = run_dev_ids
train_params.VERBOSE_MODE = True

# todo: modify following
ckpt_dict = {
    'vlad': '/mnt/Exp_4/valid_cache/netvlad_vgg16.tar',
    'ckpt': '/mnt/Tango/pg/pg_akt_same_size/bce_yfcc_1dsfm_logs/Oct12_00-56-13_cs-gruvi-24s_1dsfm_gt/checkpoints/iter_015001.pth.tar'
}
prior_box = SameSizeLocalGlobalVLADTrainBox(train_params=train_params, ckpt_path_dict=ckpt_dict)
prior_box._prepare_eval()

print('Load trained model %s from %s' % (prior_box.__class__.__name__, ckpt_dict))

# In[2] Dataset --------------------------------------------------------------------------------------------------------
with open(json_file_path) as f:
    json_file = json.load(f)
    ds_type = list(json_file['train'].keys())[0] if 'train' in json_file else list(json_file['valid'].keys())[0]
    output_lmdb_file_name = json_file['train'][ds_type]['node_edge_lmdb_name'] if 'train' in json_file else json_file['valid'][ds_type]['node_edge_lmdb_name']

with open(datasets_config_filename) as f:
    json_file = json.load(f)
    output_lmdb_dir = json_file[ds_type]['node_edge_feat_dir']
    if not os.path.exists(output_lmdb_dir):
        os.mkdir(output_lmdb_dir)
out_node_edge_feat_lmdb_path = os.path.join(output_lmdb_dir, output_lmdb_file_name)
out_node_edge_feat_meta_path = out_node_edge_feat_lmdb_path.split('.lmdb')[0] + '.bin'
print('Output LMDB path: %s' % out_node_edge_feat_lmdb_path)

train_set, valid_set = make_dataset(json_file_path)

""" Function -----------------------------------------------------------------------------------------------------------
"""
def add_samples_to_lmdb(dataset, lmdb, processed_edge_dict, processed_node_dict):
    train_loader = DataLoader(dataset, num_workers=0, shuffle=True)

    pbar = tqdm(total=len(dataset))
    for sample in train_loader:
        dataset_name, idx, img_names, imgs, img_ori_dim, cam_Es, cam_Ks, _, img_id2sub_id, sub_id2img_id, _, edge_subnode_idx, edge_type, edge_local_matches_n1, edge_local_matches_n2, edge_rel_Rt, edge_rel_err, _, _ = sample

        # mark processed edges
        process_edge_idx = []
        process_match_n1 = []
        process_match_n2 = []

        # recover node linkage
        edge_ori_idx = []
        for e_i, e in enumerate(edge_subnode_idx):
            sub_n1, sub_n2 = e[0].item(), e[1].item()
            n1, n2 = sub_id2img_id[sub_n1], sub_id2img_id[sub_n2]
            # edge_ori_idx.append((n1, n2))
            key1 = '%s,%d-%d' % (dataset_name[0], n1, n2)
            key2 = '%s,%d-%d' % (dataset_name[0], n2, n1)
            if key1 not in processed_edge_dict and key2 not in processed_edge_dict:
                process_edge_idx.append(e)
                process_match_n1.append(edge_local_matches_n1[e_i])
                process_match_n2.append(edge_local_matches_n2[e_i])

        # run once
        if len(process_edge_idx) > 0:

            with torch.no_grad():
                node_feats = prior_box.cache_vlad_feats(imgs)
                edge_feats = prior_box.extract_edge_feats([
                    img_ori_dim, process_edge_idx, process_match_n1, process_match_n2
                ])
                prior_box.clean_cache()
            node_feats = node_feats.detach().cpu().numpy()
            edge_feats = edge_feats.detach().cpu().numpy()

            # save to lmdb
            for node_i in range(node_feats.shape[0]):
                node_ori_idx = sub_id2img_id[node_i]
                key = '%s,%d' % (dataset_name[0], node_ori_idx)
                if key in processed_node_dict:
                    continue

                node_feat = node_feats[node_i].ravel()
                lmdb.write_array(key, node_feat)
                processed_node_dict[key] = True

            for edge_i in range(edge_feats.shape[0]):
                e = process_edge_idx[edge_i]
                sub_n1, sub_n2 = e[0].item(), e[1].item()
                n1, n2 = sub_id2img_id[sub_n1], sub_id2img_id[sub_n2]
                key = '%s,%d-%d' % (dataset_name[0], n1, n2)
                if key in processed_edge_dict:
                    continue

                edge_feat = edge_feats[edge_i].ravel()
                lmdb.write_array(key, edge_feat)
                processed_edge_dict[key] = True

        pbar.update(1)

""" Dump to lmdb
"""
# init lmdb
lmdb = LMDBWriter(out_node_edge_feat_lmdb_path, earse_exist=delete_lmdb)
processed_edge_dict = dict()
processed_node_dict = dict()

if delete_lmdb is not True and os.path.exists(out_node_edge_feat_meta_path):
    with open(out_node_edge_feat_meta_path, 'rb') as f:
        o = pickle.load(f)
        processed_edge_dict, processed_node_dict = o

if train_set is not None:
    print('Add Train Dataset: %d items' % (len(train_set)))
    add_samples_to_lmdb(train_set, lmdb, processed_edge_dict, processed_node_dict)

if valid_set is not None:
    print('Add Valid Dataset: %d items' % (len(valid_set)))
    add_samples_to_lmdb(valid_set, lmdb, processed_edge_dict, processed_node_dict)

with open(out_node_edge_feat_meta_path, 'wb') as f:
    pickle.dump([processed_edge_dict, processed_node_dict], f)

lmdb.close_session()
