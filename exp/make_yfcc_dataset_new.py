import socket
import shutil
import os
import pickle
import json
import glob
from torch.utils.data import ConcatDataset
from core_dl.train_params import TrainParameters
from data.capture.capture_dataset_nonimage import CaptureDataset
import torchvision.transforms as transforms
import numpy as np
import json

server_name = socket.gethostname()

if 'cs-gruvi-24s' in server_name:
    datasets_config_filename = "./exp/dataset_configs/dataset_config_24s.json"
elif 'cs-guv-gpu02' in server_name:
    datasets_config_filename = "./exp/dataset_configs/dataset_config_gpu02.json"
elif '50s' in server_name:
    datasets_config_filename = "./exp/dataset_configs/dataset_config_50s.json"
elif '49s' in server_name:
    datasets_config_filename = "./exp/dataset_configs/dataset_config_49s.json"


def clean_cache(datasets_config, train_valid_config):
    paths = []
    for dataset_name, val in train_valid_config["train"].items():
        paths.append(os.path.join(datasets_config[dataset_name]["sampled_subgraph_dir"],
                                  val["sample_res_cache"]))
    for dataset_name, val in train_valid_config["valid"].items():
        paths.append(os.path.join(datasets_config[dataset_name]["sampled_subgraph_dir"],
                                  val["sample_res_cache"]))
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def make_dataset(json_config=None):
    if json_config is None:
        json_config = "./exp/dataset_configs/make_train_valid.json"
    print('Load Dataset config from %s' % json_config)

    with open(json_config, "r") as fin:
        train_valid_config = json.load(fin)
    with open(datasets_config_filename, "r") as fin:
        datasets_config = json.load(fin)

    if train_valid_config["clean_cache"]:
        clean_cache(datasets_config, train_valid_config)

    train_dataset_list = []
    valid_dataset_list = []
    for dataset_lib, dataset_dict in train_valid_config["train"].items():
        train_dataset = []
        for train in dataset_dict["dataset_list"]:
            for it in datasets_config[dataset_lib]["dataset_list"]:
                if it["name"] == train:
                    train_dataset.append(it)
                    break

        if not os.path.exists(datasets_config[dataset_lib]["node_edge_feat_dir"]):
            os.mkdir(datasets_config[dataset_lib]["node_edge_feat_dir"])

        if not os.path.exists(datasets_config[dataset_lib]["sampled_subgraph_dir"]):
            os.mkdir(datasets_config[dataset_lib]["sampled_subgraph_dir"])

        load_node_edge_feat = dataset_dict["load_node_edge_feat"]
        train_dataset_list.append(CaptureDataset(dataset_dir=datasets_config[dataset_lib]["res_dir"],
                                                 img_lmdb_paths=datasets_config[dataset_lib]["lmdb_cache"],
                                                 node_edge_lmdb_path=None if dataset_dict[
                                                                                 "node_edge_lmdb_name"] is None else os.path.join(
                                                     datasets_config[dataset_lib]["node_edge_feat_dir"],
                                                     dataset_dict["node_edge_lmdb_name"]),
                                                 dataset_list=train_dataset,
                                                 img_max_dim=dataset_dict["img_max_dim"],
                                                 sample_res_cache=None if dataset_dict[
                                                                              "sample_res_cache"] is None else os.path.join(
                                                     datasets_config[dataset_lib]["sampled_subgraph_dir"],
                                                     dataset_dict["sample_res_cache"]),
                                                 sampling_num_range=dataset_dict["sampling_num_range"],
                                                 load_img=dataset_dict["load_img"],
                                                 load_keypt_match=dataset_dict["load_keypt_match"],
                                                 load_node_edge_feat=load_node_edge_feat,
                                                 sub_graph_nodes=dataset_dict["sub_graph_nodes"],
                                                 outlier_edge_thres_deg=dataset_dict['outlier_thres'] if 'outlier_thres' in dataset_dict else None,
                                                 sampling_undefined_edge=dataset_dict["sampling_undefine_edge"])
                                  )
    for dataset_lib, dataset_dict in train_valid_config["valid"].items():
        valid_dataset = []
        for train in dataset_dict["dataset_list"]:
            for it in datasets_config[dataset_lib]["dataset_list"]:
                if it["name"] == train:
                    valid_dataset.append(it)
                    break
        valid_dataset_list.append(CaptureDataset(dataset_dir=datasets_config[dataset_lib]["res_dir"],
                                                 img_lmdb_paths=datasets_config[dataset_lib]["lmdb_cache"],
                                                 node_edge_lmdb_path=None if dataset_dict[
                                                                                 "node_edge_lmdb_name"] is None else os.path.join(
                                                     datasets_config[dataset_lib]["node_edge_feat_dir"],
                                                     dataset_dict["node_edge_lmdb_name"]),
                                                 dataset_list=valid_dataset,
                                                 img_max_dim=dataset_dict["img_max_dim"],
                                                 sample_res_cache=None if dataset_dict[
                                                                              "sample_res_cache"] is None else os.path.join(
                                                     datasets_config[dataset_lib]["sampled_subgraph_dir"],
                                                     dataset_dict["sample_res_cache"]),
                                                 sampling_num_range=dataset_dict["sampling_num_range"],
                                                 load_img=dataset_dict["load_img"],
                                                 load_keypt_match=dataset_dict["load_keypt_match"],
                                                 load_node_edge_feat=dataset_dict["load_node_edge_feat"],
                                                 sub_graph_nodes=dataset_dict["sub_graph_nodes"],
                                                 outlier_edge_thres_deg=dataset_dict['outlier_thres'] if 'outlier_thres' in dataset_dict else None,
                                                 sampling_undefined_edge=dataset_dict["sampling_undefine_edge"])
                                  )
    train_set = None if len(train_dataset_list) == 0 else ConcatDataset(train_dataset_list)
    valid_set = None if len(valid_dataset_list) == 0 else ConcatDataset(valid_dataset_list)
    return train_set, valid_set


if __name__ == '__main__':
    # clean_cache()
    dataset_config_filename = "./exp/dataset_configs/make_train_valid.json"
    train_set, valid_set = make_dataset(dataset_config_filename)

    datset_name, idx, img_names, imgs, img_ori_dim, cam_Es, cam_Ks, out_graph_mat, img_id2sub_id, sub_id2img_id, out_covis_mat, edge_subnode_idx, edge_type, edge_local_matches_n1, edge_local_matches_n2, edge_rel_Rt, node_feat, edge_feat = \
    train_set[0]
    print(len(edge_rel_Rt))