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
from data.capture.capt_data_supergraph import SupergraphDataset
import sys
server_name = socket.gethostname()

datasets_config_filename = "./exp/dataset_configs/dataset_config_default.json"

def clean_cache(datasets_config, train_valid_config):
    paths = []

    for dataset_name, val in train_valid_config["valid"].items():
        paths.append(os.path.join(datasets_config[dataset_name]["sampled_subgraph_dir"],
                                  val["sample_res_cache"]))
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def make_dataset(json_config, load_img, load_node_edge_feat):
    if json_config is None:
        json_config = "./exp/dataset_configs/make_train_valid.json"

    with open(json_config, "r") as fin:
        train_valid_config = json.load(fin)
    with open(datasets_config_filename, "r") as fin:
        datasets_config = json.load(fin)

    if train_valid_config["clean_cache"]:
        clean_cache(datasets_config, train_valid_config)

    test_dataset_list = []
    
    for dataset_lib, dataset_dict in train_valid_config["valid"].items():
        valid_dataset = []
        for train in dataset_dict["dataset_list"]:
            for it in datasets_config[dataset_lib]["dataset_list"]:
                if it["name"] == train:
                    valid_dataset.append(it)
                    break
        
        test_dataset_list.append(SupergraphDataset(dataset_dir=datasets_config[dataset_lib]["res_dir"],
                                                   img_lmdb_paths=datasets_config[dataset_lib]["lmdb_cache"],
                                                   node_edge_lmdb_path=None if dataset_dict[
                                                                                 "node_edge_lmdb_name"] is None else os.path.join(
                                                   datasets_config[dataset_lib]["node_edge_feat_dir"],
                                                   valid_dataset[0]['name'],
                                                   dataset_dict["node_edge_lmdb_name"]),
                                                   dataset_list=valid_dataset,
                                                   img_max_dim=dataset_dict["img_max_dim"],
                                                   sample_res_cache=None if dataset_dict[
                                                                              "sample_res_cache"] is None else os.path.join(
                                                   datasets_config[dataset_lib]["sampled_subgraph_dir"],
                                                   valid_dataset[0]['name'],
                                                   dataset_dict["sample_res_cache"]),
                                                   load_img=load_img,
                                                   load_keypt_match=load_img,
                                                   load_node_edge_feat=load_node_edge_feat,
                                                   sub_graph_nodes=dataset_dict["sub_graph_nodes"],
                                                   subgraph_edge_cover_ratio=dataset_dict["subgraph_edge_cover_ratio"]
                                                   )
                                  )

    valid_set = None if len(test_dataset_list) == 0 else ConcatDataset(test_dataset_list)
    return valid_set

if __name__ == '__main__':
    # clean_cache()
    dataset_config_filename = "./test_config/onedsfm_madrid_150nodes.json"
    valid_set = make_dataset(dataset_config_filename, load_img=True, load_node_edge_feat=False)
    print(len(valid_set))

    dataset_name, idx, img_names, imgs, img_ori_dim, cam_Es, cam_Ks, out_graph_mat, valid_id2sub_id, sub_id2valid_id, img_id2sub_id, sub_id2img_id, out_covis_mat, edge_subnode_idx, edge_type, edge_local_matches_n1, edge_local_matches_n2, edge_rel_Rt, edge_rel_err, node_feats, edge_feats = \
    valid_set[0]
