import socket, shutil, os, pickle, json, glob
from torch.utils.data import ConcatDataset
from core_dl.train_params import TrainParameters
from data.capture.capture_dataset_nonimage import CaptureDataset
import torchvision.transforms as transforms
import numpy as np

server_name = socket.gethostname()

if server_name == 'cs-gruvi-24-cmpt-sfu-ca':
    # 1dsfm dataset dir
    onedsfm_res_dir = '/mnt/Exp_5/onedsfm_train/'
    onedsfm_img_dir = '/mnt/Exp_2/1dsfm/imgs/'
    onedsfm_lmdb = [
        '/mnt/Exp_5/onedsfm_train/cache.lmdb',
        '/mnt/Exp_5/onedsfm_train/cache_meta.bin'
    ]

    # captured dataset
    cap_res_dir = '/mnt/Exp_5/pg_train/'
    cap_res_lmdb_cache = [
        '/mnt/Exp_5/pg_train/cache.lmdb',
        '/mnt/Exp_5/pg_train/cache_meta.bin'
    ]
    # node_edge_lmdb = {'node': '/mnt/Exp_5/pg_train/node_f.lmdb',
    #                   'edge': '/mnt/Exp_5/pg_train/edge_f.lmdb'}

    # yfcc_dataset
    yfcc_res_dir = '/mnt/Exp_5/yfcc100/set/'
    yfcc_lmdb_cache = ['/mnt/Exp_5/yfcc100/set/cache.lmdb',
                       '/mnt/Exp_5/yfcc100/set/cache_meta.bin']
    yfcc_node_feat_lmdb = '/mnt/Exp_5/yfcc100/64_node_edge_feat.lmdb'

elif server_name == 'cs-guv-gpu02':
    # 1dsfm dataset dir
    onedsfm_res_dir = '/local-scratch6/1dsfm/iccv15_raw'
    onedsfm_img_dir = '/local-scratch6/1dsfm/images'
    onedsfm_lmdb_cache = ['/local-scratch6/1dsfm/images/cache.lmdb',
                          '/local-scratch6/1dsfm/images/cache_meta.bin']

    # captured dataset
    cap_res_dir = '/local-scratch6/CaptureData/'
    cap_res_lmdb_cache = ['/local-scratch6/CaptureData/cache.lmdb',
                          '/local-scratch6/CaptureData/cache_meta.bin']
    node_edge_lmdb = {'node': '/local-scratch6/CaptureData/node_f.lmdb',
                      'edge': '/local-scratch6/CaptureData/edge_f.lmdb'}

    # yfcc_dataset
    yfcc_res_dir = '/local-scratch6/yfcc100/set/'
    yfcc_lmdb_cache = ['/local-scratch6/yfcc100/cache.lmdb',
                       '/local-scratch6/yfcc100/cache_meta.bin']
def clean_cache():
    paths = [
         # os.path.join(onedsfm_res_dir, 'subnode_sampling_o_train.bin'),
         # os.path.join(onedsfm_res_dir, 'subnode_sampling_o_valid.bin'),
         # os.path.join(cap_res_dir, 'subnode_sampling_c_train.bin'),
         # os.path.join(cap_res_dir, 'subnode_sampling_c_valid.bin'),
         os.path.join(yfcc_res_dir, 'sample_80_nodes.bin'),
         os.path.join(yfcc_res_dir, 'sample_40_nodes.bin'),
    ]
    for path in paths:
        if os.path.exists(path):
            os.remove(path)

def make_dataset():
    """ Captured Dataset -----------------------------------------------------------------------------------------------
    """

    """ 1DSFM Dataset --------------------------------------------------------------------------------------------------
    """
    train_o_datalist = [
        {'name': 'Alamo', 'bundle_prefix': '1dsfm'},
        {'name': 'Ellis_Island', 'bundle_prefix': '1dsfm'},
        {'name': 'Madrid_Metropolis', 'bundle_prefix': '1dsfm'},
        {'name': 'Montreal_Notre_Dame', 'bundle_prefix': '1dsfm'},
        {'name': 'NYC_Library', 'bundle_prefix': '1dsfm'},
        {'name': 'Piazza_del_Popolo', 'bundle_prefix': '1dsfm'},
        {'name': 'Piccadilly', 'bundle_prefix': '1dsfm'},
        {'name': 'Roman_Forum', 'bundle_prefix': '1dsfm'},
        {'name': 'Tower_of_London', 'bundle_prefix': '1dsfm'},
        {'name': 'Union_Square', 'bundle_prefix': '1dsfm'},
        {'name': 'Vienna_Cathedral', 'bundle_prefix': '1dsfm'},
        {'name': 'Yorkminster', 'bundle_prefix': '1dsfm'},
    ]

    valid_o_datalist = [
        {'name': 'Gendarmenmarkt', 'bundle_prefix': '1dsfm'},
        {'name': 'Trafalgar', 'bundle_prefix': '1dsfm'}
    ]

    # train_1dsfm = CaptureDataset(dataset_dir=onedsfm_res_dir,
    #                             dataset_list=train_o_datalist,
    #                             img_max_dim=400,
    #                             img_lmdb_paths=yfcc_lmdb_cache,
    #                             sample_res_cache=os.path.join(onedsfm_res_dir, 'train_sample_64_nodes.bin'),
    #                             sampling_num_range=[50, 80],
    #                             load_img=True,
    #                             load_keypt_match=True,
    #                             sampling_undefined_edge=False,
    #                             sub_graph_nodes=64)
    #
    # valid_1dsfm = CaptureDataset(dataset_dir=onedsfm_res_dir,
    #                             dataset_list=valid_o_datalist,
    #                             img_max_dim=400,
    #                             img_lmdb_paths=yfcc_lmdb_cache,
    #                             sample_res_cache=os.path.join(onedsfm_res_dir, 'valid_sample_64_nodes.bin'),
    #                             sampling_num_range=[20, 30],
    #                             load_img=True,
    #                             load_keypt_match=True,
    #                             sampling_undefined_edge=False,
    #                             sub_graph_nodes=64)


    """ Yfcc Dataset ---------------------------------------------------------------------------------------------------
    """
    yfcc_valid_list = [
        # {'name': 'westminster_abbey_2', 'bundle_prefix': 'yfcc', 'valid_num': 226},
        # {'name': 'london_bridge_2', 'bundle_prefix': 'yfcc', 'valid_num': 185},
        # {'name': 'florence_cathedral_side', 'bundle_prefix': 'yfcc', 'valid_num': 103},
        # {'name': 'big_ben_1', 'bundle_prefix': 'yfcc', 'valid_num': 188},
        # {'name': 'sistine_chapel_ceiling_1', 'bundle_prefix': 'yfcc', 'valid_num': 71},
        {'name': 'st_peters_basilica_interior_1', 'bundle_prefix': 'yfcc', 'valid_num': 540},
        # {'name': 'blue_mosque_interior_1', 'bundle_prefix': 'yfcc', 'valid_num': 432},
        # {'name': 'grand_place_brussels_1', 'bundle_prefix': 'yfcc', 'valid_num': 406},
        # {'name': 'florence_cathedral_dome_interior_1', 'bundle_prefix': 'yfcc', 'valid_num': 317},
        # {'name': 'piazza_dei_miracoli', 'bundle_prefix': 'yfcc', 'valid_num': 117},
    ]

    valid_yfcc = CaptureDataset(dataset_dir=yfcc_res_dir,
                                dataset_list=yfcc_valid_list,
                                img_max_dim=400,
                                img_lmdb_paths=yfcc_lmdb_cache,
                                node_edge_lmdb_path=yfcc_node_feat_lmdb,
                                sampling_num_range=[4, 8],
                                load_img=False,
                                load_keypt_match=False,
                                load_node_edge_feat=True,
                                sampling_undefined_edge=False,
                                sub_graph_nodes=80)


    """ Concatenate all dataset
    """
    # train_set = ConcatDataset([train_yfcc])
    valid_set = ConcatDataset([valid_yfcc])

    return None, valid_set

if __name__ == '__main__':
    # clean_cache()
    train_set, valid_set = make_dataset()

    idx, img_names, imgs, img_ori_dim, cam_Es, cam_Ks, out_graph_mat, img_id2sub_id, sub_id2img_id, out_covis_mat, edge_subnode_idx, edge_type, edge_local_matches_n1, edge_local_matches_n2, edge_rel_Rt = train_set[0]
    print(len(edge_rel_Rt))