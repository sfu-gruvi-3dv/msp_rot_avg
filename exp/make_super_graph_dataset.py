import socket, shutil, os, pickle, json, glob
from torch.utils.data import ConcatDataset
from core_dl.train_params import TrainParameters
from data.capture.capt_data_supergraph import SupergraphDataset
import torchvision.transforms as transforms
import numpy as np

server_name = socket.gethostname()

if 'cs-gruvi-24s' in server_name:
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

    # yfcc_dataset
    yfcc_res_dir = '/mnt/Exp_5/yfcc100/set/'
    yfcc_lmdb_cache = ['/mnt/Exp_5/yfcc100/set/cache.lmdb',
                       '/mnt/Exp_5/yfcc100/set/cache_meta.bin']
    yfcc_node_feat_lmdb = '/mnt/Exp_5/yfcc100/test_node_edge_feat.lmdb'

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
    yfcc_res_dir = '/local-scratch6/CaptureData/'

elif 'jamal-50s' in server_name:

    # yfcc_dataset
    yfcc_res_dir = '/mnt/Tango/yfcc100/set/'
    yfcc_lmdb_cache = ['/mnt/Tango/yfcc100/set/cache.lmdb',
                       '/mnt/Tango/yfcc100/set/cache_meta.bin']
    yfcc_node_feat_lmdb = '/mnt/Tango/yfcc100/test_node_edge_feat.lmdb'

def clean_cache():
    paths = [
         # os.path.join(onedsfm_res_dir, 'subnode_sampling_o_train.bin'),
         # os.path.join(onedsfm_res_dir, 'subnode_sampling_o_valid.bin'),
         # os.path.join(cap_res_dir, 'subnode_sampling_c_train.bin'),
         # os.path.join(cap_res_dir, 'subnode_sampling_c_valid.bin'),
        #  os.path.join(yfcc_res_dir, 'train_sample_100_nodes_dbg.bin'),
        #  os.path.join(yfcc_res_dir, 'valid_sample_100_nodes_dbg.bin'),
         os.path.join(yfcc_res_dir, 'test_subgraphs_80n.bin'),
    ]
    for path in paths:
        if os.path.exists(path):
            os.remove(path)

def make_dataset():

    """ 1DSFM Dataset --------------------------------------------------------------------------------------------------
    """
    train_o_datalist = [
        {'name': 'Alamo'},
        {'name': 'Ellis_Island'},
        {'name': 'Madrid_Metropolis'},
        {'name': 'Montreal_Notre_Dame'},
        {'name': 'NYC_Library'},
        {'name': 'Piazza_del_Popolo'},
        {'name': 'Piccadilly'},
        {'name': 'Roman_Forum'},
        {'name': 'Tower_of_London'},
        {'name': 'Union_Square'},
        {'name': 'Vienna_Cathedral'},
        {'name': 'Yorkminster'},
    ]

    valid_o_datalist = [{'name': 'Gendarmenmarkt'},
                        {'name': 'Trafalgar'}],

    # train_o_dataset = OneDSFMDataset(iccv_res_dir=onedsfm_res_dir,
    #                                  image_dir=onedsfm_img_dir,
    #                                  lmdb_paths=onedsfm_lmdb_cache,
    #                                  dataset_list=train_o_datalist,
    #                                  img_max_dim=480,
    #                                  sample_res_cache=os.path.join(onedsfm_res_dir, 'subnode_sampling_o_train.bin'),
    #                                  sampling_num_range=[200, 300],
    #                                  sub_graph_nodes=30)
    #
    # valid_o_dataset = OneDSFMDataset(iccv_res_dir=onedsfm_res_dir,
    #                                  image_dir=onedsfm_img_dir,
    #                                  lmdb_paths=onedsfm_lmdb_cache,
    #                                  dataset_list=valid_o_datalist,
    #                                  img_max_dim=480,
    #                                  sample_res_cache=os.path.join(onedsfm_res_dir, 'subnode_sampling_o_valid.bin'),
    #                                  sampling_num_range=[200, 300],
    #                                  sub_graph_nodes=30)

    """ Yfcc Dataset ---------------------------------------------------------------------------------------------------
    """
    yfcc_valid_list = [
        # {'name': 'london_bridge_2', 'bundle_prefix': 'yfcc', 'valid_num': 185},
        # {'name': 'florence_cathedral_side', 'bundle_prefix': 'yfcc', 'valid_num': 103},
        # {'name': 'big_ben_1', 'bundle_prefix': 'yfcc', 'valid_num': 188},
        # {'name': 'sistine_chapel_ceiling_1', 'bundle_prefix': 'yfcc', 'valid_num': 71},
        {'name': 'st_peters_basilica_interior_1', 'bundle_prefix': 'yfcc', 'valid_num': 540},
        # {'name': 'blue_mosque_interior_1', 'bundle_prefix': 'yfcc', 'valid_num': 432},
        # {'name': 'grand_place_brussels_1', 'bundle_prefix': 'yfcc', 'valid_num': 406},
        # {'name': 'florence_cathedral_dome_interior_1', 'bundle_prefix': 'yfcc', 'valid_num': 317}
    ]

    valid_yfcc = SupergraphDataset(dataset_dir=yfcc_res_dir,
                                   dataset_list=yfcc_valid_list,
                                   img_max_dim=400,
                                   img_lmdb_paths=yfcc_lmdb_cache,
                                   node_edge_lmdb_path=yfcc_node_feat_lmdb,
                                   sub_graph_nodes=150,
                                   sample_res_cache=os.path.join(yfcc_res_dir, 'test_subgraphs_150n.bin'),
                                   load_img=True,
                                   load_keypt_match=True,
                                   load_node_edge_feat=True)

    """ Concatenate all dataset
    """
    valid_set = ConcatDataset([valid_yfcc])

    return valid_set

if __name__ == '__main__':
    valid_set = make_dataset()

    dataset_name, idx, img_names, imgs, img_ori_dim, cam_Es, cam_Ks, out_graph_mat, img_id2sub_id, sub_id2img_id, out_covis_mat, edge_subnode_idx, edge_type, edge_local_matches_n1, edge_local_matches_n2, edge_rel_Rt, edge_rel_err, node_feats, edge_feats = valid_set[0]
    print(len(edge_rel_Rt))