import socket, shutil, os, pickle, json, glob
from torch.utils.data import ConcatDataset
from core_dl.train_params import TrainParameters
from data.oned_sfm.local_feat_dataset_fast import OneDSFMDataset
from data.capture.capture_dataset_cache_node_edge_feat import CaptureDataset
# from data.capture.capture_dataset_rel_fast_marked import CaptureDataset
import torchvision.transforms as transforms
import numpy as np

server_name = socket.gethostname()

if server_name == 'cs-gruvi-24-cmpt-sfu-ca':
    # 1dsfm dataset dir
    onedsfm_res_dir = '/mnt/Tango/pg/ICCV15_raw/'
    onedsfm_img_dir = '/mnt/Exp_2/1dsfm/imgs/'
    onedsfm_lmdb_cache = None

    # captured dataset
    cap_res_dir = '/mnt/Exp_5/pg_train/'
    cap_res_lmdb_cache = ['/mnt/Exp_5/pg_train/cache.lmdb',
                          '/mnt/Exp_5/pg_train/cache_meta.bin']

    node_edge_lmdb = {'node': '/mnt/Exp_5/pg_train/node_f.lmdb',
                      'edge': '/mnt/Exp_5/pg_train/edge_f.lmdb'}

    n_spt_node_mask = '/mnt/Exp_4/graph_dbg/spt_nodes_count.npy'

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

    n_spt_node_mask = '/local-scratch5/graph_dbg/spt_nodes_count.npy'

def clean_cache():
    paths = [os.path.join(onedsfm_res_dir, 'subnode_sampling_o_train.bin'),
             os.path.join(onedsfm_res_dir, 'subnode_sampling_o_valid.bin'),
             os.path.join(cap_res_dir, 'subnode_sampling_c_train.bin'),
             os.path.join(cap_res_dir, 'subnode_sampling_c_valid.bin')
             ]
    for path in paths:
        if os.path.exists(path):
            os.remove(path)

def make_dataset():
    """ Captured Dataset -----------------------------------------------------------------------------------------------
    """
    train_c_datalist = [
                        # {'name': 'sv_seq5', 'bundle_prefix': 'bundle'},
                        # {'name': 'bottles_2', 'bundle_prefix': 'bundle'},
                        # {'name': 'buildings', 'bundle_prefix': 'bundle'},
                        # # {'name': 'buildings2', 'bundle_prefix': 'bundle'},
                        # {'name': 'sv_seq4', 'bundle_prefix': 'bundle'},
                        # {'name': 'sv_seq6', 'bundle_prefix': 'bundle'},
                        # {'name': 'sv_box2', 'bundle_prefix': 'bundle'},
                        {'name': 'sv_statue1', 'bundle_prefix': 'bundle'},
                        # {'name': 'bottles_4', 'bundle_prefix': 'bundle'},
                        # {'name': 'sv_shoes', 'bundle_prefix': 'bundle'},
                        # {'name': 'bottles_1', 'bundle_prefix': 'bundle'},
                        # {'name': 'sv_seq2', 'bundle_prefix': 'bundle'},
                        # {'name': 'bottles_3', 'bundle_prefix': 'bundle'},
                        # {'name': 'sv_seq3', 'bundle_prefix': 'bundle'},
                        # {'name': 'boat', 'bundle_prefix': 'bundle'},
                        # {'name': 'bottles_6', 'bundle_prefix': 'bundle'},
                        # {'name': 'aq', 'bundle_prefix': 'bundle'},
                        # {'name': 'sv_box1', 'bundle_prefix': 'bundle'},
                        # {'name': 'sv_seq1', 'bundle_prefix': 'bundle'},
                        # {'name': 'bottles_5', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture1', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture2', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture3', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture5', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture6', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture8', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture9', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture10', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture12', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture13', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture14', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture15', 'bundle_prefix': 'bundle'},
                        # {'name': 'furniture17', 'bundle_prefix': 'bundle'},
                        ]

    # valid_c_datalist = [
    #     # {'name': 'Westerminster_colmap_manual', 'bundle_prefix': 'bundle'},
    #     # {'name': 'hall', 'bundle_prefix': 'bundle'},
    #     # {'name': 'sv_seq7', 'bundle_prefix': 'bundle'},
    #     # {'name': 'bottles_7', 'bundle_prefix': 'bundle'},
    #     # {'name': 'bottles_6', 'bundle_prefix': 'bundle'},
    #     # {'name': 'furniture16', 'bundle_prefix': 'bundle'},
    #     {'name': 'furniture7', 'bundle_prefix': 'bundle'},
    # ]

    train_c_dataset = CaptureDataset(iccv_res_dir=cap_res_dir,
                                     image_dir=cap_res_dir,
                                     lmdb_paths=cap_res_lmdb_cache,
                                     node_edge_lmdb=node_edge_lmdb,
                                     dataset_list=train_c_datalist,
                                     img_max_dim=400,
                                     # sample_res_cache=os.path.join(cap_res_dir, 'subnode_sampling_c_train.bin'),
                                     sampling_num_range=[100, 150],
                                    #  num_nodes_mask_path=n_spt_node_mask,
                                     sub_graph_nodes=48)

    # valid_c_dataset = CaptureDataset(iccv_res_dir=cap_res_dir,
    #                                  image_dir=cap_res_dir,
    #                                  lmdb_paths=cap_res_lmdb_cache,
    #                                  dataset_list=valid_c_datalist,
    #                                  img_max_dim=400,
    #                                  # sample_res_cache=os.path.join(cap_res_dir, 'subnode_sampling_c_valid.bin'),
    #                                  sampling_num_range=[20, 30],
    #                                  sub_graph_nodes=30)

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

    """ Concate all dataset
    """
    train_set = ConcatDataset([train_c_dataset])
    # valid_set = ConcatDataset([valid_c_dataset])

    return train_set, None