import socket, shutil, os, pickle, json, glob
from torch.utils.data import ConcatDataset
from core_dl.train_params import TrainParameters
from data.capture.capture_dataset_nonimage import CaptureDataset
import torchvision.transforms as transforms
import numpy as np

server_name = socket.gethostname()

if 'cs-gruvi-24' in server_name:
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
    yfcc_node_feat_lmdb = '/mnt/Exp_5/yfcc100/100_node_edge_feat.lmdb'

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
    train_c_datalist = [
        {'name': 'sv_seq5', 'bundle_prefix': 'bundle'},
        # {'name': 'buildings', 'bundle_prefix': 'bundle'},
        # {'name': 'buildings2', 'bundle_prefix': 'bundle'},
        {'name': 'sv_seq4', 'bundle_prefix': 'bundle'},
        {'name': 'sv_seq6', 'bundle_prefix': 'bundle'},
        {'name': 'sv_box2', 'bundle_prefix': 'bundle'},
        {'name': 'sv_statue1', 'bundle_prefix': 'bundle'},
        {'name': 'bottles_2', 'bundle_prefix': 'bundle'},
        {'name': 'bottles_4', 'bundle_prefix': 'bundle'},
        {'name': 'sv_shoes', 'bundle_prefix': 'bundle'},
        {'name': 'bottles_1', 'bundle_prefix': 'bundle'},
        {'name': 'sv_seq2', 'bundle_prefix': 'bundle'},
        {'name': 'bottles_3', 'bundle_prefix': 'bundle'},
        {'name': 'sv_seq3', 'bundle_prefix': 'bundle'},
        # {'name': 'boat', 'bundle_prefix': 'bundle'},
        {'name': 'bottles_6', 'bundle_prefix': 'bundle'},
        # {'name': 'aq', 'bundle_prefix': 'bundle'},
        {'name': 'sv_box1', 'bundle_prefix': 'bundle'},
        {'name': 'sv_seq1', 'bundle_prefix': 'bundle'},
        {'name': 'bottles_5', 'bundle_prefix': 'bundle'},
        {'name': 'furniture1', 'bundle_prefix': 'bundle'},
        {'name': 'furniture2', 'bundle_prefix': 'bundle'},
        {'name': 'furniture3', 'bundle_prefix': 'bundle'},
        {'name': 'furniture5', 'bundle_prefix': 'bundle'},
        {'name': 'furniture6', 'bundle_prefix': 'bundle'},
        {'name': 'furniture8', 'bundle_prefix': 'bundle'},
        {'name': 'furniture9', 'bundle_prefix': 'bundle'},
        {'name': 'furniture10', 'bundle_prefix': 'bundle'},
        {'name': 'furniture12', 'bundle_prefix': 'bundle'},
        {'name': 'furniture13', 'bundle_prefix': 'bundle'},
        {'name': 'furniture14', 'bundle_prefix': 'bundle'},
        {'name': 'furniture15', 'bundle_prefix': 'bundle'},
        {'name': 'sv_sv1', 'bundle_prefix': 'bundle'},
        {'name': 'sv_sv2', 'bundle_prefix': 'bundle'},
        {'name': 'sv_sv3', 'bundle_prefix': 'bundle'},
        {'name': 'sv_sv4', 'bundle_prefix': 'bundle'},
        {'name': 'sv_sv6', 'bundle_prefix': 'bundle'},
        {'name': 'sv_sv7', 'bundle_prefix': 'bundle'},
        {'name': 'sv_sv9', 'bundle_prefix': 'bundle'},
        {'name': 'sv_sv10', 'bundle_prefix': 'bundle'},
        {'name': 'sv_sv11', 'bundle_prefix': 'bundle'},
        {'name': 'furniture24', 'bundle_prefix': 'bundle'},
        {'name': 'furniture23', 'bundle_prefix': 'bundle'},
        {'name': 'furniture22', 'bundle_prefix': 'bundle'},
        {'name': 'furniture21', 'bundle_prefix': 'bundle'},
        {'name': 'furniture20', 'bundle_prefix': 'bundle'},
        {'name': 'sv_bin1', 'bundle_prefix': 'bundle'},
    ]

    valid_c_datalist = [
        {'name': 'hall', 'bundle_prefix': 'bundle'},
        {'name': 'sv_seq7', 'bundle_prefix': 'bundle'},
        {'name': 'bottles_7', 'bundle_prefix': 'bundle'},
        {'name': 'bottles_6', 'bundle_prefix': 'bundle'},
        {'name': 'furniture16', 'bundle_prefix': 'bundle'},
        {'name': 'furniture7', 'bundle_prefix': 'bundle'},
    ]

    # train_c_dataset = CaptureDataset(iccv_res_dir=cap_res_dir,
    #                                  image_dir=cap_res_dir,
    #                                  lmdb_paths=cap_res_lmdb_cache,
    #                                  node_edge_lmdb=node_edge_lmdb,
    #                                  dataset_list=train_c_datalist,
    #                                  img_max_dim=400,
    #                                  sample_res_cache=os.path.join(cap_res_dir, 'subnode_sampling_c_fast_train_30.bin'),
    #                                  sampling_num_range=[100, 160],
    #                                  load_img=False,
    #                                 #  num_nodes_mask_path=n_spt_node_mask,
    #                                  sub_graph_nodes=30)

    # valid_c_dataset = CaptureDataset(iccv_res_dir=cap_res_dir,
    #                                  image_dir=cap_res_dir,
    #                                  lmdb_paths=cap_res_lmdb_cache,
    #                                  dataset_list=valid_c_datalist,
    #                                  img_max_dim=400,
    #                                  sample_res_cache=os.path.join(cap_res_dir, 'subnode_sampling_c_fast_valid.bin'),
    #                                  sampling_num_range=[40, 80],
    #                                  sub_graph_nodes=40)

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
    yfcc_train_list = [
        {'name': 'sagrada_familia_2', 'bundle_prefix': 'yfcc', 'valid_num': 186},
        {'name': 'st_peters_square', 'bundle_prefix': 'yfcc', 'valid_num': 2264},
        {'name': 'grand_place_brussels_3', 'bundle_prefix': 'yfcc', 'valid_num': 239},
        {'name': 'sistine_chapel_ceiling_2', 'bundle_prefix': 'yfcc', 'valid_num': 78},
        {'name': 'old_town_square_prague', 'bundle_prefix': 'yfcc', 'valid_num': 1100},
        {'name': 'statue_of_liberty_1', 'bundle_prefix': 'yfcc', 'valid_num': 90},
        {'name': 'milan_cathedral', 'bundle_prefix': 'yfcc', 'valid_num': 117},
        {'name': 'hagia_sophia_interior', 'bundle_prefix': 'yfcc', 'valid_num': 827},
        {'name': 'paris_opera_2', 'bundle_prefix': 'yfcc', 'valid_num': 195},
        {'name': 'colosseum_exterior', 'bundle_prefix': 'yfcc', 'valid_num': 1881},
        {'name': 'taj_mahal_entrance', 'bundle_prefix': 'yfcc', 'valid_num': 68},
        {'name': 'pieta_michelangelo', 'bundle_prefix': 'yfcc', 'valid_num': 134},
        {'name': 'natural_history_museum_london', 'bundle_prefix': 'yfcc', 'valid_num': 43},
        {'name': 'temple_kyoto_japan', 'bundle_prefix': 'yfcc', 'valid_num': 274},
        {'name': 'united_states_capitol_rotunda', 'bundle_prefix': 'yfcc', 'valid_num': 251},
        {'name': 'united_states_capitol', 'bundle_prefix': 'yfcc', 'valid_num': 231},
        {'name': 'pike_place_market', 'bundle_prefix': 'yfcc', 'valid_num': 394},
        {'name': 'sagrada_familia_1', 'bundle_prefix': 'yfcc', 'valid_num': 214},
        {'name': 'vatican_museum_ceiling', 'bundle_prefix': 'yfcc', 'valid_num': 58},
        {'name': 'colosseum_interior', 'bundle_prefix': 'yfcc', 'valid_num': 850},
        {'name': 'temple_nara_japan', 'bundle_prefix': 'yfcc', 'valid_num': 830},
        {'name': 'sagrada_familia_3', 'bundle_prefix': 'yfcc', 'valid_num': 170},
        {'name': 'notre_dame_front_facade', 'bundle_prefix': 'yfcc', 'valid_num': 3470},
        {'name': 'sacre_coeur', 'bundle_prefix': 'yfcc', 'valid_num': 1084},
        {'name': 'palace_of_westminster', 'bundle_prefix': 'yfcc', 'valid_num': 875},
        {'name': 'palace_of_versailles_chapel', 'bundle_prefix': 'yfcc', 'valid_num': 136},
        {'name': 'taj_mahal', 'bundle_prefix': 'yfcc', 'valid_num': 1223},
        {'name': 'statue_of_liberty_2', 'bundle_prefix': 'yfcc', 'valid_num': 100},
        {'name': 'palazzo_pubblico', 'bundle_prefix': 'yfcc', 'valid_num': 182},
        {'name': 'pantheon_exterior', 'bundle_prefix': 'yfcc', 'valid_num': 1264},
        {'name': 'petra_jordan', 'bundle_prefix': 'yfcc', 'valid_num': 104},
        {'name': 'piazza_della_signoria', 'bundle_prefix': 'yfcc', 'valid_num': 105},
        {'name': 'paris_opera_1', 'bundle_prefix': 'yfcc', 'valid_num': 387},
        {'name': 'piazza_san_marco', 'bundle_prefix': 'yfcc', 'valid_num': 228},
        {'name': 'st_pauls_cathedral', 'bundle_prefix': 'yfcc', 'valid_num': 550},
        {'name': 'lincoln_memorial', 'bundle_prefix': 'yfcc', 'valid_num': 321},
        {'name': 'brandenburg_gate', 'bundle_prefix': 'yfcc', 'valid_num': 1177},
        {'name': 'florence_cathedral_side', 'bundle_prefix': 'yfcc', 'valid_num': 103},
        {'name': 'notre_dame_rosary_window', 'bundle_prefix': 'yfcc', 'valid_num': 496},
        {'name': 'national_gallery_london', 'bundle_prefix': 'yfcc', 'valid_num': 290},
        {'name': 'trevi_fountain_1', 'bundle_prefix': 'yfcc', 'valid_num': 2654},
        {'name': 'grand_central_terminal_new_york', 'bundle_prefix': 'yfcc', 'valid_num': 216},
        {'name': 'old_town_square_prague_clock', 'bundle_prefix': 'yfcc', 'valid_num': 981},
        {'name': 'mount_rushmore', 'bundle_prefix': 'yfcc', 'valid_num': 124},
        {'name': 'western_wall_jerusalem', 'bundle_prefix': 'yfcc', 'valid_num': 99},
        {'name': 'british_museum', 'bundle_prefix': 'yfcc', 'valid_num': 600},
        {'name': 'st_vitus_cathedral', 'bundle_prefix': 'yfcc', 'valid_num': 144},
        {'name': 'trevi_fountain_2', 'bundle_prefix': 'yfcc', 'valid_num': 202},
        {'name': 'westminster_abbey_1', 'bundle_prefix': 'yfcc', 'valid_num': 738},
        {'name': 'westminster_abbey_2', 'bundle_prefix': 'yfcc', 'valid_num': 226},
        {'name': 'sistine_chapel_ceiling_1', 'bundle_prefix': 'yfcc', 'valid_num': 71},
    ]

    yfcc_valid_list = [
        {'name': 'london_bridge_1', 'bundle_prefix': 'yfcc', 'valid_num': 196},
        {'name': 'london_bridge_3', 'bundle_prefix': 'yfcc', 'valid_num': 189},
        {'name': 'london_bridge_2', 'bundle_prefix': 'yfcc', 'valid_num': 185},
        {'name': 'big_ben_1', 'bundle_prefix': 'yfcc', 'valid_num': 188},
        {'name': 'big_ben_2', 'bundle_prefix': 'yfcc', 'valid_num': 163},
        {'name': 'grand_place_brussels_1', 'bundle_prefix': 'yfcc', 'valid_num': 406},
        {'name': 'grand_place_brussels_2', 'bundle_prefix': 'yfcc', 'valid_num': 330},
        {'name': 'st_peters_basilica_interior_1', 'bundle_prefix': 'yfcc', 'valid_num': 540},
        {'name': 'st_peters_basilica_interior_2', 'bundle_prefix': 'yfcc', 'valid_num': 269},
        {'name': 'blue_mosque_interior_1', 'bundle_prefix': 'yfcc', 'valid_num': 432},
        {'name': 'blue_mosque_interior_2', 'bundle_prefix': 'yfcc', 'valid_num': 166},
        {'name': 'florence_cathedral_dome_interior_1', 'bundle_prefix': 'yfcc', 'valid_num': 317},
        {'name': 'florence_cathedral_dome_interior_2', 'bundle_prefix': 'yfcc', 'valid_num': 246},
        {'name': 'pantheon_interior', 'bundle_prefix': 'yfcc', 'valid_num': 881},
        {'name': 'louvre', 'bundle_prefix': 'yfcc', 'valid_num': 624},
        {'name': 'lincoln_memorial_statue', 'bundle_prefix': 'yfcc', 'valid_num': 745},
        {'name': 'piazza_dei_miracoli', 'bundle_prefix': 'yfcc', 'valid_num': 117},
    ]

    names = [e['name'] for e in yfcc_valid_list]
    for name in names:
        print('"%s",' % name)

    exit(0)

    train_yfcc = CaptureDataset(dataset_dir=yfcc_res_dir,
                                dataset_list=yfcc_train_list,
                                img_max_dim=400,
                                img_lmdb_paths=yfcc_lmdb_cache,
                                sample_res_cache=os.path.join(yfcc_res_dir, 'train_sample_100_nodes.bin'),
                                sampling_num_range=[20, 30],
                                load_img=True,
                                load_keypt_match=True,
                                load_node_edge_feat=False,
                                sampling_undefined_edge=False,
                                sub_graph_nodes=100)

    valid_yfcc = CaptureDataset(dataset_dir=yfcc_res_dir,
                                dataset_list=yfcc_valid_list,
                                img_max_dim=400,
                                img_lmdb_paths=yfcc_lmdb_cache,
                                sample_res_cache=os.path.join(yfcc_res_dir, 'valid_sample_100_nodes.bin'),
                                sampling_num_range=[10, 20],
                                load_img=True,
                                load_keypt_match=True,
                                load_node_edge_feat=False,
                                sampling_undefined_edge=False,
                                sub_graph_nodes=100)

    """ Concatenate all dataset
    """
    train_set = ConcatDataset([train_yfcc])
    valid_set = ConcatDataset([valid_yfcc])
    
    return train_set, valid_set

if __name__ == '__main__':
    # clean_cache()
    train_set, valid_set = make_dataset()

    idx, img_names, imgs, img_ori_dim, cam_Es, cam_Ks, out_graph_mat, img_id2sub_id, sub_id2img_id, out_covis_mat, edge_subnode_idx, edge_type, edge_local_matches_n1, edge_local_matches_n2, edge_rel_Rt = train_set[0]
    print(len(edge_rel_Rt))