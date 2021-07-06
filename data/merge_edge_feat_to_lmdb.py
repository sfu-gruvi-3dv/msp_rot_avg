import h5py, os, pickle
import numpy as np
from core_io.lmdb_writer import LMDBWriter

""" Captured Dataset
"""
cap_res_dir = '/mnt/Exp_5/pg_train'
lmdb_path = '/mnt/Exp_5/pg_train/cache.lmdb'
lmdb_meta_path = '/mnt/Exp_5/pg_train/cache_meta.bin'
output_h5_file_name = 'node_edge_feat.h5'

out_edge_f_lmdb_path = '/mnt/Exp_5/pg_train/edge_f.lmdb'
out_node_f_lmdb_path = '/mnt/Exp_5/pg_train/node_f.lmdb'

dataset_list = [
    {'name': 'sv_seq5', 'bundle_prefix': 'bundle'},
    {'name': 'bottles_2', 'bundle_prefix': 'bundle'},
    {'name': 'buildings', 'bundle_prefix': 'bundle'},
    # {'name': 'buildings2', 'bundle_prefix': 'bundle'},
    {'name': 'sv_seq4', 'bundle_prefix': 'bundle'},
    {'name': 'sv_seq6', 'bundle_prefix': 'bundle'},
    {'name': 'sv_box2', 'bundle_prefix': 'bundle'},
    {'name': 'sv_statue1', 'bundle_prefix': 'bundle'},
    {'name': 'bottles_4', 'bundle_prefix': 'bundle'},
    {'name': 'sv_shoes', 'bundle_prefix': 'bundle'},
    {'name': 'bottles_1', 'bundle_prefix': 'bundle'},
    {'name': 'sv_seq2', 'bundle_prefix': 'bundle'},
    {'name': 'bottles_3', 'bundle_prefix': 'bundle'},
    {'name': 'sv_seq3', 'bundle_prefix': 'bundle'},
    {'name': 'boat', 'bundle_prefix': 'bundle'},
    {'name': 'bottles_6', 'bundle_prefix': 'bundle'},
    {'name': 'aq', 'bundle_prefix': 'bundle'},
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
    {'name': 'hall', 'bundle_prefix': 'bundle'},
    {'name': 'sv_seq7', 'bundle_prefix': 'bundle'},
    {'name': 'bottles_7', 'bundle_prefix': 'bundle'},
    {'name': 'bottles_6', 'bundle_prefix': 'bundle'},
    {'name': 'furniture16', 'bundle_prefix': 'bundle'},
    {'name': 'furniture7', 'bundle_prefix': 'bundle'},
    {'name': 'furniture10', 'bundle_prefix': 'bundle'},
    {'name': 'sv_box9', 'bundle_prefix': 'bundle'},
    {'name': 'sv_box10', 'bundle_prefix': 'bundle'},
    {'name': 'furniture19', 'bundle_prefix': 'bundle'},
    {'name': 'furniture20', 'bundle_prefix': 'bundle'},
    {'name': 'furniture21', 'bundle_prefix': 'bundle'},
    {'name': 'furniture22', 'bundle_prefix': 'bundle'},
    {'name': 'furniture23', 'bundle_prefix': 'bundle'},
    {'name': 'furniture24', 'bundle_prefix': 'bundle'},
    {'name': 'test_seq', 'bundle_prefix': 'bundle'},
    # {'name': 'hall', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_seq7', 'bundle_prefix': 'bundle'},
    # {'name': 'bottles_7', 'bundle_prefix': 'bundle'},
    # {'name': 'bottles_6', 'bundle_prefix': 'bundle'},
    # {'name': 'furniture16', 'bundle_prefix': 'bundle'},
    # {'name': 'furniture7', 'bundle_prefix': 'bundle'},
]


""" Pipeline -----------------------------------------------------------------------------------------------------------
"""
out_edge_feat_lmdb = LMDBWriter(out_edge_f_lmdb_path)
out_node_feat_lmdb = LMDBWriter(out_node_f_lmdb_path)

# load data
for dataset in dataset_list:
    dataset_name = dataset['name']
    bundle_prefix = dataset['bundle_prefix']

    print("Processing on %s" % dataset_name)

    # load edge cache
    with open(os.path.join(cap_res_dir, dataset_name, 'edge_feat_pos_cache.bin'), 'rb') as f:
        edge_cache = pickle.load(f)

    # load node, edge feature from h5 file
    h5_file = h5py.File(os.path.join(cap_res_dir, dataset_name, output_h5_file_name), 'r')
    edge_feat_group = h5_file.get('edge_feat')
    node_feat_group = h5_file.get('node_feat')

    # add node feat
    node_keys = node_feat_group.keys()
    for key in node_keys:
        lmdb_keyname = '%s,%s' % (dataset_name, key)
        node_feat = node_feat_group.get(key).value
        out_node_feat_lmdb.write_array(lmdb_keyname, node_feat.ravel())

    # add edge feat
    for e_k in edge_cache.keys():
        n1, n2 = e_k.split('-')
        n1, n2 = int(n1), int(n2)
        edge_feat = edge_feat_group.get('%d-%d' % (n1, n2)).value
        edge_feat = np.asarray(edge_feat)
        out_edge_feat_lmdb.write_array('%s,%d-%d' % (dataset_name, n1, n2), edge_feat.ravel())

        n1 += 1

    h5_file.close()

out_node_feat_lmdb.close_session()
out_edge_feat_lmdb.close_session()