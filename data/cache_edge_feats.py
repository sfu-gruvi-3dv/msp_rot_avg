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
import torchvision.transforms as transforms

# from exp.local_feat_gat_trainbox import LocalGlobalGATTrainBox, cluster_pool
# from exp.no_local_feat_gat_trainbox import LocalGlobalGATTrainBox
from exp.local_feat_notgat_trainbox import LocalGlobalVLADTrainBox as LocalGlobalGATTrainBox_Prior
# from exp.rot_avg_org_loss_trainbox import LocalGlobalVLADTrainBox as LocalGlobalGATTrainBox
from exp.rot_avg_refined_finenet import *

# In[1] Dataset
""" Captured Dataset
"""
cap_res_dir = '/mnt/Exp_5/pg_train'
lmdb_path = '/mnt/Exp_5/pg_train/cache.lmdb'
lmdb_meta_path = '/mnt/Exp_5/pg_train/cache_meta.bin'
output_h5_file_name = 'node_edge_feat.h5'

run_dev_ids = [0, 0]
max_batch_size = 30

dataset_list = [
    # {'name': 'sv_seq5', 'bundle_prefix': 'bundle'},
    # {'name': 'bottles_2', 'bundle_prefix': 'bundle'},
    # # {'name': 'buildings', 'bundle_prefix': 'bundle'},
    # # {'name': 'buildings2', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_seq4', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_seq6', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_box2', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_statue1', 'bundle_prefix': 'bundle'},
    # {'name': 'bottles_4', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_shoes', 'bundle_prefix': 'bundle'},
    # {'name': 'bottles_1', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_seq2', 'bundle_prefix': 'bundle'},
    # {'name': 'bottles_3', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_seq3', 'bundle_prefix': 'bundle'},
    # # {'name': 'boat', 'bundle_prefix': 'bundle'},
    # {'name': 'bottles_6', 'bundle_prefix': 'bundle'},
    # # {'name': 'aq', 'bundle_prefix': 'bundle'},
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
    # {'name': 'sv_sv1', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_sv2', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_sv3', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_sv4', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_sv6', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_sv7', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_sv9', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_sv10', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_sv11', 'bundle_prefix': 'bundle'},
    # {'name': 'furniture24', 'bundle_prefix': 'bundle'},
    # {'name': 'furniture23', 'bundle_prefix': 'bundle'},
    # {'name': 'furniture22', 'bundle_prefix': 'bundle'},
    # {'name': 'furniture21', 'bundle_prefix': 'bundle'},
    # {'name': 'furniture20', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_bin1', 'bundle_prefix': 'bundle'},
    # {'name': 'hall', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_seq7', 'bundle_prefix': 'bundle'},
    # {'name': 'bottles_7', 'bundle_prefix': 'bundle'},
    # {'name': 'bottles_6', 'bundle_prefix': 'bundle'},
    # {'name': 'furniture16', 'bundle_prefix': 'bundle'},
    # {'name': 'furniture7', 'bundle_prefix': 'bundle'},
    {'name': 'test_seq', 'bundle_prefix': 'bundle'},
    # {'name': 'sv_box10', 'bundle_prefix': 'bundle'},
    # {'name': 'test_seq', 'bundle_prefix': 'bundle'}
]

maximum_img_dim = 480
transform_func = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

print('Processing on gpu:%s with %d scenes' % (str(run_dev_ids), len(dataset_list)))
""" Function -----------------------------------------------------------------------------------------------------------
"""
def split_node_data(frame_list, img_lmdb, img_lmdb_meta, dataset_name, transform_func=None):
    imgs = []
    img_ori_dim = []
    for frame_name in frame_list:

        # load image
        img_key = dataset_name + '/' + frame_name
        img = img_lmdb.read_ndarray_by_key(img_key, dtype=np.uint8)
        h, w = img_lmdb_meta[img_key]['dim']
        res_h, res_w = img_lmdb_meta[img_key]['lmdb_dim']
        img = img.reshape(int(res_h), int(res_w), 3)
        img_ori_dim.append((torch.from_numpy(numpy.asarray([h])), torch.from_numpy(numpy.asarray([w]))))

        # resize the image, and convert to torch Tensor
        max_dim = res_h if res_h > res_w else res_w
        down_factor = float(maximum_img_dim) / float(max_dim)
        img = cv2.resize(img, dsize=(
            int(res_w * down_factor), int(res_h * down_factor)))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        if transform_func is not None:
            img = transform_func(img)

        imgs.append(img.unsqueeze(0))

    return imgs

def split_data(n1, n1_neighbors, img_lmdb, img_lmdb_meta, edge_cache, frame_list, dataset_name, transform_func=None):
    img_set = [n1] + n1_neighbors

    imgs = []
    img_ori_dim = []

    for img_id in img_set:

        # load image
        img_key = dataset_name + '/' + frame_list[img_id]
        img = img_lmdb.read_ndarray_by_key(img_key, dtype=np.uint8)
        h, w = img_lmdb_meta[img_key]['dim']
        res_h, res_w = img_lmdb_meta[img_key]['lmdb_dim']
        img = img.reshape(int(res_h), int(res_w), 3)
        img_ori_dim.append((torch.from_numpy(numpy.asarray([h])), torch.from_numpy(numpy.asarray([w]))))

        # resize the image, and convert to torch Tensor
        max_dim = res_h if res_h > res_w else res_w
        down_factor = float(maximum_img_dim) / float(max_dim)
        img = cv2.resize(img, dsize=(
            int(res_w * down_factor), int(res_h * down_factor)))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        if transform_func is not None:
            img = transform_func(img)

        imgs.append(img.unsqueeze(0))

    # build edge idx
    edge_node_idx = []

    # keypoint matches
    edge_local_matches_n1 = []
    edge_local_matches_n2 = []
    for n2_idx, n2 in enumerate(n1_neighbors):
        if '%d-%d' % (n1, n2) in edge_cache:
            key = '%d-%d' % (n1, n2)
            pts1 = torch.from_numpy(edge_cache[key]['n1_feat_pos'])
            pts2 = torch.from_numpy(edge_cache[key]['n2_feat_pos'])
        elif '%d-%d' % (n2, n1) in edge_cache:
            key = '%d-%d' % (n2, n1)
            pts1 = torch.from_numpy(edge_cache[key]['n2_feat_pos'])
            pts2 = torch.from_numpy(edge_cache[key]['n1_feat_pos'])
        else:
            raise Exception('No Found (%d, %d)' % (n1, n2))
        edge_local_matches_n1.append(pts1.unsqueeze(0))
        edge_local_matches_n2.append(pts2.unsqueeze(0))
        edge_node_idx.append((torch.from_numpy(numpy.asarray([0])), torch.from_numpy(numpy.asarray([n2_idx]))))

    return imgs, img_ori_dim, edge_node_idx, edge_local_matches_n1, edge_local_matches_n2

""" Network ------------------------------------------------------------------------------------------------------------
"""
train_params = TrainParameters()
train_params.DEV_IDS = run_dev_ids
train_params.VERBOSE_MODE = True
prior_box = LocalGlobalGATTrainBox_Prior(train_params=train_params, ckpt_path_dict={
    'vlad': '/mnt/Exp_4/valid_cache/netvlad_vgg16.tar',
    'ckpt': '/mnt/Exp_4/valid_cache/iter_nogat.pth.tar'
})
prior_box._prepare_eval()


""" Pipeline -----------------------------------------------------------------------------------------------------------
"""
img_lmdb = LMDBModel(lmdb_path, lock=False, read_only=True)
img_lmdb_meta = pickle.load(open(lmdb_meta_path, 'rb'))

# load data
for dataset in dataset_list:
    dataset_name = dataset['name']
    bundle_prefix = dataset['bundle_prefix']

    print("Processing on %s" % dataset_name)

    # load edge cache
    with open(os.path.join(cap_res_dir, dataset_name, 'edge_feat_pos_cache.bin'), 'rb') as f:
        edge_cache = pickle.load(f)

    # load frame list
    frame_list = read_image_list(os.path.join(cap_res_dir, dataset_name, bundle_prefix + '.list.txt'))

    # build adjacent list
    num_edges = 0
    adj_list = [[] for x in frame_list]
    for e_k in edge_cache.keys():
        n1, n2 = e_k.split('-')
        n1, n2 = int(n1), int(n2)
        adj_list[n1].append(n2)
        num_edges += 1
    print('Number of edges: %d' % num_edges)

    # create output file
    output_h5_file = h5py.File(os.path.join(cap_res_dir, dataset_name, output_h5_file_name), 'w')
    edge_feat_group = output_h5_file.create_group('edge_feat')
    node_feat_group = output_h5_file.create_group('node_feat')

    # image features
    num_process_batches = (len(frame_list) // max_batch_size) + 1
    for batch in range(num_process_batches):
        start_idx = batch * max_batch_size
        end_idx = batch * max_batch_size + max_batch_size
        if end_idx > len(frame_list):
            end_idx = len(frame_list)

        frame_sub_list = frame_list[start_idx: end_idx]
        imgs = split_node_data(frame_sub_list, img_lmdb, img_lmdb_meta, dataset_name, transform_func=transform_func)
        node_feats = prior_box.extract_node_feat(imgs)

        for i in range(len(frame_sub_list)):
            n1_node_feat = node_feats[i].view(-1).detach().cpu().numpy()
            n1 = start_idx + i
            node_feat_group.create_dataset('%d' % (n1), data=n1_node_feat)

    # load image one by one
    n1 = 0
    for n1_neighbors in tqdm(adj_list):

        num_process_batches = (len(n1_neighbors) // max_batch_size) + 1
        for batch in range(num_process_batches):
            start_idx = batch*max_batch_size
            end_idx = batch*max_batch_size + max_batch_size
            n1_sub_neighbors = n1_neighbors[start_idx: end_idx]
            if end_idx > len(n1_sub_neighbors):
                end_idx = len(n1_sub_neighbors)

            if len(n1_sub_neighbors) == 0:
                continue

            # split data
            samples = split_data(n1, n1_sub_neighbors, img_lmdb, img_lmdb_meta, edge_cache,
                                 frame_list=frame_list, dataset_name=dataset_name, transform_func=transform_func)

            # extract features from node and edge
            with torch.no_grad():
                node_feats, edge_feats = prior_box.extract_edge_feats(samples)
                node_feats = node_feats.detach().cpu().numpy()
                edge_feats = edge_feats.detach().cpu().numpy()

            # dump to hdf5
            for e_i, n2 in enumerate(n1_sub_neighbors):
                key = '%d-%d' % (n1, n2)
                edge_feat = edge_feats[e_i]
                edge_feat_group.create_dataset(key, data=edge_feat)

        n1 += 1

    output_h5_file.close()

