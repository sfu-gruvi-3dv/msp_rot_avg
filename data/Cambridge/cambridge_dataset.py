# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os, sys, cv2, pickle
import numpy as np
import tqdm
import torch
from core_io.lmdb_reader import LMDBModel
from torch.utils.data import Dataset
from core_io.ply_io import load_pointcloud_from_ply
from data.ambi.read_helper import *
from data.ambi.ambi_parser import *
from evaluator.basic_metric import rel_distance, rel_rot_angle
from sampling import *
from tqdm import tqdm
from core_dl.expr_ctx import ExprCtx
from data.oned_sfm.SceneModel import ODSceneModel
import torchvision.transforms as transforms


# %%
def drawSamplingOutMat(imgs, cam_Es, cam_Cs, out_graph_mat):
    cam_Es = cam_Es.cpu().detach().numpy()
    cam_Cs = cam_Cs.cpu().detach().numpy()
    out_graph_mat = out_graph_mat.cpu().detach().numpy()

    plt.figure(0, figsize=(10, 10))
    ax = plt.gca()

    count_0 = 0;
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if out_graph_mat[i][j] == 1:
                a = (cam_Cs[i, 0], cam_Cs[j, 0])
                b = (cam_Cs[i, 1], cam_Cs[j, 1])
                ax.plot(a, b, '-', color='blue', linewidth=0.5);
            elif out_graph_mat[i][j] == -1:
                a = (cam_Cs[i, 0], cam_Cs[j, 0])
                b = (cam_Cs[i, 1], cam_Cs[j, 1])
                ax.plot(a, b, '-', color='green', linewidth=0.1);
            else:
                count_0 = count_0 + 1
    ax.scatter(cam_Cs[:, 0], cam_Cs[:, 1], color='r', s=80)
    plt.show()


# %%
def imgs2batchsamesize(imgs):
    size_imgs = {}
    batch_imgs = []
    batch_idx = []
    for idx, img in enumerate(imgs):
        if img.shape not in size_imgs:
            size_imgs[img.shape] = []
        size_imgs[img.shape].append((idx, img))
    for img_list in size_imgs.values():
        if len(img_list) == 1:
            batch_imgs.append(img_list[0][1])
            batch_idx.append([img_list[0][0]])
        else:
            imgs_list = [x[1] for x in img_list]
            idx_list = [x[0] for x in img_list]
            batch_idx.append(idx_list)
            batch_imgs.append(torch.cat(imgs_list))
    return batch_idx, batch_imgs


def read_manual_data(file_path, num_camera=None):
    res = np.loadtxt(file_path)
    if num_camera is None:
        nC = int(np.max(res[:, 1])) + 1
    else:
        nC = num_camera
    inoutMat = np.zeros((nC, nC))
    for i in range(res.shape[0]):
        h = res[i]
        out = res[i][2]
        from_ = int(h[0])
        to_ = int(h[1])
        inoutMat[from_, to_] = int(out)
        inoutMat[to_, from_] = int(out)
    return inoutMat


# %%
class CambridgeDataset(Dataset):

    def __init__(self, iccv_res_dir, image_dir, dataset_list, lmdb_paths=None,
                 downsample_scale=0.25, sampling_num_range=[100, 500], sub_graph_nodes=24,
                 sample_res_cache=None,
                 transform_func='default'):
        # sampling_count: sampling the numbers of subgraph for a dataset

        self.num_dataset = len(dataset_list)
        self.iccv_res_dir = iccv_res_dir
        self.image_dir = image_dir
        self.sampling_num_range = sampling_num_range
        self.sub_graph_nodes = sub_graph_nodes
        self.transform_func = transform_func
        self.downsample_scale = downsample_scale
        if lmdb_paths is not None:
            self.use_lmdb = True
            self.lmdb_db = LMDBModel(lmdb_paths[0])
            self.lmdb_meta = pickle.load(open(lmdb_paths[1], 'rb'))
        else:
            self.use_lmdb = False

        if self.transform_func == 'default':
            self.transform_func = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])

        # read image list and calibration
        self.frame_list = {}
        self.K = {}
        self.dataset_names = []
        for ds in dataset_list:
            dataset_name = ds['name']
            self.dataset_names.append(dataset_name)

            frame_list = read_image_list(os.path.join(iccv_res_dir, dataset_name, 'ImageList.txt'))
            frame_list = [f[2:].split('.')[0].strip() for f in frame_list]

            self.frame_list[dataset_name] = frame_list

            K, img_dim = read_calibration(os.path.join(iccv_res_dir, dataset_name, 'calibration.txt'))
            self.K[dataset_name] = K

        self.Es = {}
        self.Cs = {}

        self.covis_map = {}
        self.edge_local_feat_cache = {}
        self.inout_mat = {}
        self.total_sample_num = 0

        max_scene_edges = 0         # determine the max edges for ratio sampling
        min_scene_edges = 1400000
        print('[1dsfm dataset Init] load in. and out. edges')
        for ds in tqdm(dataset_list):
            dataset_name = ds['name']
            Es, Cs = read_poses(os.path.join(iccv_res_dir, dataset_name, 'bundle.out'))
            self.Es[dataset_name] = Es
            self.Cs[dataset_name] = Cs

            inoutMat = np.load(os.path.join(iccv_res_dir, dataset_name, 'inoutMat.npy'))
            covis_map = np.load(os.path.join(iccv_res_dir, dataset_name, 'covis_map.npy'))

            with open(os.path.join(iccv_res_dir, dataset_name, 'edge_feat_pos_cache.bin'), 'rb') as f:
                edge_feat_pos_cache = pickle.load(f)
            self.edge_local_feat_cache[dataset_name] = edge_feat_pos_cache
            num_edges = len(edge_feat_pos_cache)
            if num_edges > max_scene_edges:
                max_scene_edges = num_edges
            if num_edges < min_scene_edges:
                min_scene_edges = num_edges

            self.inout_mat[dataset_name] = inoutMat
            self.covis_map[dataset_name] = covis_map
            self.edge_local_feat_cache[dataset_name] = edge_feat_pos_cache

        if min_scene_edges * 6 < max_scene_edges:
            # sampling ratio from the scene has most edges should be clamped.
            max_scene_edges = 6 * min_scene_edges

        """ Sampling ---------------------------------------------------------------------------------------------------
        """
        self.edge_sampler = {}
        self.samples = []                                                           # (dataset_id, sub-graph sample_id)

        if sample_res_cache is None or not os.path.exists(sample_res_cache):

            print('[1dsfm dataset Init] sampling sub_graphs')
            for ds_id, ds in enumerate(dataset_list):
                dataset_name = ds['name']

                n_Cameras = len(self.Cs[dataset_name])
                inoutMat = self.inout_mat[dataset_name]
                num_edges = len(self.edge_local_feat_cache[dataset_name])

                # determine sampling number based on ratio of edges among other scenes
                sample_ratio = num_edges / max_scene_edges
                sample_num = int(sampling_num_range[1] * sample_ratio)
                if sample_num < sampling_num_range[0]:
                    sample_num = sampling_num_range[0]
                if sample_num > sampling_num_range[1]:
                    sample_num = sampling_num_range[1]

                # todo: fix sub_graph_nodes
                gen = SamplingGenerator(n_Cameras, inoutMat)
                gen.setSamplingSize(sub_graph_nodes)
                gen.setSamplingNumber(sample_num)
                gen.generation()

                filtered_sampled_num = len(gen.sampling_node)
                print('[1dsfm dataset Init] %s: (filtered: %d, all: %d)' % (dataset_name, filtered_sampled_num, num_edges))

                self.samples += [(ds_id, i) for i in range(filtered_sampled_num)]
                self.edge_sampler[dataset_name] = (gen.sampling_node, gen.sampling_edge, gen.sampling_edge_label)

            if sample_res_cache is not None:
                with open(sample_res_cache, 'wb') as f:
                    pickle.dump([self.samples, self.edge_sampler], f)
                print('[1dsfm Init] Save subgraph fast cache to %s.' % sample_res_cache)

        elif os.path.exists(sample_res_cache):
            with open(sample_res_cache, 'rb') as f:
                s = pickle.load(f)
                self.samples, self.edge_sampler = s
            print('[1dsfm Init] Load subgraph fast cache from %s.' % sample_res_cache)

        print('[1dsfm Init] Done, %d samples' % len(self.samples))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        dataset_idx, sub_graph_id = self.samples[idx]

        dataset_name = self.dataset_names[dataset_idx]
        frame_list = self.frame_list[dataset_name]
        sampling_node, sampling_edge, sampling_edge_label = self.edge_sampler[dataset_name]
        edge_local_feat_cache = self.edge_local_feat_cache[dataset_name]

        subgraph_nodes = sampling_node[sub_graph_id]
        subgraph_edges = sampling_edge[sub_graph_id]
        subgraph_label = sampling_edge_label[sub_graph_id]
        sub_graph_nodes = len(subgraph_nodes)

        # todo: read image
        imgs = []
        img_ori_dim = []
        cam_Es, cam_Cs, cam_Ks = [], [], []
        img_id2sub_id = {}
        sub_id2img_id = {}
        # print(dataset_name)
        for i, imageID in enumerate(subgraph_nodes):

            # image ID
            if self.use_lmdb is True:
                img_key = dataset_name + '/' + frame_list[imageID] + '.jpg'
                img = self.lmdb_db.read_ndarray_by_key(img_key, dtype=np.uint8)
                h, w = self.lmdb_meta[img_key]['dim']
                img = img.reshape(int(h * 0.5), int(w * 0.5), 3)
            else:
                img_path = os.path.join(self.image_dir, dataset_name, 'images', frame_list[imageID] + '.jpg')
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
            img_ori_dim.append((h, w))

            if h > 1200 or w > 1200:
                img = cv2.resize(img, dsize=(int(img.shape[1] * 0.6),
                                             int(img.shape[0] * 0.6)))

            img = cv2.resize(img, dsize=(int(w * self.downsample_scale), int(h * self.downsample_scale)))
            img = img.astype(np.float32) / 255.0

            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            if self.transform_func is not None:
                img = self.transform_func(img)

            imgs.append(img)

            camera_C = self.Cs[dataset_name][imageID]
            cam_Cs.append(torch.from_numpy(camera_C).float())
            camera_E = self.Es[dataset_name][imageID]
            cam_Es.append(torch.from_numpy(camera_E).float())
            camera_K = self.K[dataset_name][imageID]
            cam_Ks.append(torch.from_numpy(camera_K).float())

            img_id2sub_id[imageID] = i
            sub_id2img_id[i] = imageID

        cam_Cs = torch.stack(cam_Cs, dim=0)
        cam_Es = torch.stack(cam_Es, dim=0)
        cam_Ks = torch.stack(cam_Ks, dim=0)

        # todo: read edge to adjacent matrix
        out_graph_mat = np.zeros((sub_graph_nodes, sub_graph_nodes), dtype=np.float32)
        out_covis_mat = np.zeros((sub_graph_nodes, sub_graph_nodes), dtype=np.float32)

        edge_local_matches_n1 = []
        edge_local_matches_n2 = []
        edge_subnode_idx = []
        edge_type = torch.zeros(len(subgraph_edges), dtype=torch.long)

        for i, edge in enumerate(subgraph_edges):
            reconnect_idx = (img_id2sub_id[edge[0]], img_id2sub_id[edge[1]])  # remap index to subgraph
            edge_subnode_idx.append(reconnect_idx)

            label = subgraph_label[i]
            covis_value = self.covis_map[dataset_name][edge[0], edge[1]]
            if covis_value == 0:
                covis_value = self.covis_map[dataset_name][edge[1], edge[0]]
            out_graph_mat[reconnect_idx[0], reconnect_idx[1]] = label
            out_graph_mat[reconnect_idx[1], reconnect_idx[0]] = label
            out_covis_mat[reconnect_idx[0], reconnect_idx[1]] = covis_value
            out_covis_mat[reconnect_idx[1], reconnect_idx[0]] = covis_value

            n1 = edge[0]
            n2 = edge[1]
            if '%d-%d' % (n1, n2) in edge_local_feat_cache:
                edge_cache = edge_local_feat_cache['%d-%d' % (n1, n2)]
                pts1 = torch.from_numpy(edge_cache['n1_feat_pos'])
                pts2 = torch.from_numpy(edge_cache['n2_feat_pos'])
                edge_type[i] = 1 if edge_cache['type'] == 'I' else 0
            elif '%d-%d' % (n2, n1) in edge_local_feat_cache:
                edge_cache = edge_local_feat_cache['%d-%d' % (n2, n1)]
                pts1 = torch.from_numpy(edge_cache['n2_feat_pos'])
                pts2 = torch.from_numpy(edge_cache['n1_feat_pos'])
                edge_type[i] = 1 if edge_cache['type'] == 'I' else 0

            edge_local_matches_n1.append(pts1)
            edge_local_matches_n2.append(pts2)

        out_graph_mat = torch.from_numpy(out_graph_mat)
        out_covis_mat = torch.from_numpy(out_covis_mat)

        return imgs, img_ori_dim, cam_Es, cam_Cs, cam_Ks, out_graph_mat, img_id2sub_id, sub_id2img_id, out_covis_mat, edge_subnode_idx, edge_type, edge_local_matches_n1, edge_local_matches_n2


if __name__ == '__main__':

    iccv_res_dir = '/mnt/Tango/pg/ICCV15_raw/'
    image_dir = '/mnt/Exp_2/1dsfm/imgs/'
    datalist = [
        {'name': 'Alamo'},
        {'name': 'Ellis_Island'},
        {'name': 'Gendarmenmarkt'},
        {'name': 'Madrid_Metropolis'},
        # {'name': 'Montreal_Notre_Dame'},
        # {'name': 'NYC_Library'},
    #     {'name': 'Piazza_del_Popolo'},
    #     {'name': 'Piccadilly'},
    #     {'name': 'Roman_Forum'},
    #     {'name': 'Tower_of_London'},
    # # {'name': 'Trafalgar'},
    #     {'name': 'Union_Square'},
    #     {'name': 'Vienna_Cathedral'},
    #     {'name': 'Yorkminster'},
    ]

    dataset = CambridgeDataset(iccv_res_dir=iccv_res_dir, image_dir=image_dir, dataset_list=datalist, sample_res_cache='/mnt/Tango/pg/ICCV15_raw/subgraph_cache.bin')

    # imgs, cam_Es, cam_Cs, out_graph_mat, img_id2sub_id, sub_id2img_id, out_covis_mat, edge_subnode_idx, edge_type, edge_local_matches_n1, edge_local_matches_n2 = dataset[80]
    # print(cam_Cs)
    # print(len(edge_local_matches_n1))