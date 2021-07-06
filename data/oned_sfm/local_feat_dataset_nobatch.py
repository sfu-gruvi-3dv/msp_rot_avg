# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os, sys, cv2, pickle
import numpy as np
import tqdm
import torch

from torch.utils.data import Dataset
from core_io.ply_io import load_pointcloud_from_ply
from data.ambi.read_helper import *
from data.ambi.ambi_parser import *
from evaluator.basic_metric import rel_distance, rel_rot_angle
from sampling import *
from tqdm import tqdm
from data.oned_sfm.SceneModel import ODSceneModel


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

# %%
datalist = [
    {'name': 'Roman_Forum'},
    # {'name': 'Trafalgar'},
    # {'name': 'Union_Square'},
]


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
class OneDSFMDataset(Dataset):

    def __init__(self, iccv_res_dir, image_dir, dataset_list, downsample_scale=0.25, sampling_num=100, sub_graph_nodes=24,
                 transform_func=None):
        # sampling_count: sampling the numbers of subgraph for a dataset

        self.num_dataset = len(dataset_list)
        self.iccv_res_dir = iccv_res_dir
        self.sampling_num = sampling_num
        self.image_dir = image_dir
        self.sub_graph_nodes = sub_graph_nodes
        self.transform_func = transform_func
        self.downsample_scale = downsample_scale

        # read image list and calibration
        self.frame_list = {}
        self.K = {}
        self.dataset_names = []
        for ds in dataset_list:
            dataset_name = ds['name']
            self.dataset_names.append(dataset_name)

            frame_list = read_image_list(os.path.join(iccv_res_dir, dataset_name, 'ImageList.txt'))
            frame_list = [os.path.join('images', f[2:].split('.')[0].strip()) for f in frame_list]

            self.frame_list[dataset_name] = frame_list

            K, img_dim = read_calibration(os.path.join(iccv_res_dir, dataset_name, 'calibration.txt'))
            self.K[dataset_name] = K

        self.Es = {}
        self.Cs = {}

        self.edge_sampler = {}
        self.covis_map = {}
        self.edge_local_feat_cache = {}

        print('[1dsfm dataset Init] load in. and out. edges and sampling sub_graphs')
        for ds in tqdm(dataset_list):
            dataset_name = ds['name']
            # eg_file_path = os.path.join(image_dir, dataset_name, 'EGs.txt')
            # bundle_file_name = os.path.join(image_dir, dataset_name, ds['bundle_file'])

            Es, Cs = read_poses(os.path.join(iccv_res_dir, dataset_name, 'bundle.out'))
            self.Es[dataset_name] = Es
            self.Cs[dataset_name] = Cs

            n_Cameras = len(Cs)
            inoutMat = np.load(os.path.join(iccv_res_dir, dataset_name, 'inoutMat.npy'))
            covis_map = np.load(os.path.join(iccv_res_dir, dataset_name, 'covis_map.npy'))

            with open(os.path.join(iccv_res_dir, dataset_name, 'edge_feat_pos_cache.bin'), 'rb') as f:
                edge_feat_pos_cache = pickle.load(f)

            # random generate a sub graph
            # todo: fix sub_graph_nodes
            gen = SamplingGenerator(n_Cameras, inoutMat)
            gen.setSamplingSize(sub_graph_nodes)
            gen.setSamplingNumber(sampling_num)
            gen.generation()
            self.edge_sampler[dataset_name] = gen
            self.covis_map[dataset_name] = covis_map
            self.edge_local_feat_cache[dataset_name] = edge_feat_pos_cache

        print('[1dsfm Init] Done')

    def __len__(self):
        return self.num_dataset * self.sampling_num

    def __getitem__(self, idx):

        dataset_idx = idx / self.sampling_num
        sub_graph_id = idx % self.sampling_num  # todo, need a new random idx
        dataset_idx = int(dataset_idx)
        sub_graph_id = int(sub_graph_id)

        dataset_name = self.dataset_names[dataset_idx]
        frame_list = self.frame_list[dataset_name]
        sampled_subgraphs = self.edge_sampler[dataset_name]
        edge_local_feat_cache = self.edge_local_feat_cache[dataset_name]

        subgraph_nodes = sampled_subgraphs.sampling_node[sub_graph_id]
        subgraph_edges = sampled_subgraphs.sampling_edge[sub_graph_id]
        subgraph_label = sampled_subgraphs.sampling_edge_label[sub_graph_id]
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
            img_path = os.path.join(self.image_dir, dataset_name, frame_list[imageID] + '.jpg')
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            img_ori_dim.append((h, w))
            img = cv2.resize(img, dsize=(int(w * self.downsample_scale), int(h * self.downsample_scale)))
            # h, w = img.shape[:2]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    dataset = OneDSFMDataset(iccv_res_dir=iccv_res_dir, image_dir=image_dir, dataset_list=datalist)

    imgs, cam_Es, cam_Cs, out_graph_mat, img_id2sub_id, sub_id2img_id, out_covis_mat, edge_subnode_idx, edge_type, edge_local_matches_n1, edge_local_matches_n2 = dataset[80]
    print(cam_Cs)
    print(len(edge_local_matches_n1))