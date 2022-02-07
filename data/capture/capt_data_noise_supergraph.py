from data.capture.capture_dataset_utils import *
from graph_utils.subgraph_sampler import SubgraphGenerator
import shutil
from data.rotmap import w2R
# preset for different type of dataset
data_meta_dict = {
    'bundle': {
        'calib': 'calibration.txt',
        'frame_list': 'bundle.list.txt',
        'recon': 'bundle'
    },
    'yfcc': {
        'calib': 'recaled_calibration.txt',
        'frame_list': 'raw_images.list.txt',
        'recon': 'bundle.out',
    },
    '1dsfm': {
        'calib': 'calibration.txt',
        'frame_list': 'ImageList.txt',
        'recon': 'gt_bundle.out',
    }
}

def filtering_edge_cache(edge_cache: dict, num_nodes, verbose=False):
    filtered_edge_cache = dict()
    filtered_inout_mat = np.zeros((num_nodes, num_nodes))

    for key, val in edge_cache.items():
        Rt = val['Rt']
        if np.abs(Rt[0, 0]) < 1e-4 and np.abs(Rt[1, 1]) < 1e-4 and np.abs(Rt[2, 2]) < 1e-4:
            continue

        if np.isnan(Rt).sum() > 1 or np.isinf(Rt).sum() > 1:
            continue

        filtered_edge_cache[key] = val

        tokens = key.split('-')
        n1, n2 = int(tokens[0]), int(tokens[1])
        filtered_inout_mat[n1, n2] = 1
        filtered_inout_mat[n2, n1] = 1

    if verbose:
        print('[Filtered] %d / %d' % (len(filtered_edge_cache), len(edge_cache)))
    return filtered_edge_cache, filtered_inout_mat


class SupergraphDataset(Dataset):

    def __init__(self, dataset_dir, dataset_list, img_lmdb_paths=None,
                 node_edge_lmdb_path=None,
                 img_max_dim=480,
                 sub_graph_nodes=24,
                 sample_res_cache=None,
                 load_img=True,
                 load_keypt_match=False,
                 load_node_edge_feat=False,
                 transform_func='default',
                 subgraph_edge_cover_ratio=0.4,
                 noise=10,
                 error=0.05):

        self.num_dataset = len(dataset_list)
        self.dataset_dir = dataset_dir
        self.sub_graph_nodes = sub_graph_nodes
        self.img_max_dim = img_max_dim
        self.load_img = load_img
        self.load_keypt_match = load_keypt_match
        self.load_node_edge_feat = load_node_edge_feat
        self.use_lmdb = False
        self.noise = noise
        self.error = error

        if self.load_node_edge_feat is True:
            if node_edge_lmdb_path is None:
                raise Exception('Can not found node edge feature lmdb: %s' % node_edge_lmdb_path)
            self.node_feat_lmdb = LMDBModel(node_edge_lmdb_path)
            node_edge_meta_path = node_edge_lmdb_path.split('.lmdb')[0] + '.bin'
            with open(node_edge_meta_path, 'rb') as f:
                self.edge_feat_dict, self.node_feat_meta_dict = pickle.load(f)
                f.close()

        if self.load_img is True and img_lmdb_paths is not None:
            self.use_lmdb = True
            self.lmdb_db = LMDBModel(img_lmdb_paths[0])
            self.lmdb_meta = pickle.load(open(img_lmdb_paths[1], 'rb'))

        self.transform_func = transform_func
        if self.transform_func == 'default':
            self.transform_func = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                           std=[0.229, 0.224, 0.225])])

        # read image list and calibration ------------------------------------------------------------------------------
        self.frame_list = {}
        self.K = {}
        self.dataset_names = []
        for ds in dataset_list:
            dataset_name = ds['name']
            self.dataset_names.append(dataset_name)

            # frame-list
            bundle_prefix = ds['bundle_prefix']
            frame_list = read_image_list(
                os.path.join(dataset_dir, dataset_name, data_meta_dict[bundle_prefix]['frame_list']))
            self.frame_list[dataset_name] = frame_list

            # camera calibration
            K, img_dim = read_calibration(
                os.path.join(dataset_dir, dataset_name, data_meta_dict[bundle_prefix]['calib']))
            self.K[dataset_name] = K

        # read camera extrinsic ----------------------------------------------------------------------------------------
        self.Es = {}

        self.inout_mat = {}
        self.covis_map = {}
        self.edge_local_feat_cache = {}
        self.total_sample_num = 0

        print('[SuperGraph dataset Init] load in. and out. edges')

        for ds in tqdm(dataset_list):
            dataset_name = ds['name']
            bundle_prefix = ds['bundle_prefix']

            # read camera poses
            Es, _ = read_poses(os.path.join(dataset_dir, dataset_name, data_meta_dict[bundle_prefix]['recon']))
            self.Es[dataset_name] = Es

            # graph connectivity and inlier/outlier marker
            inoutMat = np.load(os.path.join(dataset_dir, dataset_name, 'inoutMat.npy'))
            covis_map = np.load(os.path.join(dataset_dir, dataset_name, 'covis.npy'))
            self.inout_mat[dataset_name] = inoutMat
            self.covis_map[dataset_name] = covis_map

            # read edge cache (Rt)
            if self.load_keypt_match:
                with open(os.path.join(dataset_dir, dataset_name, 'edge_feat_pos_cache.bin'), 'rb') as f:
                    edge_feat_pos_cache = pickle.load(f)
                    f.close()
            else:
                with open(os.path.join(dataset_dir, dataset_name, 'edge_rt_cache.bin'), 'rb') as f:
                    edge_feat_pos_cache = pickle.load(f)
                    print('Load from %s', os.path.join(dataset_dir, dataset_name, 'edge_rt_cache.bin'))
                    f.close()
            edge_feat_pos_cache, inoutMat = filtering_edge_cache(edge_feat_pos_cache, inoutMat.shape[0])
            self.edge_local_feat_cache[dataset_name] = edge_feat_pos_cache
            self.inout_mat[dataset_name] = inoutMat
            self.covis_map[dataset_name] = inoutMat

            print('[Supergraph dataset Init] ds: %s, nodes: %d, edges: %d' % (dataset_name, covis_map.shape[0], len(edge_feat_pos_cache)))

        """ Sampling ---------------------------------------------------------------------------------------------------
        """
        self.subgraphs = {}
        self.samples = []  # (dataset_id, sub-graph sample_id)

        if sample_res_cache is None or not os.path.exists(sample_res_cache):

            print('[SuperGraph dataset Init] sampling sub_graphs')
            for ds_id, ds in enumerate(dataset_list):
                dataset_name = ds['name']
                # edge_feat_pos_cache = self.edge_local_feat_cache[dataset_name]

                n_Cameras = len(self.Es[dataset_name])
                inoutMat = self.inout_mat[dataset_name]
                num_edges = len(self.edge_local_feat_cache[dataset_name])
                edge_cache = self.edge_local_feat_cache[dataset_name]

                # generate edge list
                edge_list = []
                for e in edge_cache.keys():
                    e_split = e.split('-')
                    n1, n2 = int(e_split[0]), int(e_split[1])
                    edge_list.append((n1, n2))

                # todo: do we need function generationCoverNode?
                subgraph_gen = SubgraphGenerator.fromEdgelist(edge_list,
                                                              maxNodeNum=self.sub_graph_nodes,
                                                             minNodeNum=0.9*self.sub_graph_nodes,
                                                             maxCoverRatio=0.5)
                # subgraphs = subgraph_gen.generationCoverNodeByRatio(subgraph_node_cover_ratio)
                subgraphs = subgraph_gen.generationCoverEdge(subgraph_edge_cover_ratio)
                print('[SuperGraph dataset Init] %s: (total subgraphs: %d)' % (dataset_name, len(subgraphs)))

                self.samples += [(ds_id, i) for i in range(len(subgraphs))]
                self.subgraphs[dataset_name] = subgraphs

            if sample_res_cache is not None:
                with open(sample_res_cache, 'wb') as f:
                    pickle.dump([self.samples, self.subgraphs], f)
                    f.close()
                print('[SuperGraph Init] Save subgraph fast cache to %s.' %
                      sample_res_cache)

        elif os.path.exists(sample_res_cache):
            with open(sample_res_cache, 'rb') as f:
                s = pickle.load(f)
                self.samples, self.subgraphs = s
                f.close()

            print('[SuperGraph Init] Load subgraph fast cache from %s.' %
                  sample_res_cache)

        # remap to valid node id
        self.id2valid_id = dict()
        for ds_id, ds in enumerate(dataset_list):
            dataset_name = ds['name']
            edge_cache = self.edge_local_feat_cache[dataset_name]

            id2valid_id = dict()
            for e in edge_cache.keys():
                e_split = e.split('-')
                n1, n2 = int(e_split[0]), int(e_split[1])

                if n1 not in id2valid_id:
                    id2valid_id[n1] = len(id2valid_id)
                if n2 not in id2valid_id:
                    id2valid_id[n2] = len(id2valid_id)

            self.id2valid_id[dataset_name] = id2valid_id

        print('[SuperGraph Init] Done, %d samples' % len(self.samples))

        # random.shuffle(self.samples)
        self.noise_list = dict()
        sigma = self.noise * math.pi /180 / math.sqrt(3)
        for dataset in list(self.edge_local_feat_cache.keys()):
            edge_local_feat_cache_now = self.edge_local_feat_cache[dataset]
            edge_local_feat_cache_new = dict()
            Es = self.Es[dataset]
            key_list = list(edge_local_feat_cache_now.keys())
            self.noise_list[dataset] = random.sample(key_list, int(len(key_list)*self.error))
            for key, val in edge_local_feat_cache_now.items():
                n1 = int(key.split("-")[0])
                n2 = int(key.split("-")[1])
                E1 = Es[n1][:,:3]
                E2 = Es[n2][:,:3]
                new_rel_R = np.matmul(E1,E2.T)
                w = torch.from_numpy(np.random.normal(0,1,[3]))
                w = w / torch.norm(w)*sigma*torch.from_numpy(np.random.normal(0,1,[1])) / 10
                y = 2 * (random.random() - 0.5)
                w2 =  torch.from_numpy(np.asarray([y, math.sqrt(1 - y*y), 0])*sigma*random.gauss(0,1))
                new_rel_R = torch.from_numpy(new_rel_R).type(torch.float64)
                new_rel_R = torch.mm(new_rel_R, w2R(w2))
                new_rel_R = torch.mm(new_rel_R, w2R(w))

                if key in self.noise_list[dataset]:
                    new_rel_R = w2R(torch.from_numpy((random.gauss(0,1)*90*pi/180/sqrt(3))*np.random.normal(0,1,[3])))
                val["Rt"][:,:3] = new_rel_R.numpy()
                edge_local_feat_cache_new[key] = val
            self.edge_local_feat_cache[dataset] = edge_local_feat_cache_new
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # todo: update the subgraph_nodes, subgraph_edges with new structure

        dataset_idx, sub_graph_id = self.samples[idx]

        dataset_name = self.dataset_names[dataset_idx]
        frame_list = self.frame_list[dataset_name]
        # sampling_node, sampling_edge, sampling_edge_label = self.subgraphs[dataset_name]
        subgraphs = self.subgraphs[dataset_name]
        edge_local_feat_cache = self.edge_local_feat_cache[dataset_name]
        inoutMat = self.inout_mat[dataset_name]
        id2valid_id = self.id2valid_id[dataset_name]

        # if len(self.noise_list) == 0:
        #     key_list = list(edge_local_feat_cache.keys())
        #     self.noise_list = random.sample(key_list, int(len(key_list) * self.noise))
        #     for key in self.noise_list:
        #         Rt = edge_local_feat_cache[key]['Rt']
        #         R = Rt[:,:3]
        #         R = torch.mm(torch.from_numpy(R).type(torch.float64), w2R(torch.rand(3)))
        #         Rt[:,:3] = R.numpy()
        #         edge_local_feat_cache[key]['Rt'] = Rt


        subgraph_edges = subgraphs[sub_graph_id]
        subgraph_nodes = []
        subgraph_label = []
        for edge in subgraph_edges:
            if edge[0] not in subgraph_nodes:
                subgraph_nodes.append(edge[0])
            if edge[1] not in subgraph_nodes:
                subgraph_nodes.append(edge[1])
            subgraph_label.append(inoutMat[edge[0], edge[1]])
        sub_graph_nodes = len(subgraph_nodes)

        """ Node -------------------------------------------------------------------------------------------------------
        """
        imgs = []
        img_names = []
        img_ori_dim = []
        cam_Es, cam_Ks = [], []
        img_id2sub_id = {}
        sub_id2img_id = {}
        valid_id2sub_id = {}
        sub_id2valid_id = {}
        node_feats = []

        # print(dataset_name)
        for i, imageID in enumerate(subgraph_nodes):

            # load image by image ID
            if dataset_name.split("_")[0] == "Trafalgar":
                dataset_name_pre = dataset_name.split("_")[0]
            else:
                dataset_name_pre = dataset_name
            img_key = dataset_name_pre+ '/' + frame_list[imageID]
            if self.load_img is True:
                if self.use_lmdb is True:
                    # load image from lmdb
                    img = self.lmdb_db.read_ndarray_by_key(img_key, dtype=np.uint8)
                    h, w = self.lmdb_meta[img_key]['dim']
                    res_h, res_w = self.lmdb_meta[img_key]['lmdb_dim']
                    img = img.reshape(int(res_h), int(res_w), 3)
                else:
                    # load image from image file
                    img_path = os.path.join(
                        self.dataset_dir, dataset_name, frame_list[imageID] + '.jpg')
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = img.shape[:2]

                # resize the image
                res_h, res_w = img.shape[:2]
                min_dim = res_h if res_h < res_w else res_w
                down_factor = float(self.img_max_dim) / float(min_dim)
                img = cv2.resize(img, dsize=(
                    int(res_w * down_factor), int(res_h * down_factor)))
                img = img.astype(np.float32) / 255.0

                img = torch.from_numpy(img)
                img = img.permute(2, 0, 1)
                if self.transform_func is not None:
                    img = self.transform_func(img)

                imgs.append(img)
            else:
                res_h = h = 0
                res_w = w = 0
                imgs.append(torch.ones(1))

            # load image feature (node feat)
            if self.load_node_edge_feat:
                key = '%s,%d' % (dataset_name, imageID)
                if key not in self.node_feat_meta_dict:
                    raise Exception('Node feat %s not found' % key)
                node_feat = self.node_feat_lmdb.read_ndarray_by_key(key).ravel()
                node_feats.append(torch.from_numpy(node_feat).unsqueeze(0))
            else:
                node_feat = torch.zeros((1, 1))
                node_feats.append(node_feat)

            # load remaining information (E, K)
            img_ori_dim.append((h, w))
            img_names.append(img_key)

            camera_E = self.Es[dataset_name][imageID]
            cam_Es.append(torch.from_numpy(camera_E).float())
            camera_K = self.K[dataset_name][imageID]
            cam_Ks.append(torch.from_numpy(camera_K).float())

            # create converter dict
            img_id2sub_id[imageID] = i
            sub_id2img_id[i] = imageID

            valid_id = id2valid_id[imageID]
            valid_id2sub_id[valid_id] = i
            sub_id2valid_id[i] = valid_id

        cam_Es = torch.stack(cam_Es, dim=0)
        cam_Ks = torch.stack(cam_Ks, dim=0)
        node_feats = torch.cat(node_feats, dim=0)

        """ Edges ------------------------------------------------------------------------------------------------------
        """
        out_graph_mat = np.zeros((sub_graph_nodes, sub_graph_nodes), dtype=np.float32)
        out_covis_mat = np.zeros((sub_graph_nodes, sub_graph_nodes), dtype=np.float32)

        edge_local_matches_n1 = []
        edge_local_matches_n2 = []
        edge_subnode_idx = []
        edge_type = torch.zeros(len(subgraph_edges), dtype=torch.long)
        edge_rel_Rt = []
        edge_rel_err = []
        edge_feats = []


        for i, edge in enumerate(subgraph_edges):
            # remap index to subgraph
            reconnect_idx = (img_id2sub_id[edge[0]], img_id2sub_id[edge[1]])
            edge_subnode_idx.append(reconnect_idx)

            label = subgraph_label[i]
            covis_value = self.covis_map[dataset_name][edge[0], edge[1]]
            if covis_value == 0:
                covis_value = self.covis_map[dataset_name][edge[1], edge[0]]

            out_graph_mat[reconnect_idx[0], reconnect_idx[1]] = label
            out_graph_mat[reconnect_idx[1], reconnect_idx[0]] = label
            out_covis_mat[reconnect_idx[0], reconnect_idx[1]] = covis_value
            out_covis_mat[reconnect_idx[1], reconnect_idx[0]] = covis_value

            n1, n2 = edge[0], edge[1]

            if '%d-%d' % (n1, n2) in edge_local_feat_cache:
                edge_cache = edge_local_feat_cache['%d-%d' % (n1, n2)]

                if self.load_keypt_match:
                    pts1 = torch.from_numpy(edge_cache['n1_feat_pos'])
                    pts2 = torch.from_numpy(edge_cache['n2_feat_pos'])
                # edge_type[i] = 1 if edge_cache['type'] == 'I' else 0

                Rt = edge_cache['Rt'].astype(np.float32)
                Rt_inv = cam_opt.camera_pose_inv(Rt[:3, :3], Rt[:3, 3])
                edge_rel_Rt.append(torch.from_numpy(Rt_inv))

                rel_err = edge_cache['rel_err']
                edge_type[i] = 1 if rel_err < 20 else 0
                edge_rel_err.append(rel_err)

            elif '%d-%d' % (n2, n1) in edge_local_feat_cache:
                edge_cache = edge_local_feat_cache['%d-%d' % (n2, n1)]
                if self.load_keypt_match:
                    pts1 = torch.from_numpy(edge_cache['n2_feat_pos'])
                    pts2 = torch.from_numpy(edge_cache['n1_feat_pos'])
                # edge_type[i] = 1 if edge_cache['type'] == 'I' else 0
                Rt_n2ton1 = edge_cache['Rt'].astype(np.float32)
                edge_rel_Rt.append(torch.from_numpy(Rt_n2ton1))

                rel_err = edge_cache['rel_err']
                edge_type[i] = 1 if rel_err < 20 else 0
                edge_rel_err.append(rel_err)

            else:
                raise Exception("edge not found %s (%d, %d)" % (dataset_name, n1, n2))

            if self.load_keypt_match:
                edge_local_matches_n1.append(pts1)
                edge_local_matches_n2.append(pts2)
            else:
                edge_local_matches_n1 = torch.zeros(1)
                edge_local_matches_n2 = torch.zeros(2)

            # load edge feat
            if self.load_node_edge_feat:
                key1 = '%s,%d-%d' % (dataset_name, n1, n2)
                key2 = '%s,%d-%d' % (dataset_name, n2, n1)
                if key1 in self.edge_feat_dict:
                    edge_feat = self.node_feat_lmdb.read_ndarray_by_key(key1)
                    edge_feats.append(torch.from_numpy(edge_feat.ravel()).unsqueeze(0))
                elif key2 in self.edge_feat_dict:
                    edge_feat = self.node_feat_lmdb.read_ndarray_by_key(key2)
                    edge_feats.append(torch.from_numpy(edge_feat.ravel()).unsqueeze(0))
                else:
                    raise Exception('Edge feature not found on %s' % key1)
            else:
                edge_feats.append(torch.zeros((1, 1)))

        edge_rel_err = torch.from_numpy(np.asarray(edge_rel_err).astype(np.float32))
        out_graph_mat = torch.from_numpy(out_graph_mat)
        out_covis_mat = torch.from_numpy(out_covis_mat)
        edge_feats = torch.cat(edge_feats, dim=0)

        if len(edge_subnode_idx) != len(edge_rel_Rt):
            raise Exception("Error")

        return dataset_name, idx, img_names, imgs, img_ori_dim, cam_Es, cam_Ks, out_graph_mat, valid_id2sub_id, sub_id2valid_id, img_id2sub_id, sub_id2img_id, out_covis_mat, edge_subnode_idx, edge_type, edge_local_matches_n1, edge_local_matches_n2, edge_rel_Rt, edge_rel_err, node_feats, edge_feats


            

            