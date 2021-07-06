from data.capture.capture_dataset_utils import *

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

class CaptureDataset(Dataset):

    def __init__(self, dataset_dir, dataset_list, img_lmdb_paths=None,
                 node_edge_lmdb_path=None,
                 img_max_dim=480,
                 sampling_num_range=[100, 500],
                 sub_graph_nodes=24,
                 sample_res_cache=None,
                 sampling_undefined_edge=False,
                 load_img=True,
                 load_keypt_match=False,
                 load_node_edge_feat=False,
                 transform_func='default'):

        self.num_dataset = len(dataset_list)
        self.dataset_dir = dataset_dir
        self.sampling_num_range = sampling_num_range
        self.sub_graph_nodes = sub_graph_nodes
        self.img_max_dim = img_max_dim
        self.sampling_undefined_edge = sampling_undefined_edge
        self.load_img = load_img
        self.load_keypt_match = load_keypt_match
        self.load_node_edge_feat = load_node_edge_feat
        self.use_lmdb = False

        if self.load_node_edge_feat is True:
            if node_edge_lmdb_path is  None:
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
            frame_list = read_image_list(os.path.join(dataset_dir, dataset_name, data_meta_dict[bundle_prefix]['frame_list']))
            self.frame_list[dataset_name] = frame_list

            # camera calibration
            K, img_dim = read_calibration(os.path.join(dataset_dir, dataset_name, data_meta_dict[bundle_prefix]['calib']))
            self.K[dataset_name] = K

        # read camera extrinsic ----------------------------------------------------------------------------------------
        self.Es = {}

        self.inout_mat = {}
        self.covis_map = {}
        self.edge_local_feat_cache = {}
        self.total_sample_num = 0

        max_scene_edges = 0  # determine the max edges for ratio sampling
        min_scene_edges = 1400000
        print('[Captured dataset Init] load in. and out. edges')
        z_flip = np.diag([1, 1, -1])
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
                    f.close()
            self.edge_local_feat_cache[dataset_name] = edge_feat_pos_cache

            num_edges = len(edge_feat_pos_cache)
            if num_edges > max_scene_edges:
                max_scene_edges = num_edges
            if num_edges < min_scene_edges:
                min_scene_edges = num_edges

        if min_scene_edges * 30 < max_scene_edges:
            # sampling ratio from the scene has most edges should be clamped.
            max_scene_edges = 30 * min_scene_edges

        """ Sampling ---------------------------------------------------------------------------------------------------
        """
        self.edge_sampler = {}
        self.samples = []                                       # (dataset_id, sub-graph sample_id)

        if sample_res_cache is None or not os.path.exists(sample_res_cache):

            print('[Captured dataset Init] sampling sub_graphs')
            for ds_id, ds in enumerate(dataset_list):
                dataset_name = ds['name']
                edge_feat_pos_cache = self.edge_local_feat_cache[dataset_name]

                n_Cameras = len(self.Es[dataset_name])
                inoutMat = self.inout_mat[dataset_name]
                # edge_feat_pos_cache = self.edge_local_feat_cache[dataset_name]
                for i in range(n_Cameras):
                    for j in range(n_Cameras):
                        if (("%d-%d" % (i, j)) in edge_feat_pos_cache or
                            ("%d-%d" % (j, i)) in edge_feat_pos_cache):
                            inoutMat[i, j] = 1
                num_edges = len(self.edge_local_feat_cache[dataset_name])

                # determine sampling number based on ratio of edges among other scenes
                sample_ratio = float(num_edges) / float(max_scene_edges)
                print('%s: Sampling Ratio: %.2f' % (dataset_name, sample_ratio))
                sample_num = int(sampling_num_range[1] * sample_ratio)
                if sample_num < sampling_num_range[0]:
                    sample_num = sampling_num_range[0]
                if sample_num > sampling_num_range[1]:
                    sample_num = sampling_num_range[1]

                # sampling
                gen = SamplingGenerator(n_Cameras, inoutMat)
                gen.setSamplingSize(sub_graph_nodes)
                gen.setSamplingNumber(sample_num)
                gen.generation(
                    use_undefine=self.sampling_undefined_edge, get_max_node=False)

                print("test inoutmat")
                for edges in gen.sampling_edge:
                    flag = False
                    for edge in edges:

                        if ("%d-%d" % (edge[0], edge[1]) in edge_feat_pos_cache) or ("%d-%d" % (edge[1], edge[0]) in edge_feat_pos_cache):
                            continue
                        else:
                            flag = True
                    if flag:
                        print("bad")

                filtered_sampled_num = len(gen.sampling_node)
                print('[Captured dataset Init] %s: (filtered: %d, all: %d)' % (
                    dataset_name, filtered_sampled_num, num_edges))

                self.samples += [(ds_id, i)
                                 for i in range(filtered_sampled_num)]
                self.edge_sampler[dataset_name] = (
                    gen.sampling_node, gen.sampling_edge, gen.sampling_edge_label)

            if sample_res_cache is not None:
                with open(sample_res_cache, 'wb') as f:
                    pickle.dump([self.samples, self.edge_sampler], f)
                    f.close()
                print('[Captured Init] Save subgraph fast cache to %s.' %
                      sample_res_cache)

        elif os.path.exists(sample_res_cache):
            with open(sample_res_cache, 'rb') as f:
                s = pickle.load(f)
                self.samples, self.edge_sampler = s
                f.close()

            print('[Captured Init] Load subgraph fast cache from %s.' %
                  sample_res_cache)

        print('[Captured Init] Done, %d samples' % len(self.samples))
        print('[Captured Init] Rt_rel_12: n2 to n1')
        # random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def get_linkage(self, idx):
        dataset_idx, sub_graph_id = self.samples[idx]

        dataset_name = self.dataset_names[dataset_idx]
        frame_list = self.frame_list[dataset_name]
        sampling_node, sampling_edge, sampling_edge_label = self.edge_sampler[dataset_name]
        # edge_local_feat_cache = self.edge_local_feat_cache[dataset_name]

        subgraph_nodes = sampling_node[sub_graph_id]
        subgraph_edges = sampling_edge[sub_graph_id]
        subgraph_label = sampling_edge_label[sub_graph_id]
        sub_graph_nodes = len(subgraph_nodes)

        """ Load image -------------------------------------------------------------------------------------------------
        """
        img_ids = []
        img_names = []
        img_id2sub_id = {}
        sub_id2img_id = {}

        # print(dataset_name)
        for i, imageID in enumerate(subgraph_nodes):
            # load image by image ID
            img_key = dataset_name + '/' + frame_list[imageID]
            img_ids.append(imageID)
            img_names.append(img_key)
            img_id2sub_id[imageID] = i
            sub_id2img_id[i] = imageID

        edge_subnode_idx = []
        edge_ori_idx = []

        for i, edge in enumerate(subgraph_edges):
            reconnect_idx = (img_id2sub_id[edge[0]], img_id2sub_id[edge[1]])
            edge_subnode_idx.append(reconnect_idx)

            n1, n2 = edge[0], edge[1]
            edge_ori_idx.append((n1, n2))

        return dataset_name, img_ids, img_names, edge_ori_idx, edge_subnode_idx, img_id2sub_id, sub_id2img_id

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

        """ Node -------------------------------------------------------------------------------------------------------
        """
        imgs = []
        img_names = []
        img_ori_dim = []
        cam_Es, cam_Ks = [], []
        img_id2sub_id = {}
        sub_id2img_id = {}
        node_feats = []

        # print(dataset_name)
        for i, imageID in enumerate(subgraph_nodes):

            # load image by image ID
            img_key = dataset_name + '/' + frame_list[imageID]
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
                edge_type[i] = 1 if edge_cache['type'] == 'I' else 0
                Rt = edge_cache['Rt'].astype(np.float32)
                Rt_inv = cam_opt.camera_pose_inv(Rt[:3, :3], Rt[:3, 3])
                edge_rel_Rt.append(torch.from_numpy(Rt_inv))
            elif '%d-%d' % (n2, n1) in edge_local_feat_cache:
                edge_cache = edge_local_feat_cache['%d-%d' % (n2, n1)]
                if self.load_keypt_match:
                    pts1 = torch.from_numpy(edge_cache['n2_feat_pos'])
                    pts2 = torch.from_numpy(edge_cache['n1_feat_pos'])
                edge_type[i] = 1 if edge_cache['type'] == 'I' else 0
                Rt_n2ton1 = edge_cache['Rt'].astype(np.float32)
                edge_rel_Rt.append(torch.from_numpy(Rt_n2ton1))
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

        out_graph_mat = torch.from_numpy(out_graph_mat)
        out_covis_mat = torch.from_numpy(out_covis_mat)
        edge_feats = torch.cat(edge_feats, dim=0)

        if len(edge_subnode_idx) != len(edge_rel_Rt):
            raise Exception("Error")

        return dataset_name, idx, img_names, imgs, img_ori_dim, cam_Es, cam_Ks, out_graph_mat, img_id2sub_id, sub_id2img_id, out_covis_mat, edge_subnode_idx, edge_type, edge_local_matches_n1, edge_local_matches_n2, edge_rel_Rt, node_feats, edge_feats
