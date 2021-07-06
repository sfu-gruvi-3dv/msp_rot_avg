import torch, os, sys
from tqdm import tqdm
import numpy as np
from data.ambi.ambi_parser import readAmbiEGWithMatch
from data.ambi.read_helper import read_calibration
import data.oned_sfm.sfminit as sfminit
import core_3dv.camera_operator as cam_opt
from data.util.colmap_database import COLMAPDatabase, blob_to_array, pair_id_to_image_ids
sys.path.append('/mnt/Tango/pg/libs/OpenSfM/build/lib/')
from opensfm.reconstruction import two_view_reconstruction
from opensfm import pygeometry
from opensfm import features
from opensfm.types import PerspectiveCamera, BrownPerspectiveCamera

def build_perspective_cam(cam_id, focal, img_dim):
    cam = BrownPerspectiveCamera()
    cam.id = cam_id
    cam.focal_x = focal
    cam.focal_y = focal
    cam.c_x = img_dim[1] * 0.5
    cam.c_y = img_dim[0] * 0.5
    cam.height = img_dim[0]
    cam.width = img_dim[1]
    cam.k1 = 0.0
    cam.k2 = 0.0
    cam.p1 = 0.0
    cam.p2 = 0.0
    cam.k3 = 0.0
    return cam

def relative_pose_from_essential_mat(E, cam1, cam2, pts1, pts2, refine_iter=600):
    # pts1_normalized = features.normalized_image_coordinates(pts1[:, :2].copy(), width=cam1.width, height=cam1.height)
    # pts2_normalized = features.normalized_image_coordinates(pts2[:, :2].copy(), width=cam2.width, height=cam2.height)

    bts1 = cam1.pixel_bearing_many(pts1.copy())
    bts2 = cam2.pixel_bearing_many(pts2.copy())
    Rt = pygeometry.relative_pose_from_essential(E, bts1, bts2)
    Rt = pygeometry.relative_pose_refinement(Rt, bts1, bts2, refine_iter)
    return Rt

class COLMAPTwoViewSceneModel:

    def __init__(self, dataset_dir, db_file_name='db.db'):
        self.iccv_raw_dir = dataset_dir
        self.db = COLMAPDatabase(os.path.join(dataset_dir, db_file_name))

        # image dict
        self.img_dict = dict(
            (name, (image_id, camera_id))
            for image_id, name, camera_id in self.db.execute("SELECT image_id, name, camera_id FROM images")
        )
        self.img_id2name = dict()
        for name, val in self.img_dict.items():
            (image_id, camera_id) = val
            self.img_id2name[image_id] = name

        # calibration
        cam_id_dict = dict(
            (cam_id, (height, width, blob_to_array(params, np.float64)))
            for cam_id, height, width, params in self.db.execute("SELECT camera_id, height, width, params FROM cameras")
        )
        self.calib_name_dict = dict()
        for name, val in self.img_dict.items():
            self.calib_name_dict[name] = cam_id_dict[val[1]]

        # pairwise
        pairs_rows = self.db.execute("SELECT * FROM two_view_geometries")
        self.pairs = []
        self.pairs_dict = dict()
        for pair_row in pairs_rows:
            pair_id, n_matches, _, matches, config, F, E, H = pair_row
            img_ids = pair_id_to_image_ids(pair_id)
            if n_matches > 0:
                matches = blob_to_array(matches, np.uint32).reshape(n_matches, 2)
                F = blob_to_array(F, np.float64).reshape(3, 3)
                E = blob_to_array(E, np.float64).reshape(3, 3)
                H = blob_to_array(H, np.float64).reshape(3, 3)
            else:
                matches = None
                F, E, H = None, None, None

            key = "%d_%d" % (int(img_ids[0]), int(img_ids[1]))

            self.pairs.append(key)
            self.pairs_dict[key] = ((int(img_ids[0]), int(img_ids[1])), matches, F, E, H)

        # build match and relative pose (todo: cache)
        self.matches_dict = dict()
        self.edge_rel_Rt_dict = dict()
        ignored_count = 0
        for key, val in tqdm(self.pairs_dict.items()):
            img_ids, matches, F, E, H = val
            n1_name = self.img_id2name[img_ids[0]]
            n2_name = self.img_id2name[img_ids[1]]
            if matches is None:
                # self.matches_dict[key] = None
                # self.edge_rel_Rt_dict[key] = None
                ignored_count += 1
                continue

            # extract correspondences
            kpts = dict(
                (image_id, blob_to_array(data, np.float32, (rows, cols)))
                for image_id, rows, cols, data in self.db.execute(
                    "SELECT image_id, rows, cols, data FROM keypoints WHERE image_id IN (%d, %d)" % (
                    int(img_ids[0]), int(img_ids[1]))))
            n1_feat_id = matches[:, 0]
            n2_feat_id = matches[:, 1]
            pts1 = kpts[int(img_ids[0])][n1_feat_id][:, :2]
            pts2 = kpts[int(img_ids[1])][n2_feat_id][:, :2]

            # build cameras
            h1, w1, params1 = self.calib_name_dict[n1_name]
            h2, w2, params2 = self.calib_name_dict[n2_name]

            cam1 = build_perspective_cam(img_ids[0], params1[0], (h1, w1))
            cam2 = build_perspective_cam(img_ids[1], params2[0], (h2, w2))
            Rt = relative_pose_from_essential_mat(E, cam1, cam2, pts1, pts2)
            self.matches_dict[key] = (n1_feat_id, pts1, n2_feat_id, pts2)
            self.edge_rel_Rt_dict[key] = Rt
        print('Ignored: %d' % ignored_count)

    def num_edges(self):
        return len(self.matches_dict)

    def edge_node_indices(self, edge_idx):
        key = self.pairs[edge_idx]
        pair = self.pairs_dict[key]
        n1, n2 = pair[0]
        return n1, n2

    def edge_node_frame_list(self, edge_idx):
        key = self.pairs[edge_idx]
        pair = self.pairs_dict[key]
        n1, n2 = pair[0]
        n1_name = self.img_id2name[n1]
        n2_name = self.img_id2name[n2]
        return n1_name, n2_name

    def edge_rel_Rt(self, edge_idx):
        key = self.pairs[edge_idx]
        pair = self.pairs_dict[key]
        n1, n2 = pair[0]
        n1_name = self.img_id2name[n1]
        n2_name = self.img_id2name[n2]
        key = "%d_%d" % (n1, n2)

        inverse_flag = False
        if key not in self.pairs_dict:
            key = "%d_%d" % (n2, n1)
            inverse_flag = True
            if key not in self.pairs_dict:
                return None

        Rt = self.edge_rel_Rt_dict[key].astype(np.float32)
        if inverse_flag is True:
            Rt = cam_opt.camera_pose_inv(Rt[:3, :3], Rt[:3, 3])

        return Rt

    def edge_F_E_mat(self, edge_idx):
        key = self.pairs[edge_idx]
        img_idx, matches, F, E, H = self.pairs_dict[key]
        return F, E

    def edge_feat_match(self, edge_idx):
        key = self.pairs[edge_idx]
        pair = self.pairs_dict[key]
        n1, n2 = pair[0]
        n1_name = self.img_id2name[n1]
        n2_name = self.img_id2name[n2]
        key = "%d_%d" % (n1, n2)

        inverse_flag = False
        if key not in self.pairs_dict:
            key = "%d_%d" % (n2, n1)
            inverse_flag = True
            if key not in self.pairs_dict:
                return None

        if self.matches_dict[key] is None:
            return None, None, None, None

        if inverse_flag is False:
            n1_feat_id, pts1, n2_feat_id, pts2 = self.matches_dict[key]
        else:
            n2_feat_id, pts2, n1_feat_id, pts1 = self.matches_dict[key]

        return n1_feat_id, pts1, n2_feat_id, pts2


if __name__ == '__main__':
    datasets = ["Kumamoto"]
    data_basedir = '/mnt/Exp_5/manual_mark/' + datasets[0]
    img_dir = data_basedir

    scene_model = COLMAPTwoViewSceneModel(data_basedir, 'colmapdb.db')
    pairs = []
    for key, val in tqdm(scene_model.pairs_dict.items()):
        img_ids, matches, F, E, H = val
        if key not in scene_model.matches_dict:
            continue
        n1_name = scene_model.img_id2name[img_ids[0]]
        n2_name = scene_model.img_id2name[img_ids[1]]
        n1_feat_id, pts1, n2_feat_id, pts2 = scene_model.matches_dict[key]
        pairs.append((n1_name, n2_name, pts1, pts2))

    print(len(pairs))
    import pickle
    with open(os.path.join(data_basedir, 'all_pairs_colmap.bin'), 'wb') as f:
        pickle.dump(pairs, f)