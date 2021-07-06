import torch, os, sys
import numpy as np
from data.ambi.ambi_parser import readAmbiEGWithMatch
from data.ambi.read_helper import read_calibration
import data.oned_sfm.sfminit as sfminit
import core_3dv.camera_operator as cam_opt
from data.util.vsfm_file_paser import parse_vsfm_feat_match, parse_vsfm_matches


class CaptureSceneModel:

    def __init__(self, base_dir, match_file_name='f_mat.txt'):
        self.base_dir = base_dir

        # read the image list
        image_list_path = os.path.join(self.base_dir, 'bundle.out.list.txt')
        with open(image_list_path) as f:
            frame_list = f.readlines()
            self.frame_list = [i_frame.strip() for i_frame in frame_list]
            self.frame_idx_map = dict()
            for i, frame in enumerate(self.frame_list):
                self.frame_idx_map[frame] = i

        # read the pairs
        match_file_path = os.path.join(self.base_dir,  match_file_name)
        self.matches = parse_vsfm_matches(match_file_path)

        # read calibration
        calib_path = os.path.join(base_dir, 'cam.txt')
        with open(calib_path) as f:
            cameras = f.readlines()
            self.calib_dict = dict()
            for c_line in cameras:
                #         print(c_line)
                c_tokens = c_line.split(",")
                camera_name = c_tokens[0]
                f, cx, cy = [float(c_tokens[i].strip()) for i in range(1, 4)]
                self.calib_dict[camera_name] = (f, cx, cy)

        # read the bundle results
        bundle_path = os.path.join(self.base_dir, 'bundle.out')
        bundle = sfminit.Bundle.from_file(bundle_path)

        self.recon_pts = bundle.points
        self.recon_cams = bundle.cameras
        self.covis_map = None

    def num_edges(self):
        return len(self.matches)

    def edge_node_indices(self, edge_idx):
        e = self.matches[edge_idx]
        n1_name = e['ids_name'][0] + '.jpg'
        n2_name = e['ids_name'][1] + '.jpg'

        if n1_name not in self.frame_idx_map or n2_name not in self.frame_idx_map:
            return -1, -1

        n1 = self.frame_idx_map[n1_name]
        n2 = self.frame_idx_map[n2_name]

        return n1, n2

    def edge_node_frame_list(self, edge_idx):
        e = self.matches[edge_idx]
        n1_name = e['ids_name'][0] + '.jpg'
        n2_name = e['ids_name'][1] + '.jpg'

        if n1_name not in self.frame_idx_map or n2_name not in self.frame_idx_map:
            return None, None
        return n1_name, n2_name

    def edge_rel_Rt(self, edge_idx):
        def build_E(R, t):
            E = np.zeros((3, 4), dtype=np.float32)
            E[:, :3] = R
            E[:, 3] = t
            return E

        e = self.matches[edge_idx]
        n1_name = e['ids_name'][0] + '.jpg'
        n2_name = e['ids_name'][1] + '.jpg'

        if n1_name not in self.frame_idx_map or n2_name not in self.frame_idx_map:
            return None

        n1 = self.frame_idx_map[n1_name]
        n2 = self.frame_idx_map[n2_name]

        # rotations
        n1_Rt = build_E(self.recon_cams[n1].R, self.recon_cams[n1].t)
        n2_Rt = build_E(self.recon_cams[n2].R, self.recon_cams[n2].t)

        rel_Rt_gt = cam_opt.relateive_pose(n1_Rt[:, :3], n1_Rt[:, 3],
                                           n2_Rt[:, :3], n2_Rt[:, 3])

        return rel_Rt_gt

    def edge_feat_match(self, edge_idx):

        e = self.matches[edge_idx]
        n1_name = e['ids_name'][0] + '.jpg'
        n2_name = e['ids_name'][1] + '.jpg'

        if n1_name not in self.frame_idx_map or n2_name not in self.frame_idx_map:
            return None, None

        n1_feat_pos = np.asarray(e['n1_feat_pos'], dtype=np.float32)
        n2_feat_pos = np.asarray(e['n2_feat_pos'], dtype=np.float32)
        return n1_feat_pos, n2_feat_pos


    def recon_cam_Es(self, cam_idx):
        E = np.zeros((3, 4), dtype=np.float32)
        E[:, :3] = self.recon_cams[cam_idx].R
        E[:, 3] = self.recon_cams[cam_idx].t
        return E

    def recon_cam_focal(self, cam_idx):
        cam_name = self.frame_list[cam_idx]
        if cam_name not in self.calib_dict:
            return None
        f = self.calib_dict[cam_name][0]
        return f

    def connection_map(self):
        n_cameras = len(self.recon_cams)
        self.connect_edges = np.zeros((n_cameras, n_cameras), dtype=np.float32)
        for e_i, e in enumerate(self.matches):
            n1_name = e['ids_name'][0] + '.jpg'
            n2_name = e['ids_name'][1] + '.jpg'

            if n1_name not in self.frame_idx_map or n2_name not in self.frame_idx_map:
                continue

            n1 = self.frame_idx_map[n1_name]
            n2 = self.frame_idx_map[n2_name]
            self.connect_edges[n1, n2] = 1
            self.connect_edges[n2, n1] = 1
        return self.connect_edges

    def get_covis_map(self):
        # build covis map
        edge_dict = dict()
        for e_i, e in enumerate(self.matches):
            n1_name = e['ids_name'][0] + '.jpg'
            n2_name = e['ids_name'][1] + '.jpg'

            if n1_name not in self.frame_idx_map or n2_name not in self.frame_idx_map:
                continue

            n1 = self.frame_idx_map[n1_name]
            n2 = self.frame_idx_map[n2_name]
            edge_dict['%d-%d' % (n1, n2)] = True

        recon_pts = self.recon_pts
        recon_cams = self.recon_cams

        n_cameras = len(recon_cams)
        covis_map = np.zeros((n_cameras, n_cameras), dtype=np.float32)
        covis_count = 0

        for pt in recon_pts:
            obs = np.asarray(pt.observations)[:, :2].astype(np.int)
            obs_cam_indices = obs[:, 0]
            # obs_cam_feats = obs[:, 1]

            for cam_i in range(obs_cam_indices.shape[0]):
                for cam_j in range(cam_i, obs_cam_indices.shape[0]):
                    if cam_i == cam_j:
                        continue

                    n1 = obs_cam_indices[cam_i]
                    n2 = obs_cam_indices[cam_j]

                    if '%d-%d' % (n1, n2) in edge_dict or '%d-%d' % (n2, n1) in edge_dict:
                        covis_map[n1, n2] += 1
                        covis_map[n2, n1] += 1

        print('added %d pairs' % np.nonzero(covis_map)[0].shape[0])
        self.covis_map = covis_map
        return covis_map

    # def filtering(self):
    #     def rel_t_err(t1, t2):
    #
    #         t1 = t1 / np.linalg.norm(t1)
    #         t2 = t2 / np.linalg.norm(t2)
    #
    #         dot = np.dot(t1, t2)
    #         dot = np.clip(dot, -1, 1)
    #         t_err = np.arccos(dot)
    #         return t_err
    #
    #     inlier_R_threshold = 5
    #     inlier_t_threshold = 25
    #     outlier_R_threshold = 90
    #     outlier_t_threshold = 1
    #     inlier_covis_threshold = 50
    #     R_errs = []
    #     t_errs = []
    #
    #     n_Cameras = len(self.recon_cams)
    #     inoutMat = np.zeros([n_Cameras, n_Cameras])
    #     for e_i, edge in enumerate(self.edge_pair):
    #         n1 = edge[0]
    #         n2 = edge[1]
    #
    #         if self.covis_map[n1, n2] == 0:
    #             continue
    #
    #         n1_E = self.recon_cam_Es(n1)
    #         n2_E = self.recon_cam_Es(n2)
    #         rel_Rt_gt = cam_opt.relateive_pose(n2_E[:3, :3], n2_E[:3, 3], n1_E[:3, :3], n1_E[:3, 3])
    #         rel_Rt_pred = self.edge_rel_Rt(e_i)
    #
    #         rel_R_gt = rel_Rt_gt[:3, :3]
    #         rel_t_gt = rel_Rt_gt[:3, 3]
    #         rel_t_gt = rel_t_gt / np.linalg.norm(rel_t_gt)
    #
    #         rel_R = np.matmul(rel_Rt_pred[:3, :3], rel_R_gt.T)
    #         R_err = np.rad2deg(np.arccos((np.trace(rel_R) - 1) / 2))
    #         R_errs.append(R_err)
    #
    #         t_err = np.rad2deg(rel_t_err(rel_Rt_pred[:3, 3].ravel(), rel_t_gt.ravel()))
    #         t_errs.append(t_err)
    #
    #         if self.covis_map[n1, n2] > inlier_covis_threshold:
    #             if R_err < inlier_R_threshold and t_err < inlier_t_threshold:
    #                 inoutMat[n1, n2] = inoutMat[n2, n1] = 1
    #             elif R_err > outlier_R_threshold or t_err > outlier_t_threshold:
    #                 inoutMat[n1, n2] = inoutMat[n2, n1] = -1
    #
    #         return inoutMat, R_errs, t_errs