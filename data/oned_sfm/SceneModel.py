import torch, os, sys
import numpy as np
from data.ambi.ambi_parser import readAmbiEGWithMatch
from data.ambi.read_helper import read_calibration
import data.oned_sfm.sfminit as sfminit
import core_3dv.camera_operator as cam_opt

class ODSceneModel:

    def __init__(self, iccv_raw_dir, EGs_file_name='EGs.txt'):
        # self.odsfm_seq_dir = self.odsfm_seq_dir
        self.iccv_raw_dir = iccv_raw_dir

        # read image list
        image_list_path = os.path.join(iccv_raw_dir, 'ImageList.txt')
        if not os.path.exists(image_list_path):
            raise Exception('%s not exsits' % image_list_path)

        with open(image_list_path) as f:
            i15_frame_list = f.readlines()
            i15_frame_list = [i_frame.strip() for i_frame in i15_frame_list]
            self.frame_list = i15_frame_list

        # read pairwise
        eg_file_path = os.path.join(iccv_raw_dir, EGs_file_name)
        if not os.path.exists(eg_file_path):
            raise Exception('%s not exsits' % eg_file_path)

        pairs, Rs, ts, match_number, matches = readAmbiEGWithMatch(eg_file_path)
        self.edge_pair = pairs
        self.edge_matches = matches
        self.edge_rel_E = []
        for i in range(len(Rs)):
            R, t = Rs[i], ts[i]
            E = np.zeros((3, 4), dtype=np.float32)
            E[:, :3] = R
            E[:3, 3] = t.ravel()
            self.edge_rel_E.append(E)

        # read calibration
        self.cam_calib = read_calibration(os.path.join(iccv_raw_dir, 'calibration.txt'))

        iccv15_path = os.path.join(iccv_raw_dir, 'bundle.out')
        if not os.path.exists(iccv15_path):
            raise Exception('%s not exsits' % iccv15_path)
        iccv_bundle = sfminit.Bundle.from_file(iccv15_path)
        self.recon_pts = iccv_bundle.points
        self.recon_cams = iccv_bundle.cameras
        self.cam_feats_bank = None
        self.covis_map = None


    def num_edges(self):
        return len(self.edge_pair)

    def edge_node_indices(self, edge_idx):
        pair = self.edge_pair[edge_idx]
        return pair[0], pair[1]

    def edge_node_frame_list(self, edge_idx):
        pair = self.edge_pair[edge_idx]
        frame_a = self.frame_list[pair[0]]
        frame_b = self.frame_list[pair[1]]
        return frame_a, frame_b

    def edge_rel_Rt(self, edge_idx):
        return self.edge_rel_E[edge_idx]

    def edge_feat_match(self, edge_idx):
        matched_feat_arrays = self.edge_matches[edge_idx]
        matched_feat_arrays = np.asarray(matched_feat_arrays)
        n1_sift_feats = matched_feat_arrays[:, :5]
        n2_sift_feats = matched_feat_arrays[:, 5:]

        n1_feat_id = n1_sift_feats[:, 0].astype(np.int)
        n2_feat_id = n2_sift_feats[:, 0].astype(np.int)

        n1_feat_pos = n1_sift_feats[:, -2:]
        n2_feat_pos = n2_sift_feats[:, -2:]

        return n1_feat_id, n1_feat_pos, n2_feat_id, n2_feat_pos


    def recon_cam_Es(self, cam_idx):
        E = np.zeros((3, 4), dtype=np.float32)
        E[:, :3] = self.recon_cams[cam_idx].R
        E[:, 3] = self.recon_cams[cam_idx].t
        return E

    def recon_cam_focal(self, cam_idx):
        return self.recon_cams[cam_idx].f

    def build_feat_dict(self):
        if self.cam_feats_bank is None:
            self.cam_feats_bank = dict()
            for e_i, edge_nodes in enumerate(self.edge_pair):

                n1 = edge_nodes[0]
                n2 = edge_nodes[1]

                n1_mfeat_id, n1_mfeat_pos, n2_mfeat_id, n2_mfeat_pos = self.edge_feat_match(e_i)

                if n1 not in self.cam_feats_bank:
                    self.cam_feats_bank[n1] = dict()
                if n2 not in self.cam_feats_bank:
                    self.cam_feats_bank[n2] = dict()

                for f_i, m_feat_id in enumerate(n1_mfeat_id):
                    self.cam_feats_bank[n1][m_feat_id] = n1_mfeat_pos[f_i].ravel()

                for f_i, m_feat_id in enumerate(n2_mfeat_id):
                    self.cam_feats_bank[n2][m_feat_id] = n2_mfeat_pos[f_i].ravel()

    def connection_map(self):
        n_cameras = len(self.recon_cams)
        self.connect_edges = np.zeros((n_cameras, n_cameras), dtype=np.float32)
        for e_i, edge_nodes in enumerate(self.edge_pair):
            n1 = edge_nodes[0]
            n2 = edge_nodes[1]
            self.connect_edges[n1, n2] = 1
            self.connect_edges[n2, n1] = 1
        return self.connect_edges

    def get_covis_map(self, only_edge=True):
        if self.cam_feats_bank is None:
            self.build_feat_dict()

        edge_dict = dict()
        for e_i, edge_nodes in enumerate(self.edge_pair):
            n1 = edge_nodes[0]
            n2 = edge_nodes[1]
            edge_dict['%d-%d' % (n1, n2)] = True

        n_cameras = len(self.recon_cams)
        self.covis_map = np.zeros((n_cameras, n_cameras), dtype=np.float32)
        for pt in self.recon_pts:
            obs = np.asarray(pt.observations)[:, :2].astype(np.int)
            obs_cam_indices = obs[:, 0]
            obs_cam_feats = obs[:, 1]
            valid_flag = np.ones_like(obs_cam_indices, dtype=np.uint8)
            for o_i in range(obs_cam_indices.shape[0]):
                obs_cam_id = obs_cam_indices[o_i]
                obs_feat_id = obs_cam_feats[o_i]
                # if obs_cam_id not in self.cam_feats_bank or obs_feat_id not in self.cam_feats_bank[obs_cam_id]:
                #     valid_flag[o_i] = 0
                # else:
                #     valid_flag[o_i] = 1

            for cam_i in range(obs_cam_indices.shape[0]):
                for cam_j in range(cam_i, obs_cam_indices.shape[0]):
                    if cam_i == cam_j:
                        continue
                    # if valid_flag[cam_i] == 0 or valid_flag[cam_j] == 0:
                    #     continue

                    n1 = obs_cam_indices[cam_i]
                    n2 = obs_cam_indices[cam_j]

                    if only_edge and'%d-%d' % (n1, n2) in edge_dict or '%d-%d' % (n2, n1) in edge_dict:
                        self.covis_map[n1, n2] += 1
                        self.covis_map[n2, n1] += 1
                    elif only_edge is False:
                        self.covis_map[n1, n2] += 1
                        self.covis_map[n2, n1] += 1
        return self.covis_map

    def filtering(self):
        def rel_t_err(t1, t2):

            t1 = t1 / np.linalg.norm(t1)
            t2 = t2 / np.linalg.norm(t2)

            dot = np.dot(t1, t2)
            dot = np.clip(dot, -1, 1)
            t_err = np.arccos(dot)
            return t_err

        inlier_R_threshold = 5
        inlier_t_threshold = 25
        outlier_R_threshold = 90
        outlier_t_threshold = 1
        inlier_covis_threshold = 50
        R_errs = []
        t_errs = []

        n_Cameras = len(self.recon_cams)
        inoutMat = np.zeros([n_Cameras, n_Cameras])
        for e_i, edge in enumerate(self.edge_pair):
            n1 = edge[0]
            n2 = edge[1]

            if self.covis_map[n1, n2] == 0:
                continue

            n1_E = self.recon_cam_Es(n1)
            n2_E = self.recon_cam_Es(n2)
            rel_Rt_gt = cam_opt.relateive_pose(n2_E[:3, :3], n2_E[:3, 3], n1_E[:3, :3], n1_E[:3, 3])
            rel_Rt_pred = self.edge_rel_Rt(e_i)

            rel_R_gt = rel_Rt_gt[:3, :3]
            rel_t_gt = rel_Rt_gt[:3, 3]
            rel_t_gt = rel_t_gt / np.linalg.norm(rel_t_gt)

            rel_R = np.matmul(rel_Rt_pred[:3, :3], rel_R_gt.T)
            R_err = np.rad2deg(np.arccos((np.trace(rel_R) - 1) / 2))
            R_errs.append(R_err)

            t_err = np.rad2deg(rel_t_err(rel_Rt_pred[:3, 3].ravel(), rel_t_gt.ravel()))
            t_errs.append(t_err)

            if self.covis_map[n1, n2] > inlier_covis_threshold:
                if R_err < inlier_R_threshold and t_err < inlier_t_threshold:
                    inoutMat[n1, n2] = inoutMat[n2, n1] = 1
                elif R_err > outlier_R_threshold or t_err > outlier_t_threshold:
                    inoutMat[n1, n2] = inoutMat[n2, n1] = -1

            return inoutMat, R_errs, t_errs