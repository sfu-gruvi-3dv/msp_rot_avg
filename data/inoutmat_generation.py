# %%
import sys
sys.path.append(
    '/media/lihengl/ssd500/pg_akt_old/libs/5point_alg_pybind/build')

from data.util.vsfm_file_paser import parse_vsfm_matches, parse_vsfm_feat_match
import five_point_alg
import data.oned_sfm.sfminit as sfminit
from data.oned_sfm.SceneModel import ODSceneModel
from tqdm import tqdm
import numpy as np
import copy
import sys
import argparse
import os
import glob
import pickle
import random
from data.ambi.read_helper import read_features_from_file
import argparse

from core_3dv.essential_mat import *
import core_3dv.camera_operator as cam_opt
import scipy.linalg as linalg
from skimage.feature import match_descriptors
from visualizer.visualizer_2d import show_multiple_img
import core_3dv.camera_operator as cam_opt

def rel_t_err(t1, t2):
    t1 = t1 / np.linalg.norm(t1)
    t2 = t2 / np.linalg.norm(t2)

    dot = np.dot(t1, t2)
    dot = np.clip(dot, -1, 1)
    t_err = np.arccos(dot)
    return t_err


def get_covis_map(pts, cams):
    covis_map = np.zeros((n_cameras, n_cameras), dtype=np.float32)
    for pt in pts:
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
                covis_map[n1, n2] += 1
                covis_map[n2, n1] += 1
    return covis_map


def read_image_list(filename):
    with open(filename, "r") as fin:
        image_list = fin.readlines()
    image_list = [x.replace("\\","/") for x in image_list]
    image_list = [os.path.splitext(os.path.split(x)[-1])[0]
                  for x in image_list]
    image2idx = dict()
    for idx, file in enumerate(image_list):
        image2idx[file] = idx
    return image_list, image2idx


# %%
if __name__ == "__main__":
    # %%
    parser = argparse.ArgumentParser(
        description="Process the datasets inoutmat")
    parser.add_argument("--dataset", type=str, required=True,
                        help="The dataset name, example: cambridge or 1dsfm")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="the base dir, other dir will be relative path of base dir")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="the directory path of image file")
    parser.add_argument("--feat_match_file", type=str, default=None, required=True,
                        help="The path of feat meatch file, the number of inlier match after computing E")
    parser.add_argument("--f_mat", type=str, default=None, required=True,
                        help="The path of f mat, the number of matches")
    parser.add_argument("--bundle_out_file", type=str, default=None, required=True,
                        help="the bundle out file, only in bundler v3.0")
    parser.add_argument("--bundle_out_imagelist", type=str, default=None, required=True,
                        help="the bundle out image list")
    parser.add_argument("--feat_pos_bin", type=str, default=None,
                        help="the feat position file in binary")
    parser.add_argument("--inlier_match_threshold", type=int, default=30,
                        help="featmatch(inlier) > inlier_threshold")
    parser.add_argument("--inlier_r_threshold", type=float, default=20,
                        help="R_err < inlier_threshold")
    parser.add_argument("--inlier_t_threshold", type=float, default=20,
                        help="T_err < inlier_threshold")
    parser.add_argument("--outlier_ratio", type=float, default=0.16,
                        help="\\frac\{covis\}\{featmatch\} < outlier threshold;\nthe ratio of covis_inlier_ratio")
    parser.add_argument("--outlier_r_threshold", type=float, default=60,
                        help="r_err > outlier threshold")
    parser.add_argument("--outlier_t_threshold", type=float, default=60,
                        help="t_err > outlier threshold")

    parser.add_argument("--output_covismat", type=str, default="covis_map",
                        help="The output path of covis mat")
    parser.add_argument("--output_inoutmat", type=str, default="inoutMat",
                        help="The path of inout mat")
    parser.add_argument("--output_sift_feat_pos", type=str, default="edge_feat_pos_cache.bin",
                        help="The path of sift postion bin")

    args = parser.parse_args()

    # input directory
    base_dir = args.base_dir
    image_dir = args.image_dir
    feat_match_filename = args.feat_match_file
    f_mat = args.f_mat
    bundle_out_filename = args.bundle_out_file
    bundle_out_imagelist_filename = args.bundle_out_imagelist
    feat_position_bin_filename = args.feat_pos_bin
#%%
    # base_dir = "/media/lihengl/t2000/Dataset/Cambridge_pg/GreatCourt_jpg"
    # image_dir = "./"
    # feat_match_filename = "./feat_match_all.txt"
    # f_mat = "./f_mat.txt"
    # bundle_out_filename = "./bundle.out"
    # bundle_out_imagelist_filename = "./bundle-list.txt"
    # feat_position_bin_filename = "./sift_feat_pos.bin"

    # output directory
    output_covismat = args.output_covismat
    output_inoutmat = args.output_inoutmat
    output_feat_pos_bin = args.output_sift_feat_pos
    # output_covismat = "./covis_map"
    # output_inoutmat = "./inout_mat"
    # output_feat_pos_bin = "./edge_feat_pos_cache.bin"

    # inlier threshold
    inlier_match_threshold = args.inlier_match_threshold
    inlier_r_threshold = args.inlier_r_threshold
    inlier_t_threshold = args.inlier_t_threshold

    # inlier_match_threshold = 30
    # inlier_r_threshold = 20
    # inlier_t_threshold = 20
    # outlier threshold
    outlier_ratio = args.outlier_ratio
    outlier_r_threshold = args.outlier_r_threshold
    outlier_t_threshold = args.outlier_t_threshold
    outlier_ratio = 0.16
    outlier_r_threshold = 60
    outlier_t_threshold = 60

    # output config
    print("base_dir: %s" % base_dir)
    print("image_dir: %s" % os.path.join(base_dir, image_dir))
    print("feat_match_filename: %s" %
          os.path.join(base_dir, feat_match_filename))
    print("f mat: %s" % os.path.join(base_dir, f_mat))
    print("bundle_out_filename: %s" %
          os.path.join(base_dir, bundle_out_filename))
    print("bundle image list: %s" % os.path.join(
        base_dir, bundle_out_imagelist_filename))
    print("feat_position_bin_filename: %s" %
          os.path.join(base_dir, feat_position_bin_filename))
    print("inlier match threshold:" + str(inlier_match_threshold))
    print("inlier r threshold:" + str(inlier_r_threshold))
    print("inlier t threshold:" + str(inlier_t_threshold))
    print("outlier ratio:" + str(outlier_ratio))
    print("outlier r threshold:" + str(outlier_r_threshold))
    print("outlier t threshold:" + str(outlier_t_threshold))
#%%
    bundle_result = sfminit.Bundle.from_file(
        os.path.join(base_dir, bundle_out_filename))
    bundle_imagelist, bundle_image2idx = read_image_list(
        os.path.join(base_dir, bundle_out_imagelist_filename))

    bundle_recon_cams = bundle_result.cameras
    bundle_recon_pts = bundle_result.points
#%%
    matches = parse_vsfm_feat_match(
        os.path.join(base_dir, feat_match_filename))
    #egs = parse_vsfm_matches(os.path.join(base_dir, f_mat))

    if 'n1_feat_pos' not in matches[0]:
        with open(os.path.join(base_dir, feat_position_bin_filename), "rb") as fin:
            sift_feat_pos = pickle.load(fin)

    print('%d matches' % len(matches))
    print('%d cameras' % len(bundle_recon_cams))

    n_cameras = len(bundle_recon_cams)
    covis_map = get_covis_map(bundle_recon_pts, bundle_recon_cams)
    np.save(os.path.join(base_dir, ), covis_map)

    random.shuffle(matches)
    res = []
#%%
    print("Processing matches")
    for e_i in tqdm(range(len(matches))):
    
        match = matches[e_i]
        m = match['ids_name']
        n1_name, n2_name = m

        if n1_name not in bundle_image2idx or n2_name not in bundle_image2idx:
            continue

        if len(match['n1_feat_id']) == 0 or len(match['n2_feat_id']) == 0:
            continue

        n1 = bundle_image2idx[n1_name]
        n2 = bundle_image2idx[n2_name]

        K1 = bundle_recon_cams[n1].K_matrix()
        K2 = bundle_recon_cams[n2].K_matrix()
#%%
        if 'n1_feat_pos' not in matches[0]:
            n1_feat_pos_array = sift_feat_pos[n1_name]
            n2_feat_pos_array = sift_feat_pos[n2_name]

            pts1 = [n1_feat_pos_array[feat_id]
                    for feat_id in match['n1_feat_id']]
            pts2 = [n2_feat_pos_array[feat_id]
                    for feat_id in match['n2_feat_id']]
            pts1 = np.asarray(pts1)[:, :2]
            pts2 = np.asarray(pts2)[:, :2]
        else:
            pts1 = match['n1_feat_pos']
            pts2 = match['n2_feat_pos']

        K1 = K1.astype(np.float32)
        K2 = K2.astype(np.float32)
        pts1 = np.asarray(pts1).astype(np.float32)
        pts2 = np.asarray(pts2).astype(np.float32)
        R, t, inliers = five_point_alg.estimate_ransac_E(
            pts1, pts2, K1, K2, 2.0, 200)
        Rt = np.zeros((3, 4), dtype=np.float32)
        Rt[:3, :3] = R
        Rt[:3, 3] = t
        res.append((e_i, n1, n2, Rt, inliers, pts1.shape[0]))

    print("size of res:%s", len(res))
    R_errs, t_errs = [], []
    outliers = []
    inlier_edges = []
    in_Rs, in_ts = [], []

    edge_feat_pos_cache = dict()
    print("Processing res")
    for i in tqdm(range(len(res))):
        pair = res[i]
        m_i = pair[0]
        n1 = pair[1]
        n2 = pair[2]
        rel_Rt = pair[3]
        inliers = pair[4]
        num_matches = pair[5]
        if inliers == 0:
            continue

        match = matches[m_i]
        m = match['ids_name']
        n1_name, n2_name = m

        if n1_name not in bundle_image2idx or n2_name not in bundle_image2idx:
            continue

        if len(match['n1_feat_id']) == 0 or len(match['n2_feat_id']) == 0:
            continue

        assert n1 == pair[1] and n2 == pair[2]

        K1 = bundle_recon_cams[n1].K_matrix()
        K2 = bundle_recon_cams[n2].K_matrix()
        covis = max(int(covis_map[n1, n2]), int(covis_map[n2, n1]))

        n1_Rt = bundle_recon_cams[n1].recon_cam_Es()
        n2_Rt = bundle_recon_cams[n2].recon_cam_Es()

        n1_C = cam_opt.camera_center_from_Tcw(n1_Rt[:3, :3], n1_Rt[:3, 3])
        n2_C = cam_opt.camera_center_from_Tcw(n2_Rt[:3, :3], n2_Rt[:3, 3])

        if n1_Rt[2, 2] == 0 or n2_Rt[2, 2] == 0:
            continue
        rel_Rt_gt = cam_opt.relateive_pose(n1_Rt[:, :3], n1_Rt[:, 3],
                                           n2_Rt[:, :3], n2_Rt[:, 3])
        Rij = rel_Rt_gt[:, :3]
        tij = rel_Rt_gt[:, 3]

        tij = tij / np.sqrt(np.sum(np.square(tij)))

        rel_R = np.matmul(rel_Rt[:3, :3], Rij.T)
        R_err = np.rad2deg(np.arccos((np.trace(rel_R) - 1) / 2))
        t_err = np.rad2deg(rel_t_err(tij, rel_Rt[:3, 3]))

        if (inliers > inlier_match_threshold and R_err < inlier_r_threshold and
            t_err < inlier_t_threshold) or (covis > 30 and R_err < inlier_r_threshold and
                                            t_err < inlier_t_threshold):
            inlier_edges.append((e_i, n1, n2, covis, R_err, t_err))

            if 'n1_feat_pos' not in matches[0]:
                n1_feat_pos_array = sift_feat_pos[n1_name]
                n2_feat_pos_array = sift_feat_pos[n2_name]

                pts1 = [n1_feat_pos_array[feat_id]
                        for feat_id in match['n1_feat_id']]
                pts2 = [n2_feat_pos_array[feat_id]
                        for feat_id in match['n2_feat_id']]
                pts1 = np.asarray(pts1)[:, :2]
                pts2 = np.asarray(pts2)[:, :2]
            else:
                pts1 = match['n1_feat_pos']
                pts2 = match['n2_feat_pos']
                pts1 = np.asarray(pts1).astype(np.float32)
                pts2 = np.asarray(pts2).astype(np.float32)
                n1_feat_id = [match[f_id][0] for f_id in range(len(match))]
                n2_feat_id = [match[f_id][5] for f_id in range(len(match))]
                n1_feat_id = np.asarray(n1_feat_id).astype(np.int)
                n2_feat_id = np.asarray(n2_feat_id).astype(np.int)

            if '%d-%d' % (n1, n2) not in edge_feat_pos_cache and '%d-%d' % (n2, n1) not in edge_feat_pos_cache:
                edge_feat_pos_cache['%d-%d' % (n1, n2)] = {
                    'n1_feat_id': n1_feat_id,
                    'n2_feat_id': n2_feat_id,
                    'n1_feat_pos': pts1,
                    'n2_feat_pos': pts2,
                    'type': 'I'
                }
            in_Rs.append(R_err)
            in_ts.append(t_err)
            continue

        if (covis != 0 and covis/inliers < outlier_ratio and (R_err > outlier_r_threshold or
                                                              t_err > outlier_t_threshold)):
            outliers.append((i, n1, n2, inliers))

            n1_feat_id = np.asarray(match['n1_feat_id']).astype(np.int)
            n2_feat_id = np.asarray(match['n2_feat_id']).astype(np.int)

            if 'n1_feat_pos' not in matches[0]:
                n1_feat_pos_array = sift_feat_pos[n1_name.strip()]
                n2_feat_pos_array = sift_feat_pos[n2_name.strip()]

                pts1 = [n1_feat_pos_array[feat_id]
                        for feat_id in match['n1_feat_id']]
                pts2 = [n2_feat_pos_array[feat_id]
                        for feat_id in match['n2_feat_id']]
                pts1 = np.asarray(pts1)[:, :2]
                pts2 = np.asarray(pts2)[:, :2]
            else:
                pts1 = match['n1_feat_pos']
                pts2 = match['n2_feat_pos']

            pts1 = np.asarray(pts1).astype(np.float32)
            pts2 = np.asarray(pts2).astype(np.float32)

            edge_feat_pos_cache['%d-%d' % (n1, n2)] = {
                'n1_feat_id': n1_feat_id,
                'n2_feat_id': n2_feat_id,
                'n1_feat_pos': pts1,
                'n2_feat_pos': pts2,
                'type': 'O'
            }
    #         break
        R_errs.append(R_err)
        t_errs.append(t_err)

    # save the results
    print("save the result")
    inout_mat_path = os.path.join(base_dir, output_inoutmat)
    edge_feat_pos_cache_path = os.path.join(
        base_dir, output_feat_pos_bin)
    n_Cameras = len(bundle_recon_cams)
    inout_mat = np.zeros((n_Cameras, n_Cameras), dtype=np.int)
    for outlier_pair in tqdm(outliers):
        n1 = outlier_pair[1]
        n2 = outlier_pair[2]
        inout_mat[n1, n2] = inout_mat[n2, n1] = -1
    for inlier_pair in tqdm(inlier_edges):
        n1 = inlier_pair[1]
        n2 = inlier_pair[2]
        inout_mat[n1, n2] = inout_mat[n2, n1] = 1
    np.save(inout_mat_path, inout_mat)

    with open(edge_feat_pos_cache_path, 'wb') as f:
        pickle.dump(edge_feat_pos_cache, f)
