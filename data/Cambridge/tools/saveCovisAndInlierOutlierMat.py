#%%
import os, sys
sys.path.append("/media/lihengl/ssd500/pg_akt")
import data.oned_sfm.sfminit as sfminit
from data.oned_sfm.SceneModel import ODSceneModel
from tqdm import tqdm
import numpy as np
import copy, sys, argparse, os, glob, pickle, random
from data.ambi.read_helper import read_features_from_file

from core_3dv.essential_mat import *
import core_3dv.camera_operator as cam_opt
import scipy.linalg as linalg
from skimage.feature import match_descriptors
from visualizer.visualizer_2d import show_multiple_img
from data.Cambridge.tools.utils import *

import core_3dv.camera_operator as cam_opt

def rel_t_err(t1, t2):
    t1 = t1 / np.linalg.norm(t1)
    t2 = t2 / np.linalg.norm(t2)
    
    dot = np.dot(t1, t2)
    dot = np.clip(dot, -1, 1)
    t_err = np.arccos(dot)
    return t_err

def get_covis_map(pts, cams):
    pt_cnt = 0
    covis_map = np.zeros((n_cameras, n_cameras), dtype=np.float32)
    for pt in pts:
        #print(pt.observations)
        pt_cnt += 1
        if pt_cnt%1000 ==0:
            print(pt_cnt)
        if len(pt.observations) == 0 :
            print(pt)
            break
        
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


#%%
if __name__ =="__main__":

#%%
    base_dir = sys.argv[1]
    feat_match_filename = sys.argv[2]
    bundle_out_filename = sys.argv[3]
    bundle_out_imagelist_filename = sys.argv[4]
    feat_position_bin_filename = sys.argv[5]
    print("base_dir: %s" % base_dir)
    print("feat_match_filename: %s" % os.path.join(base_dir, feat_match_filename))
    print("bundle_out_filename: %s" % os.path.join(base_dir, bundle_out_filename))
    print("bundle image list: %s" % os.path.join(base_dir, bundle_out_imagelist_filename))
    print("feat_position_bin_filename: %s" % os.path.join(base_dir, feat_position_bin_filename))

#%%

# Read From Bundle
    bundle_result = sfminit.Bundle.from_file(os.path.join(base_dir, bundle_out_filename))
    bundle_imagelist = read_image_list(os.path.join(base_dir, bundle_out_imagelist_filename))

    bundle_recon_cams = bundle_result.cameras
    bundle_recon_pts = bundle_result.points

    print('%d cameras' % len(bundle_recon_cams))

    n_cameras = len(bundle_recon_cams)
    covis_map = get_covis_map(bundle_recon_pts, bundle_recon_cams)
    np.save(os.path.join(base_dir, 'covis_map'), covis_map)
