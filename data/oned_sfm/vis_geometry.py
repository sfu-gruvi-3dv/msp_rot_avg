import sys, os, cv2
import data.oned_sfm.sfminit as sfminit
import argparse
import numpy as np
import copy
import core_3dv.camera_operator as cam_opt
from data.ambi.ambi_parser import parserBundle
from visualizer.visualizer_3d import Visualizer
from visualizer.ipw_3d_helper import *
import ipyvolume as ipv

data_dir = '/Volumes/Resources/1DSFM/data/datasets/Gendarmenmarkt'
input = {'EGs': os.path.join(data_dir, 'EGs.txt'),
         'ccs': os.path.join(data_dir, 'cc.txt'),
         'tracks': os.path.join(data_dir, 'tracks.txt'),
         'coords': os.path.join(data_dir, 'coords.txt'),
         'gt_soln': os.path.join(data_dir, 'gt_bundle.out')}

tracks = sfminit.Tracks.from_file(input['tracks'])
coords = sfminit.Coords.from_file(input['coords'])
models = sfminit.ModelList.from_EG_file(input['EGs'])
# gt_bundle = sfminit.Bundle.from_file(input['gt_soln'])
gt_bundle2 = sfminit.Bundle.from_file('/Users/corsy/Downloads/ICCV15/Gendarmenmarkt/bundle_00635.out')

bd = parserBundle('/Users/corsy/Downloads/ICCV15/Gendarmenmarkt/bundle_00635.out')
nC, nP, cfks, cRs, cts, pPs, pCs, pVNs, pVs, pKs, pVLs = bd

# filtering the invalid cameras
invalid_cam_flag = dict()
cameras = []
for i, cam in enumerate(gt_bundle2.cameras):
    if cam.f == 0:
        invalid_cam_flag[i] = True
    else:
        E = np.zeros((3, 4), dtype=np.float32)
        E[:3, :3] = cam.R
        E[:3, 3] = cam.t

        # E_inv = cam_opt.camera_pose_inv(cam.R, cam.t)
        # E_inv = cam_opt.camera_pose_inv(cRs[i], cts[i])
        cameras.append(E)

valid_cam2idx = dict()
valid_cams = list()
n_valid_Cams = 0
max_idx_ = 0
pts = gt_bundle2.points
for pt in pts:
    obs = pt.observations
    obs_cams = [o[0] for o in obs]
    max_idx = np.max(np.asarray(obs_cams))
    if max_idx > max_idx_:
        max_idx_ = max_idx
    for o in obs_cams:
        if o not in invalid_cam_flag and o not in valid_cam2idx:
            valid_cam2idx[o] = n_valid_Cams
            valid_cams.append(o)
            n_valid_Cams += 1

# point clouds
pts = gt_bundle2.points
pts_pos = [pt.X for pt in pts]
pts_pos = np.asarray(pts_pos)
pts_center = np.mean(pts_pos, axis=0)
dist = np.linalg.norm(pts_pos - pts_center, axis=1)
dist_flag = dist < 5.0
# pts_color = [pt.color for pt in pts]
# pts_color = np.asarray(pts_color)

filtered_pts = []
for i, pt in enumerate(pts):
    if dist_flag[i] == False:
        continue
    filtered_pts.append(pt.X)
pts_pos = np.asarray(filtered_pts)

vis = Visualizer(1024, 1024)
vis.set_point_cloud(pts_pos, pt_size=2)
for cam in cameras:
    vis.add_frame_pose(cam[:3, :3], cam[:3, 3])
# ipv_draw_pose_3d()
vis.show()
# print('Done')