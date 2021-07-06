import os, sys, cv2, pickle
import numpy as np
from core_io.ply_io import load_pointcloud_from_ply
from data.ambi.read_helper import *

base_dir = '/Users/corsy/Downloads/AmbiguousData/books'
model_dir = os.path.join(base_dir,  'Models_0423_MC_GL_withBA_0.1_DV0.05_30', 'model000')

pts, pts_color = load_pointcloud_from_ply(os.path.join(model_dir, 'points_00021.ply'))

frame_list = read_image_list(os.path.join(base_dir, 'ImageList.txt'))
Es, Cs = read_poses(os.path.join(model_dir, 'bundle_00021.out'))

frame_list = [f.split('.jpg')[0][2:] for f in frame_list]

""" Gen matches
"""
matches = {}
for i in range(len(frame_list)):
    f_name = frame_list[i]
    x1_file_path = os.path.join(base_dir, f_name + '.sift')
    x1_loc, x1_desc = read_features_from_file(x1_file_path)

    for j in range(len(frame_list)):
        if i == j:
            continue
        f2_name = frame_list[j]
        x2_file_path = os.path.join(base_dir, f2_name + '.sift')
        x2_loc, x2_desc = read_features_from_file(x2_file_path)
        match = match_descriptors(x1_desc, x2_desc)
        print('[%d to %d] %s to %s, matches=%d' % (i, j, f_name, f2_name, match.shape[0]))
        matches['%d-%d' % (i, j)] = match

with open(os.path.join(base_dir, 'matches.bin'), 'wb') as f:
    pickle.dump(matches, f)