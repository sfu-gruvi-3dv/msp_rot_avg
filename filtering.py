import os, sys, cv2, pickle
import numpy as np
from core_io.ply_io import load_pointcloud_from_ply
from data.ambi.read_helper import *
from evaluator.basic_metric import rel_distance, rel_rot_angle

base_dir = '/Users/corsy/Downloads/AmbiguousData/books'
model_dir = os.path.join(base_dir,  'Models_0423_MC_GL_withBA_0.1_DV0.05_30', 'model000')

pts, pts_color = load_pointcloud_from_ply(os.path.join(model_dir, 'points_00021.ply'))

frame_list = read_image_list(os.path.join(base_dir, 'ImageList.txt'))
frame_list = [f.split('.jpg')[0][2:] for f in frame_list]

Es, Cs = read_poses(os.path.join(model_dir, 'bundle_00021.out'))

# load matches
with open(os.path.join(base_dir, 'matches.bin'), 'rb') as f:
    matches = pickle.load(f)

