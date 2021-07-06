import sys; import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import data.oned_sfm.sfminit as sfminit

import argparse
import numpy as np
import copy
import sys

data_dir = '/Volumes/Resources/1DSFM/data/datasets/Gendarmenmarkt'
input = {'EGs': os.path.join(data_dir, 'EGs.txt'),
         'ccs': os.path.join(data_dir, 'cc.txt'),
         'tracks': os.path.join(data_dir, 'tracks.txt'),
         'coords': os.path.join(data_dir, 'coords.txt'),
         'gt_soln': os.path.join(data_dir, 'gt_bundle.out')}

tracks = sfminit.Tracks.from_file(input['tracks'])
coords = sfminit.Coords.from_file(input['coords'])
models = sfminit.ModelList.from_EG_file(input['EGs'])
gt_bundle = sfminit.Bundle.from_file(input['gt_soln'])

gt_bundle2 = sfminit.Bundle.from_file('/Users/corsy/Downloads/ICCV15/Gendarmenmarkt/bundle_00635.out')

# filtering the invalid cameras
invalid_cam_flag = dict()
for i, cam in enumerate(gt_bundle2.cameras):
    if cam.f == 0:
        invalid_cam_flag[i] = True

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

print('Test')
