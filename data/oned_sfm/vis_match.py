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

feats = coords.data[0].keys

print(models)
