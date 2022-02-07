import data.oned_sfm.sfminit as sfminit
import numpy as np
import torch
import torchvision.transforms as transforms
import copy, sys, argparse, os, glob, pickle, cv2

data_basedir = '/mnt/Exp_2/1dsfm/data/datasets/Alamo'
img_dir = '/mnt/Exp_2/1dsfm/imgs/Alamo'
iccv15_path = '/mnt/Tango/pg/ICCV15/Alamo/bundle_00574.out'
diff_cam_key_flag = False

# load list
from data.oned_sfm.read_util import read_list

input = {'EGs': os.path.join(data_basedir, 'EGs.txt'),
         'ccs': os.path.join(data_basedir, 'cc.txt'),
         'tracks': os.path.join(data_basedir, 'tracks.txt'),
         'coords': os.path.join(data_basedir, 'coords.txt'),
         'gt_soln': os.path.join(data_basedir, 'gt_bundle.out')}

cc = np.loadtxt(input['ccs'])
frame_list = read_list(os.path.join(data_basedir, 'list.txt'))
gt_bundle = sfminit.Bundle.from_file(input['gt_soln'])
coords = sfminit.Coords.from_file(input['coords'])
feat_data = coords.data
iccv_bundle = sfminit.Bundle.from_file(iccv15_path)
models = sfminit.ModelList.from_EG_file(input['EGs'])

n_CC = cc.shape[0]
cc_reorder_map = dict()
for i in range(n_CC):
    camera_id = int(cc[i])
    cc_reorder_map[camera_id] = i

# netvlad
from vlad_encoder import VLADEncoder
vlad_pretrain_path = 'cache/netvlad_vgg16.tar'
vlad = VLADEncoder(checkpoint_path=vlad_pretrain_path, train_feat_extractor=False, train_vlad=False)
with torch.cuda.device(0):
    vlad.cuda()

transform_func = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# cache the vlad_feats
with torch.cuda.device(0):
    for i in range(n_CC):
        c = cc[i]
        frame = frame_list[i]
        img = cv2.imread(os.path.join(img_dir, frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = transform_func(img).unsqueeze(0).cuda()

    #     vlad_feat = vlad.forward(img.cuda())
