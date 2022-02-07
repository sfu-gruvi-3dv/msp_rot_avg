import numpy as np
import os, sys, glob, cv2
from tqdm import tqdm
from vlad_encoder import *
from data.ambi.read_helper import *

base_dir = '/mnt/Tango/pg/Ambi'
seq_name = ['cup', 'books', 'cereal', 'desk', 'oats', 'street']
frame_list_filename = "ImageList.txt"
output_frame_feat = "ImageFeat.txt"



vlad_db = VLADEncoder(checkpoint_path='./cache/netvlad_vgg16.tar', dev_id=0)




# for train_frame in tqdm(train_seq.frames, desc='add train samples'):
#     img, depth, _, _, _ = get_frame_scene_coord(base_dir, train_frame, resize_hw=(192, 256))
#     vlad_db.add_sample(img, train_frame)
# vlad_db.load(os.path.join(base_dir, 'train_vlad_feats.bin'))
# vlad_db.dump(os.path.join(base_dir, 'train_vlad_feats.bin'))
"""
for test_frame in test_seq.frames:
    img, depth, _, _, _ = get_frame_scene_coord(base_dir, test_frame, resize_hw=(192, 256))
    res = vlad_db.find_close_samples(img)
    print(res)
"""

img = torch.randn([5,30,30,3])
output = vlad_db.forward(img)

