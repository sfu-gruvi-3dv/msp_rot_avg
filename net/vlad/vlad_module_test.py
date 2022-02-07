import numpy as np
import os, sys, glob, cv2
from tqdm import tqdm
from frame_seq_data import FrameSeqData
from relocal_data.cambridge.read_util import get_frame_scene_coord

base_dir = '/mnt/Tango/pg/Ambi'
seq_name = ['cup', 'books', 'cereal', 'desk', 'oats', 'street']
frame_list_filename = "ImageList.txt"

train_seq = FrameSeqData(os.path.join(base_dir, seq_name, 'seq.json'))
test_seq = FrameSeqData(os.path.join(base_dir, seq_name, 'seq_test.json'))

from relocal.vlad_encoder import VLADEncoder

vlad_db = VLADEncoder(checkpoint_path='data/netvlad_vgg16.tar', dev_id=0)

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

for dataset in seq_name:
    image_list = read_image_list(os.path.join(base_dir, dataset, frame_list_filename))
    
