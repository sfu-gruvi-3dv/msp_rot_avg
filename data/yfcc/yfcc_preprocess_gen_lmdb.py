from core_io.lmdb_writer import LMDBWriter
import os, sys, glob, pickle
from tqdm import tqdm
import cv2
import numpy as np
import socket, shutil, os, pickle
server_name = socket.gethostname()

if server_name == 'cs-gruvi-24-cmpt-sfu-ca':
    captured_img_dir = '/mnt/Exp_5/yfcc100/set/'  # '/local-scratch6/1dsfm/images'
    output_lmdb_path = '/mnt/Exp_5/yfcc100/set/cache.lmdb'  # '/local-scratch7/1dsfm_img.lmdb'
    output_img_meta_path = '/mnt/Exp_5/yfcc100/set/cache_meta.bin'
elif server_name == 'cs-guv-gpu02':
    captured_img_dir = '/local-scratch6/yfcc100/set/' #'/local-scratch6/1dsfm/images'
    output_lmdb_path =  '/local-scratch6/yfcc100/set/cache.lmdb' # '/local-scratch7/1dsfm_img.lmdb'
    output_img_meta_path = '/local-scratch6/yfcc100/set/cache_meta.bin'

resize_max_dim = 640

lmdb_db = LMDBWriter(output_lmdb_path)
img_meta_dict = dict()
if os.path.exists(output_img_meta_path):
    with open(output_img_meta_path, 'rb') as f:
        img_meta_dict = pickle.load(f)

scene_list = glob.glob(os.path.join(captured_img_dir, '*'))
for scene_dir in scene_list:
    if os.path.isdir(scene_dir):
        scene_name = scene_dir.split('/')[-1]
        if scene_name == 'cache.lmdb':
            continue

        image_list = glob.glob(os.path.join(scene_dir, '*.jpg'))
        image_list += glob.glob(os.path.join(scene_dir, '*.png'))
        print('[Save %s]' % scene_name)

        for img_path in tqdm(image_list):
            img_name = img_path.split('/')[-1].strip()
            img_name = img_name.split('.')[0].strip()       # remove .jpg .png
            key = scene_name + '/' + img_name + '.jpg'

            # if key in img_meta_dict:
            #     continue

            # read image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            res_h, res_w = img.shape[:2]
            max_dim = res_h if res_h > res_w else res_w
            down_factor = float(resize_max_dim) / float(max_dim)
            img = cv2.resize(img, dsize=(int(res_w * down_factor), int(res_h * down_factor)))
            # img = img.astype(np.float32) / 255.0

            lmdb_h, lmdb_w = img.shape[:2]

            lmdb_db.write_array(key, img.astype(np.uint8))

            # save img meta information
            img_meta_dict[key] = {'dim': (h, w), 'lmdb_dim': (lmdb_h, lmdb_w)}

lmdb_db.close_session()
with open(output_img_meta_path, 'wb') as f:
    pickle.dump(img_meta_dict, f)