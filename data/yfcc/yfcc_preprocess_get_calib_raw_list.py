from utils.nvm_reader import *
import sys, os, cv2
import numpy as np

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('usage: python yfcc_preprocess_get_calib_list.py <yfcc_scene_dir>')

    yfcc_scene_dir = sys.argv[1]
    nvm_file_path = os.path.join(yfcc_scene_dir, 'model.nvm')
    img_dim_def_path = os.path.join(yfcc_scene_dir, 'images.dim')

    # create dim dict
    img_dim_dict = {}
    with open(img_dim_def_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            else:
                tokens = line.split(' ')
                img_name = tokens[0]
                width = int(tokens[1])
                height = int(tokens[2])
                img_dim_dict[img_name] = (height, width)

    model = readNvm(nvm_file_path)
    assert len(model.modelArray) == 1
    cams = model.modelArray[0].cameraArray
    img_list_txt = []
    calibrate_txt = []
    flip_list_txt = []
    ignored = 0

    for cam in cams:
        rot_q = np.asarray([float(q_i) for q_i in cam.quaternionArray])
        cam_center = np.asarray([float(c) for c in cam.cameraCenter])
        file_name = cam.fileName
        focal = float(cam.focalLength)
        dim = img_dim_dict[file_name]
        im_h, im_w = dim

        img_file_path = os.path.join(yfcc_scene_dir, file_name)

        img_list_txt.append(file_name)
        calibrate_txt.append((focal, im_w / 2, im_h / 2, im_w, im_h))

    # dump imge list
    with open(os.path.join(yfcc_scene_dir, 'raw_images.list.txt'), 'w') as f:
        for img in img_list_txt:
            f.write('%s\n' % img)

    # dump calibration list
    with open(os.path.join(yfcc_scene_dir, 'raw_calibration.txt'), 'w') as f:
        for calib in calibrate_txt:
            for i in range(5):
                if i != 4:
                    f.write('%f ' % calib[i])
                elif i == 4:
                    f.write('%f\n' % calib[i])