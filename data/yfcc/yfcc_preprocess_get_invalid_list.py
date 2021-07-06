from data.util.nvm_reader import *
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

    flip_list_txt = []
    invalid_list = []
    ignored = 0

    for cam in cams:
        rot_q = np.asarray([float(q_i) for q_i in cam.quaternionArray])
        cam_center = np.asarray([float(c) for c in cam.cameraCenter])
        file_name = cam.fileName
        focal = float(cam.focalLength)
        dim = img_dim_dict[file_name]

        img_file_path = os.path.join(yfcc_scene_dir, file_name)
        if os.path.exists(img_file_path):

            # get original image size
            # im = Image.open(img_file_path)
            img = cv2.imread(img_file_path)
            im_h, im_w = img.shape[:2]
            # im_w, im_h = im.size

            scale_h = im_h / dim[0]
            scale_w = im_w / dim[1]

            if abs(scale_h / scale_w - 1) > 1e-2:

                scale_h_ = im_w / dim[0]
                scale_w_ = im_h / dim[1]
                if abs(scale_h_ / scale_w_ - 1) < 1e-2:
                    flip_list_txt.append(file_name)
                    continue    # just ignore those cases for now
                else:
                    invalid_list.append(file_name)
                    # ignored += 1
                    # print('File %s will be ignored' % img_file_path)
                    # raise Exception('The original image aspect ratio doesnt match the one defined in images.dim')

        else:
            invalid_list.append(file_name)

    # dump invalid list
    with open(os.path.join(yfcc_scene_dir, 'invalid_images.list.txt'), 'w') as f:
        for img in invalid_list:
            f.write('%s\n' % img)

    # print flip images:
    with open(os.path.join(yfcc_scene_dir, 'todo_flip_img.txt'), 'w') as f:
        for img in flip_list_txt:
            f.write('%s\n' % img)
            print(img)