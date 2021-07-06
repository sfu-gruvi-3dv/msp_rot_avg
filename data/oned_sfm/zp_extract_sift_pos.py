import skimage.io
import numpy as np
import os, glob, pickle, sys

def read_features_from_file(filename):
    """ read feature properties and return in matrix form"""
    # f = loadtxt(filename)
    # return f[:,:4],f[:,4:] # feature locations, descriptors
    f = skimage.io.load_sift(filename)
    loc = [[x[1], x[0], x[2], x[3]] for x in f]
    des = [x[4] for x in f]
    loc = np.asarray(loc)
    des = np.asarray(des)
    return loc, des  # feature locations, descriptors

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('python3 zp_extract_sift_pos.py <sift_dir> <output_file_path>.bin')

    input_sift_dir = sys.argv[1]
    output_sift_pos_file_path = sys.argv[2]
    sift_files = glob.glob(os.path.join(input_sift_dir, '*.sift'))

    sift_pos_dict = dict()
    total_sift_files = len(sift_files)
    for i, sift_file_path in enumerate(sift_files):
        loc, _ = read_features_from_file(sift_file_path)
        sift_pos_dict[sift_file_path] = loc

        if i % 10 == 0:
            print('[%d/%d] finished' % (i, total_sift_files))

    with open(output_sift_pos_file_path, 'wb') as f:
        pickle.dump(sift_pos_dict, f)