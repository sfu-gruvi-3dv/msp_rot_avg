from PIL import Image
import os
from numpy import *
from pylab import *
import cv2
import skimage
import skimage.io
from skimage.feature import match_descriptors

def read_image_list(image_list_path):
    with open(image_list_path) as f:
        files = f.readlines()
        files = [item.strip().strip(".").strip("/").strip("\\") for item in files]
        return files

def read_poses(pose_file_path):

    with open(pose_file_path) as f:
        lines = f.readlines()

    num_frames = 0
    raw_data = []

    for i, line in enumerate(lines):

        if line.startswith('#'):
            continue

        vals = np.fromstring(line, dtype=np.float32, sep=" ")
        if i == 1:
            num_frames = int(vals[0])
        else:
            raw_data.append(vals)

        if len(raw_data) == num_frames * 5:
            break

    raw_data = np.asarray(raw_data)
    Es, Cs = [], []

    for i in range(num_frames):
        f = raw_data[i*5, :]
        R = raw_data[i*5+1:i*5+4, :]
        t = raw_data[i*5+4:i*5+5, :]

        E = np.zeros((3, 4))
        E[:, :3] = R
        E[:, 3] = t.ravel()

        c = np.matmul(-R.T, t.reshape(3, 1))

        Es.append(E)
        Cs.append(c.ravel())

    return Es, Cs

def read_calibration(calib_file_path):
    raw_calib = np.loadtxt(calib_file_path)
    f = raw_calib[:, 0]
    cx = raw_calib[:, 1]
    cy = raw_calib[:, 2]

    Ks = []
    for i in range(f.shape[0]):
        K = np.zeros((3, 3), dtype=np.float32)
        K[0, 0] = f[i]
        K[1, 1] = f[i]
        K[0, 2] = cx[i]
        K[1, 2] = cy[i]
        K[2, 2] = 1.0

        
        Ks.append(K)

    return Ks, raw_calib[:, 3:].astype(np.int32)

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

def write_features_to_file(filename, locs, desc):
    """ save feature location and descriptor to file"""
    savetxt(filename, hstack((locs, desc)))

def plot_correspondences(a_img, b_img, a_locs, b_locs, matches):
    plt.figure(0)
    ax = plt.gca()
    skimage.feature.plot_matches(ax, a_img, b_img, a_locs, b_locs, matches)
    ax.axis('off')
    # ax.set_title("A Image vs. B Image")
    plt.show()


if __name__ == "__main__":

    img_list = read_image_list('/Users/corsy/Downloads/AmbiguousData/oats/ImageList.txt')
    calib = read_calibration('/Users/corsy/Downloads/AmbiguousData/oats/calibration.txt')

    read_poses('/Users/corsy/Downloads/AmbiguousData/books/Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_init.out')

    x1_file_path = '/Users/corsy/Downloads/AmbiguousData/oats/P1010140.sift'
    x2_file_path = '/Users/corsy/Downloads/AmbiguousData/oats/P1010141.sift'

    x1_img = cv2.imread('/Users/corsy/Downloads/AmbiguousData/oats/P1010140.jpg')
    x2_img = cv2.imread('/Users/corsy/Downloads/AmbiguousData/oats/P1010141.jpg')
    x1_loc, x1_desc = read_features_from_file(x1_file_path)
    x2_loc, x2_desc = read_features_from_file(x2_file_path)
    matches = match_descriptors(x1_desc, x2_desc)

    plot_correspondences(x1_img, x2_img, x1_loc, x2_loc, matches)
