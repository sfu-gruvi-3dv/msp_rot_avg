import json, os, cv2, pickle
import numpy as np

import matplotlib.pyplot as plt

from data.ambi.ambi_parser import *
from data.ambi.ambi_dataset import *
from data.ambi.read_helper import *

import scipy.linalg as linalg

def drawlines(img1, img2, lines, pts1, pts2, draw_lines=False):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''

    if len(img1.shape) == 2:  # grayscale input
        print("Grayscale Input (drawLines)")
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    elif len(img1.shape) == 3:  # RGB
        print("Color Input (drawLines)")

    else:
        print(len(img1.shape))

    r, c, ch = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        if draw_lines:
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        pt1 = pt1.astype(np.int32)
        pt2 = pt2.astype(np.int32)

        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)

    return img1, img2


def parserAmbiLoadFromJson(filename):
    """
    structure:
        baseDir: the base path of dataset
        names: list, the name list of dataset
        dataset: dict, [name]
            dataDir: the dataset path
            bundle_file: the bundle file path
            imageList:
            EGs
            calibration
            listFocal
            matches
    """
    if os.path.exists(filename) is False:
        print("File does not exist")
        return
    with open(filename, "r") as f:
        args = json.load(f)
    
    return args

class AmbiVerifyDataset:
    def __init__(self, base_dir, data_dict):
        # set the init parameters
        self.dataset_base_dir = base_dir
        self.dataset_dict = data_dict
        self.dataset_dir = os.path.join(self.dataset_base_dir, self.dataset_dict["dataDir"])

        # read EGs from file
        pairs, R, t, match_number, matches = readAmbiEGWithMatch(
            os.path.join(self.dataset_dir, self.dataset_dict["EGs"])
        )

        self.EGs_pairs = pairs
        self.EGs_R = R
        self.EGs_t = t 
        self.EGs_match_number = match_number
        self.EGs_matches = matches

        # read bundle file
        nC, nP, cfks, cRs, cts, pPs, pCs, pVNs, pVs, pKs, pVLs = parserBundle(
            os.path.join(self.dataset_dir, self.dataset_dict["bundle_file"])
        )

        self.bundle_nC = nC
        self.bundle_nP = nP
        self.bundle_cfks = cfks
        self.cRs = cRs
        self.cts = cts
        self.pPs = pPs
        self.pVNs = pVNs
        self.pVs = pVs
        self.pKs = pKs
        self.pVLs = pVLs

        # read Image List
        self.image_list = read_image_list(
            os.path.join(self.dataset_dir, self.dataset_dict["imageList"])
        )
        self.images = []
        for name in self.image_list:
            print(name)
            self.images.append(cv2.imread(
            os.path.join(self.dataset_dir,name)
        ))

        # read calibration
        self.calibration = read_calibration(
            os.path.join(self.dataset_dir, self.dataset_dict["calibration"])
        )

        # TODO: read features from the file
        # self.features = read_features_from_file(
        #     os.path.join(self.dataset_dir, self.data_dict[""])
        # )

    def showEGsPairs(self):
        # plt.style.use('seaborn-whitegrid')
        plt.figure(0, figsize=(10, 10))
        ax = plt.gca()

        camera_C = np.asarray(self.cts)
        ax.scatter(camera_C[:,0], camera_C[:,1], color='r', s = 50)

        for (i,j) in self.EGs_pairs:
            a = (camera_C[i, 0], camera_C[j, 0])
            b = (camera_C[i, 1], camera_C[j, 1])
            ax.plot(a, b, '-', color='blue', linewidth=0.5)
        
        plt.show()

    def showAllPairsWithEGsAndMark(self):
        #TODO: show all pairs with egs
        pass
    
    def showEGPairWithEG(self, index):
        #TODO: show pair with EG:
        # (left, right) = self.EGs_pairs[index]
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(cv2.imread(
        #     os.path.join(self.dataset_dir, self.read_image_list[left])
        # ))
        # ax[1].imshow(cv2.imread(
        #     os.path.join(self.dataset_dir, self.read_image_list[right])
        # ))

        # fig, ax = plt.subplots(1,2) 
        # pt1, pt2 = parserRandomPickMatches2Loc(self.EGS_matches[index], num_feat)
        # pt1 = np.asarray(pt1)
        # pt2 = np.asarray(pt2)

        # Kleft = self.calibration[0][left]
        # Kright = self.calibration[0][right]
        # F = np.matmul(linalg.inv(Kleft),)

        (j,i) = self.EGs_pairs[index]
        fig,ax = plt.subplots(1,2,figsize=(20, 30))

        ax[0].imshow(self.images[i])
        ax[1].imshow(self.images[j])
        # plt.show()

        plt.figure(0, figsize=(10, 10))
        ax = plt.gca()

        camera_C = np.asarray(self.cts)
        ax.scatter(camera_C[:,0], camera_C[:,1], color='r', s = 50)

        
        a = (camera_C[i, 0], camera_C[j, 0])
        b = (camera_C[i, 1], camera_C[j, 1])
        ax.plot(a, b, '-', color='blue', linewidth=0.5)

        # newt = np.matmul(-self.EGs_R[index].T, camera_C[i,:].reshape([3,1]))
        # newt = np.matmul(self.EGs_R[index],self.EGs_t[index].reshape([3,1]))
        newt = self.EGs_t[index]
        newt= newt / linalg.norm(newt)
        temp = camera_C[i,:] + newt

        a = (camera_C[i, 0], temp[0])
        b = (camera_C[i, 1], temp[1])
        print(a)
        print(b)
        ax.plot(a,b, '-', color='green', linewidth=0.5)
        ax.scatter(temp[0],temp[1], color='yellow', s=100)


    def showBundlePairs(self):
        # plt.style.use('seaborn-whitegrid')
        plt.figure(0, figsize=(10, 10))
        ax = plt.gca()

        camera_C = np.asarray(self.cts)
        ax.scatter(camera_C[:,0], camera_C[:,1], color='r', s = 50)

        for (i,j) in self.EGs_pairs:
            a = (camera_C[i, 0], camera_C[j, 0])
            b = (camera_C[i, 1], camera_C[j, 1])
            ax.plot(a, b, '-', color='blue', linewidth=0.1)
        
        plt.show()

    def showImageByID(self, imageid):
        fig,ax = plt.subplots()
        ax.imshow(cv2.imread(
            os.path.join(self.dataset_dir,self.image_list[imageid])
        ))
        plt.show()

    def showCameraPosition(self):
        # plt.style.use('seaborn-whitegrid')
        plt.figure(0, figsize=(10, 10))
        ax = plt.gca()

        camera_C = np.asarray(self.cts)
        ax.scatter(camera_C[:,0], camera_C[:,1], color='r', s = 50)

        plt.show()

    def verifyAll(self):
        # TODO: verify all
        pass

    def verifyBySampling(self):
        # TODO: verify by sampling
        pass
        
