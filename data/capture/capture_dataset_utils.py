import os
import sys
import cv2
import pickle
import numpy as np
from torch import matmul
import tqdm
import torch
from core_io.lmdb_reader import LMDBModel
from torch.utils.data import Dataset
from core_io.ply_io import load_pointcloud_from_ply
from data.ambi.read_helper import *
from data.ambi.ambi_parser import *
from evaluator.basic_metric import rel_distance, rel_rot_angle
from sampling import *
from tqdm import tqdm
from core_dl.expr_ctx import ExprCtx
from data.oned_sfm.SceneModel import ODSceneModel
import torchvision.transforms as transforms
import core_3dv.camera_operator as cam_opt
from data.util.nvm_reader import *


def drawSamplingOutMat(imgs, cam_Es, cam_Cs, out_graph_mat):
    cam_Es = cam_Es.cpu().detach().numpy()
    cam_Cs = cam_Cs.cpu().detach().numpy()
    out_graph_mat = out_graph_mat.cpu().detach().numpy()

    plt.figure(0, figsize=(10, 10))
    ax = plt.gca()

    count_0 = 0
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if out_graph_mat[i][j] == 1:
                a = (cam_Cs[i, 0], cam_Cs[j, 0])
                b = (cam_Cs[i, 1], cam_Cs[j, 1])
                ax.plot(a, b, '-', color='blue', linewidth=0.5)
            elif out_graph_mat[i][j] == -1:
                a = (cam_Cs[i, 0], cam_Cs[j, 0])
                b = (cam_Cs[i, 1], cam_Cs[j, 1])
                ax.plot(a, b, '-', color='green', linewidth=0.1)
            else:
                count_0 = count_0 + 1
    ax.scatter(cam_Cs[:, 0], cam_Cs[:, 1], color='r', s=80)
    plt.show()


# %%
def imgs2batchsamesize(imgs):
    size_imgs = {}
    batch_imgs = []
    batch_idx = []
    for idx, img in enumerate(imgs):
        if img.shape not in size_imgs:
            size_imgs[img.shape] = []
        size_imgs[img.shape].append((idx, img))
    for img_list in size_imgs.values():
        if len(img_list) == 1:
            batch_imgs.append(img_list[0][1])
            batch_idx.append([img_list[0][0]])
        else:
            imgs_list = [x[1] for x in img_list]
            idx_list = [x[0] for x in img_list]
            batch_idx.append(idx_list)
            batch_imgs.append(torch.cat(imgs_list))
    return batch_idx, batch_imgs


def read_manual_data(file_path, num_camera=None):
    res = np.loadtxt(file_path)
    if num_camera is None:
        nC = int(np.max(res[:, 1])) + 1
    else:
        nC = num_camera
    inoutMat = np.zeros((nC, nC))
    for i in range(res.shape[0]):
        h = res[i]
        out = res[i][2]
        from_ = int(h[0])
        to_ = int(h[1])
        inoutMat[from_, to_] = int(out)
        inoutMat[to_, from_] = int(out)
    return inoutMat
