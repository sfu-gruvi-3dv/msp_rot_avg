import numpy as np
import scipy.linalg as linalg

from data.ambi.read_helper import *
from data.ambi.ambi_parser import *

from core_3dv.essential_mat import *
import core_3dv.camera_operator as cam_opt


"""
nC: # cameras
mat: adjacency matrix
"""


def rel_t_err(rel1, rel2):

    R1, R2 = rel1[:3, :3], rel2[:3, :3]
    t1, t2 = rel1[:3, 3], rel2[:3, 3]

    t1 = t1 / linalg.norm(t1)
    t2 = t2 / linalg.norm(t2)

    dot = np.dot(t1, t2)
    dot = np.clip(dot, -1, 1)
    t_err = np.arccos(dot)
    return t_err


class InOutlierGenerator:
    def __init__(self, bundle_filename, egs_filename):
        self.bundle_filename = bundle_filename
        self.egs_filename = egs_filename
        [pairs, pred_rel_R, pred_rel_t, pred_match_number, pred_matches] = \
            readAmbiEGWithMatch(egs_filename)

        self.pairs = pairs
        self.pred_rel_R = pred_rel_R
        self.pred_rel_t = pred_rel_t
        self.pred_match_number = pred_match_number
        self.pred_matches = pred_matches

        self.Es, self.Cs = read_poses(self.bundle_filename)

        nC, nP, cfks, cRs, cts, pPs, pCs, pVNs, pVs, pKs, pVLs =\
            parserBundle(self.bundle_filename)
        self.nC = nC
        self.nP = nP
        self.cfks = cfks
        self.cRs = cRs
        self.cts = cts
        self.pPs = pPs
        self.pCs = pCs
        self.pVNs = pVNs
        self.pVs = pVs
        self.pKs = pKs
        self.pVLs = pVLs

        cpn, cpl, mvis, mnvis = buildCovisableMatrix(nC, nP, pVNs, pVs)

        self.cpn = cpn
        self.cpl = cpl
        self.mvis = mvis
        self.mnvis = mnvis

        self.inlier_R_threshold = 5
        self.inlier_t_threshold = 25
        self.outlier_R_threshold = 90
        self.outlier_t_threshold = 2
        self.inlier_covis_threshold = 50

    def setThreshold(self, irt, itt, ort, ott, ict):
        self.inlier_R_threshold = irt
        self.inlier_t_threshold = itt
        self.outlier_R_threshold = ort
        self.outlier_t_threshold = ott
        self.inlier_covis_threshold = ict

    def buildInOutlierMat(self):
        R_errs = []
        t_errs = []
        fl = self.nC
        fl_st = 0
        self.inoutMat = np.zeros([fl, fl])
        for i in range(fl_st, fl):
            for j in range(fl_st, fl):
                if i == j:
                    self.inoutMat[i][j] = inf
                    continue
                elif i > j:
                    continue

                ti = i
                tj = j
                if (ti, tj) not in self.pairs:
                    ti, tj = tj, ti
                if (ti, tj) not in self.pairs:
                    self.inoutMat[ti][tj] = self.inoutMat[tj][ti] = inf
                    continue

                idxinpairs = self.pairs.index((ti, tj))

                Eij_R_pred = self.pred_rel_R[idxinpairs]
                Eij_t_pred = self.pred_rel_t[idxinpairs]

                Eij_Rt_pred = np.hstack((Eij_R_pred, Eij_t_pred.reshape(3, 1)))

                rel_Rt_gt = cam_opt.relateive_pose(
                    self.Es[ti][:3, :3], self.Es[ti][:3, 3], self.Es[tj][:3, :3], self.Es[tj][:3, 3])
                rel_R_gt = rel_Rt_gt[:3, :3]
                rel_t_gt = rel_Rt_gt[:3, 3]
                rel_t_gt = rel_t_gt / linalg.norm(rel_t_gt)

                c1 = self.Cs[i]
                c2 = self.Cs[j]
                rel_R = np.matmul(Eij_Rt_pred[:3, :3], rel_Rt_gt[:3, :3].T)
                R_err = np.rad2deg(np.arccos((np.trace(rel_R) - 1) / 2))
                R_errs.append(R_err)

                t_err = np.rad2deg(rel_t_err(Eij_Rt_pred, rel_Rt_gt))
                t_errs.append(t_err)
                if self.mvis[ti][tj] > self.inlier_covis_threshold:
                    self.inoutMat[tj][ti] = self.inoutMat[ti][tj] = 1
                elif R_err > self.outlier_R_threshold or t_err > self.outlier_t_threshold:
                    self.inoutMat[ti][tj] = self.inoutMat[tj][ti] = -1
                else:
                    self.inoutMat[ti][tj] = self.inoutMat[tj][ti] = 0


class SamplingGenerator:

    def __init__(self, nC, mat):
        self.n = nC
        self.mat = mat

        self.sampling_number = 10
        self.sampling_size = 10
        self.seed = 0
        np.random.seed(self.seed)

    def pickOne(self, get_max_node=True):

        if get_max_node:
            if len(self.candidate_pool) == 0:
                while True:
                    idx = np.random.choice(np.arange(0, self.n, 1))
                    if self.pick_vis[idx] == 0:
                        self.candidate_pool.append(idx)
                        break
        if len(self.candidate_pool) == 0:
            return -1
        while True:
            pick_node = np.random.choice(self.candidate_pool)
            if self.pick_vis[pick_node] == 0:
                return pick_node

    def addNeighbor(self, in_node, use_undefine=False):
        for idx in range(self.n):
            if (self.mat[in_node][idx] == -1 or self.mat[in_node][idx] == 1 or (use_undefine and self.mat[in_node][idx] == 0)) and self.pick_vis[idx] == 0:
                self.candidate_pool.append(idx)

        self.candidate_pool = list(set(self.candidate_pool))

    def generationOne(self, bi=False, use_undefine=False, get_max_node=True):
        self.candidate_pool = []
        self.pick_vis = [0 for x in range(self.n)]
        self.pick_node = []
        self.pick_edge = []
        self.pick_edge_label = []
        self.pick_node_num = 0
        self.pick_edge_num = 0
        valid_node = []
        for i in range(self.n):
            if np.sum(self.mat[i, :]) != 0:
                valid_node.append(i)
        self.candidate_pool.append(np.random.choice(valid_node))

        while self.pick_node_num < self.sampling_size:
            in_node = self.pickOne(get_max_node)
            if in_node == -1:
                break
            self.pick_vis[in_node] = 1
            self.pick_node.append(in_node)

            self.candidate_pool.remove(in_node)
            self.pick_node_num = self.pick_node_num + 1

            self.addNeighbor(in_node, use_undefine)

            if self.pick_node_num == self.n:
                break

        for i in range(self.pick_node_num):
            for j in range(self.pick_node_num):

                ti = self.pick_node[i]
                tj = self.pick_node[j]

                if i > j or i == j:
                    continue

                if self.mat[ti][tj] == -1 or self.mat[tj][ti] == 1 or (use_undefine and self.mat[ti][tj] == 0):
                    self.pick_edge.append((ti, tj))
                    self.pick_edge_label.append(self.mat[ti][tj])
                    self.pick_edge_num = self.pick_edge_num + 1

        return self.pick_node_num, self.pick_node, self.pick_edge_num,\
            self.pick_edge, self.pick_edge_label

    def generation(self, bi=False, use_undefine=False, get_max_node=True):
        self.sampling_node = []
        self.sampling_node_num = []
        self.sampling_edge = []
        self.sampling_edge_num = []
        self.sampling_edge_label = []

        for _ in range(self.sampling_number):
            while True:
                pick_node_num, pick_node, pick_edge_num, pick_edge, pick_edge_label = \
                    self.generationOne(bi, use_undefine, get_max_node)
                if pick_edge_label.count(1) != 0:
                    self.sampling_node.append(pick_node)  # list[list]
                    self.sampling_node_num.append(pick_node_num)  # list[num]
                    self.sampling_edge.append(pick_edge)  # list[list [tuple]]
                    self.sampling_edge_num.append(pick_edge_num)  # list[num]
                    self.sampling_edge_label.append(
                        pick_edge_label)  # list[list]
                    break

        return True
    
    def appendAndGeneration(self, num, bi=False, use_undefine=False, get_max_node=True):
        self.sampling_number += num
        for _ in range(num):
            while True:
                pick_node_num, pick_node, pick_edge_num, pick_edge, pick_edge_label = \
                    self.generationOne(bi, use_undefine, get_max_node)
                if pick_edge_label.count(1) != 0:
                    self.sampling_node.append(pick_node)  # list[list]
                    self.sampling_node_num.append(pick_node_num)  # list[num]
                    self.sampling_edge.append(pick_edge)  # list[list [tuple]]
                    self.sampling_edge_num.append(pick_edge_num)  # list[num]
                    self.sampling_edge_label.append(
                        pick_edge_label)  # list[list]
                    break

        return True

    def countingNumberOfEdgeInSubgraph(self):
        counting = dict()
        for edges in self.sampling_edge:
            for edge in edges:
                u,v = edge[0], edge[1]
                if u > v:
                    u, v = v, u
                if (u, v) not in counting:
                    counting[(u, v)] = 0
                counting[(u,v)] += 1
        return counting
    
    def appendUntilnxEdges(self,all_edges, bi=False, use_undefine=False, get_max_node=True, times=2):
        while True:
            counting = self.countingNumberOfEdgeInSubgraph()
            flag = 0
            minn = 1e8
            for edge, val in all_edges.items():
                if edge not in counting or counting[edge] < times:
                    flag = 1
                    break
            if flag == 0:
                break
            self.appendAndGeneration(100,bi, use_undefine,get_max_node)
            print(len(self.sampling_edge))

    def setSeed(self, newSeed):
        self.seed = newSeed
        np.random.seed(self.seed)

    def setSamplingSize(self, sampling_size):
        self.sampling_size = sampling_size

    def setSamplingNumber(self, samplingNumber):
        self.sampling_number = samplingNumber
