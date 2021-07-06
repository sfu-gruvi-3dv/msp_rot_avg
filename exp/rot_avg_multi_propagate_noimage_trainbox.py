# import os, shutil, warnings, cv2, sys
import numpy as np
import torch
from torch import acos
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader
from core_dl.train_params import TrainParameters
from core_dl.base_train_box import BaseTrainBox
from core_dl.torch_vision_ext import *
from torchvision.utils import make_grid
from core_dl.module_util import load_checkpoints, save_checkpoint, load_state_dict_by_key
import core_3dv.camera_operator_gpu as cam_opt_gpu
from evaluator.spt_metric import spt_statsitic_view
from tqdm import tqdm
import net.FineNet as FN
from net.GPoolCoreLayer import *
from net.dif_multi_spt_builder import SptPropagate
from net.FineNet import Net_outlier_det
import random
from data.edge_filtering import filtering

# dataset utils
from data.ambi.ambi_dataset import imgs2batchsamesize
from net.adjmat2dgl_graph import build_graph, adjmat2graph, gather_edge_feat, gather_edge_label, inv_adjmat, \
    gather_edge_feat2
# network
from vlad_encoder import VLADEncoder
from net.gat_net import MultiGATBaseConvs, EdgeClassifier, EdgeClassifier2
from net.mlp import SharedMLP
from scipy.cluster.vq import kmeans2
import torch.nn.functional as F
import graph_utils.utils as graph_util
from core_3dv.quaternion import *
import torchgeometry as tgm
import net.rot_avg_layers as r_opt
from evaluator.basic_metric import rel_R_deg
from torch.autograd.variable import Variable

def cluster_pool(pts1, feat_array, k):
    pts1 = pts1.squeeze(0).detach().cpu().numpy()
    centroids, labels = kmeans2(pts1, k, minit='points')
    feats = []
    for i in range(k):
        if (labels == i).sum() > 0:
            label_idx = np.argwhere(labels == i)
            label_idx = torch.from_numpy(label_idx).long().to(feat_array.device).view(-1)

            feat = torch.index_select(feat_array, 2, label_idx)
            feat = torch.max_pool1d(feat, kernel_size=feat.shape[-1])
            feats.append(feat)
        else:
            feats.append(torch.zeros((1, feat_array.shape[1], 1), device=feat_array.device))  # zero cluster
    feats = torch.cat(feats, dim=-1)
    return feats

def edge_model(x, edge_index):
    row, col = edge_index
    q_ij = qmul(x[col], inv_q(x[row]))
    return q_ij

def edge_model_batch(x, edge_index):
    """
    Args:
        x: pose, dim: (N, K, 4)
        edge_index: (2, E)
    Returns:
    """
    row, col = edge_index
    N, K = x.shape[:2]
    E = edge_index.shape[1]
    from_ = x[row, :, :].view(E*K, 4)
    to_ = x[col, :, :].view(E*K, 4)
    q_ij = qmul(to_, inv_q(from_)).view(E, K, 4)
    return q_ij

def huber_l1_loss(input, beta, alpha=0.05):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])
    nn = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
    nn = nn * beta

    cond = nn < alpha
    loss = torch.where(cond, 0.5 * nn ** 2 / alpha, nn - 0.5 * alpha)
    # if size_average:
    #     return loss.mean()
    # return loss.sum()
    return loss

class LocalGlobalVLADTrainBox(BaseTrainBox):

    def __init__(self, train_params: TrainParameters,
                 log_dir=None,
                 ckpt_path_dict=None,  # {'vlad': vlad_path, 'ckpt': ckpt_path (optional)}
                 top_k=6
                 ):

        assert ckpt_path_dict is not None and 'vlad' in ckpt_path_dict  # 'vlad' is needed to initialize
        assert len(train_params.DEV_IDS) >= 2  # require at least 2 gpus to train
        self.vlad_init_path = ckpt_path_dict['vlad']
        self.finenet_init_path = ckpt_path_dict['finenet'] if 'finenet' in ckpt_path_dict else None
        self.top_k = top_k
        self.use_vae = False
        if self.use_vae:
            print('Use VAE')
        super(LocalGlobalVLADTrainBox, self).__init__(train_params, log_dir,
                                                      checkpoint_path=ckpt_path_dict[
                                                          'ckpt'] if 'ckpt' in ckpt_path_dict else None,
                                                      comment_msg=train_params.NAME_TAG,
                                                      load_optimizer=False)

        # torch.manual_seed(1000)
        print('[Trainbox] Use Rel and Abs for all propagate')

    """ Network and Training Configuration ------------------------------------------------------------------------------
    """

    def _set_loss_func(self):
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _set_optimizer(self):
        # config the optimizer
        super(LocalGlobalVLADTrainBox, self)._set_optimizer()
        self.optimizer = torch.optim.Adam([
            {'params': self.appear_w.parameters(), 'lr': 1.0e-4},
            {'params': self.rot_avg_finenet.parameters(), 'lr': 1.0e-4}
        ], lr=self.train_params.START_LR)

    def _set_network(self):

        # config the network structure for training
        super(LocalGlobalVLADTrainBox, self)._set_network()
        with torch.cuda.device(self.dev_ids[0]):
            if not self.use_vae:
                self.appear_w = AppearanceFusion(in_node_feat=4, in_edge_feat=4).cuda()
            else:
                self.appear_w = AppearanceFusionVAE(in_node_feat=4, in_edge_feat=4).cuda()
            self.spt_gen = SptPropagate(top_k=self.top_k)
            self.rot_avg_finenet = Net_outlier_det(no_feats=32).cuda()
            # self.rot_avg_finenet = FineNetWithW().cuda()
            self.score_net = ScoreNetwork(top_k=self.top_k).cuda()

    def _load_network_from_ckpt(self, checkpoint_dict):
        # load network from checkpoint, ignore the instance if not found in dict.
        super(LocalGlobalVLADTrainBox, self)._load_network_from_ckpt(checkpoint_dict)
        with torch.cuda.device(self.dev_ids[0]):
            if 'appear_w' in checkpoint_dict:
                print('[Alert] Load appear_w')
                self.appear_w.load_state_dict(checkpoint_dict['appear_w'])
                # self.appear_w.cuda()
            if self.finenet_init_path is not None:
                print('Load from finenet_init_path: %s (Higher Priority than ckpt)' % self.finenet_init_path)
                finenet_dict = load_checkpoints(self.finenet_init_path)

                if 'model_state_dict' in checkpoint_dict:
                    print('Load from model_state_dict')
                    load_state_dict_by_key(self.rot_avg_finenet, checkpoint_dict['model_state_dict'])
                elif "finenet" in finenet_dict:
                    load_state_dict_by_key(self.rot_avg_finenet, finenet_dict['finenet'])
                else:
                    load_state_dict_by_key(self.rot_avg_finenet, finenet_dict["model_state_dict"])
            elif 'finenet' in checkpoint_dict:
                print('Load from finenet')
                load_state_dict_by_key(self.rot_avg_finenet, checkpoint_dict['finenet'])

                # self.rot_avg_finenet().cuda()

            # if 'score_net' in checkpoint_dict:
            #     print('Load score_net')
            #     load_state_dict_by_key(self.score_net, checkpoint_dict['score_net'])

    def _save_net_def(self, model_def_dir):
        # save the network definition, if needed in the future
        super(LocalGlobalVLADTrainBox, self)._save_net_def(model_def_dir)
        BaseTrainBox.save_instances_definition_to_dir([self.spt_gen,  # train-box
                                                      self.appear_w,
                                                      self.score_net,
                                                      self,
                                                      self.rot_avg_finenet], model_def_dir)

    def _save_checkpoint_dict(self, checkpoint_dict: dict):
        # save the instance when save_check_point was activated in training loop
        super(LocalGlobalVLADTrainBox, self)._save_checkpoint_dict(checkpoint_dict)
        checkpoint_dict['appear_w'] = self.appear_w.state_dict()
        # checkpoint_dict['score_net'] = self.score_net.state_dict()
        checkpoint_dict['finenet'] = self.rot_avg_finenet.state_dict()

    """ Separate Pipeline ---------------------------------------------------------------------------------------------- 
    """
    def forward_get_w(self, sample, sigmoid=True):
        img_lists, img_dims, Es, Ks, out_graph, sub2id, id2sub, _, _, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt, e_rel_err, node_feat, edge_feat = sample
        N = out_graph.shape[1]
        E = edge_label.shape[1]

        with torch.cuda.device(self.dev_ids[0]):
            cur_dev = torch.cuda.current_device()
            node_feat = node_feat.to(cur_dev)
            edge_feat = edge_feat.to(cur_dev)

            # build bi-direct
            bi_e_node_idx, bi_e_rel_Rt = graph_util.bi_direct_edge(e_node_idx, e_rel_Rt)
            bi_e_rel_q = rot2quaternion(bi_e_rel_Rt).to(cur_dev).detach()
            bi_e_feat = torch.cat([edge_feat, edge_feat], dim=0)
            bi_e_node_idx = bi_e_node_idx.permute(1, 0).long().detach().to(cur_dev)
            bi_e_label = torch.cat([edge_label, edge_label], dim=0).to(cur_dev).float()

            # get the weight for each edge
            # with torch.no_grad():
            if not self.use_vae:
                # input_edge_feat = torch.cat([bi_e_feat.view(2 * E, -1), bi_e_rel_q.view(2*E, -1)], dim=-1)
                input_edge_feat = bi_e_rel_q.view(2*E, -1)
                input_node_feat = torch.zeros((1,N,4)).to(cur_dev)
                ori_w, _, _ = self.appear_w.forward(input_node_feat.view(N, -1), input_edge_feat.view(2*E, -1), bi_e_node_idx)
            else:
                input_edge_feat = bi_e_rel_q.view(2*E, -1)
                input_node_feat = torch.zeros((1,N,4))
                mu, log_var = self.appear_w.forward(input_node_feat.view(N, -1), input_edge_feat.view(2*E, -1), bi_e_node_idx)
                ori_w = self.appear_w.reparameterize(mu, log_var).unsqueeze(1)

            if torch.sum(torch.isnan(ori_w)).item() > 0:
                return None, None, None, None, None, None, None

            # w_01, w_10 = sigmoid_w[:E], sigmoid_w[E:]
            # avg_sig_w = 0.5 * (w_01 + w_10)
            # sig_w = torch.cat([avg_sig_w, avg_sig_w])

            w = ori_w
            w_01, w_10 = w[:E], w[E:]
            avg_w = 0.5 * (w_01 + w_10)
            w = torch.cat([avg_w, avg_w])
            # if sigmoid:
            sigmoid_w = torch.sigmoid(w)

            # # run spt
            # start_node = graph_util.choose_anchor_node(N, e_node_idx, w[:E, :])
            # gt_R = graph_util.rel_ref_2_E_(start_node, Es)
            # gt_q = rot2quaternion(torch.cat(gt_R, dim=0).squeeze(1)).to(cur_dev)

            if self.use_vae:
                return w, bi_e_node_idx, bi_e_rel_q, mu, log_var

            return w, sigmoid_w, bi_e_node_idx, bi_e_rel_q

    """ Core -----------------------------------------------------------------------------------------------------------
    """
    # def forward_once(self, sample):
    #     img_lists, img_dims, Es, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt, e_rel_err, node_feat, edge_feat = sample
    #     N = out_graph.shape[1]
    #     E = edge_label.shape[1]
    #
    #     with torch.cuda.device(self.dev_ids[0]):
    #         cur_dev = torch.cuda.current_device()
    #         node_feat = node_feat.to(cur_dev)
    #         edge_feat = edge_feat.to(cur_dev)
    #
    #         # build bi-direct
    #         bi_e_node_idx, bi_e_rel_Rt = graph_util.bi_direct_edge(e_node_idx, e_rel_Rt)
    #         bi_e_rel_q = rot2quaternion(bi_e_rel_Rt).to(cur_dev).detach()
    #         bi_e_feat = torch.cat([edge_feat, edge_feat], dim=0)
    #         bi_e_node_idx = bi_e_node_idx.permute(1, 0).long().detach().to(cur_dev)
    #         bi_e_label = torch.cat([edge_label, edge_label], dim=0).to(cur_dev).float()
    #
    #         # get the weight for each edge
    #         # with torch.no_grad():
    #         if not self.use_vae:
    #             input_edge_feat = torch.cat([bi_e_feat.view(2 * E, -1), bi_e_rel_q.view(2*E, -1)], dim=-1)
    #             ori_w, _, _ = self.appear_w.forward(node_feat.view(N, -1), input_edge_feat.view(2*E, -1), bi_e_node_idx)
    #         else:
    #             mu, log_var = self.appear_w.forward(node_feat.view(N, -1), bi_e_feat.view(2*E, -1), bi_e_node_idx)
    #             ori_w = self.appear_w.reparameterize(mu, log_var).unsqueeze(1)
    #
    #         if torch.sum(torch.isnan(ori_w)).item() > 0:
    #             return None, None, None, None, None, None, None, None
    #         w = torch.sigmoid(ori_w)            # E*2
    #
    #         w_01, w_10 = w[:E], w[E:]
    #         avg_w = 0.5 * (w_01 + w_10)
    #         w = torch.cat([avg_w, avg_w])
    #
    #         # run propagate --------------------------------------------------------------------------------------------
    #         start_nodes = graph_util.choose_anchor_node(N, bi_e_node_idx, w[:2*E, :], top_k=self.top_k)
    #         init_qs, node_levels = self.spt_gen.iter(N, bi_e_node_idx, bi_e_rel_q, w.view(2*E), start_nodes=start_nodes, iter_num=5)
    #         init_qs = init_qs.to(cur_dev)
    #         node_levels = node_levels.to(cur_dev)
    #
    #         # run score net --------------------------------------------------------------------------------------------
    #         propagate_scores = self.score_net.forward(init_qs, node_levels, bi_e_node_idx, bi_e_rel_q)       # (K)
    #         propagate_scores = F.softmax(propagate_scores * 100.0)
    #
    #         init_qs = init_qs * propagate_scores.view(1, self.top_k, 1).expand(N, self.top_k, 1)            # (N, K, 4)
    #         init_q = torch.sum(init_qs, dim=1, keepdim=False)                                               # (N, 4)
    #         init_q = F.normalize(init_q, p=2, dim=1)
    #
    #         choosed_start_node = start_nodes[torch.argmax(propagate_scores).item()]
    #         # gt_R = graph_util.rel_ref_2_E_(choosed_start_node, Es)
    #         gt_R = Es[0, :, :3, :3]
    #         gt_q = rot2quaternion(gt_R).to(cur_dev)
    #         gt_q = self.recover_pred_q(gt_q, choosed_start_node)
    #
    #         # measure the initial relative pose err
    #         init_rel_loss = self.init_rel_loss(init_q, gt_q, bi_e_node_idx)
    #
    #         # run finenet to predict pose -------------------------------------------------------------------------------
    #         bi_e_node_idx = bi_e_node_idx.to(cur_dev)
    #         init_q = init_q.to(cur_dev)
    #         bi_e_rel_q = bi_e_rel_q.to(cur_dev)
    #
    #         in_data = [init_q, (bi_e_node_idx[0, :], bi_e_node_idx[1, :]), bi_e_rel_q, gt_q.to(cur_dev), bi_e_label.to(cur_dev)]
    #         pred_q, loss1, _, out_res = self.rot_avg_finenet(in_data)
    #
    #         init_q = self.recover_pred_q(init_q, choosed_start_node)
    #         pred_q = self.recover_pred_q(pred_q, choosed_start_node)
    #
    #     if self.use_vae:
    #         KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #         return init_q, pred_q, gt_q, ori_w, w, node_levels, loss1, choosed_start_node, KLD
    #
    #     return init_q, pred_q, gt_q, ori_w, w, node_levels, loss1, choosed_start_node

    def opt_init_forward_finenet(self,sample,max_itr=1):
        N,E, init_qs, gt_q,bi_e_node_idx, bi_e_rel_q , init_err,ds_name, start_nodes =sample
        N = N[0]
        init_qs = init_qs[0]
        gt_q = gt_q[0]
        bi_e_node_idx=bi_e_node_idx[0]
        bi_e_rel_q = bi_e_rel_q[0]
        init_err=init_err[0]
        ds_name=ds_name[0]
        start_nodes = start_nodes[0]
        with torch.cuda.device(self.dev_ids[0]):
            cur_dev = torch.cuda.current_device()
            bi_e_rel_q = bi_e_rel_q.to(cur_dev).detach()
            bi_e_label = torch.ones((bi_e_rel_q.shape[0])).to(cur_dev)
            bi_e_node_idx = bi_e_node_idx.to(cur_dev)
            bi_e_rel_q = bi_e_rel_q.to(cur_dev)
    
            k = np.random.choice(range(start_nodes.shape[0]))
            # k = 0
            gt_q = gt_q.to(cur_dev)
            start_node = start_nodes[k]
            gt_q = self.recover_pred_q(gt_q, start_node)
            init_q = init_qs[:, k, :]
            init_q = self.recover_pred_q(init_q, start_node)
    
            init_q = init_q.to(cur_dev)
            gt_q = gt_q.to(cur_dev)
    
            pred_q = init_q
            itr_losses = []
            for itr in range(max_itr):
                in_data = [pred_q, (bi_e_node_idx[0, :], bi_e_node_idx[1, :]), bi_e_rel_q, gt_q, bi_e_label]
                pred_q, loss1, _ = self.rot_avg_finenet(in_data)
                itr_losses.append(loss1)
    
            pred_q = self.recover_pred_q(pred_q, start_node)
            if max_itr > 1:
                return init_q, pred_q, gt_q, itr_losses, start_node

            return init_q, pred_q, gt_q, itr_losses[0], start_node

    def forward_finenet(self, sample, init_qs, start_nodes, edge_w, max_itr=1):
        img_lists, img_dims, Es, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt, e_rel_err, node_feat, edge_feat = sample
        N = out_graph.shape[1]
        E = edge_label.shape[1]
        with torch.cuda.device(self.dev_ids[0]):
            cur_dev = torch.cuda.current_device()
            bi_e_node_idx, bi_e_rel_Rt = graph_util.bi_direct_edge(e_node_idx, e_rel_Rt)
            bi_e_rel_q = rot2quaternion(bi_e_rel_Rt).to(cur_dev).detach()
            bi_e_feat = torch.cat([edge_feat, edge_feat], dim=0)
            bi_e_node_idx = bi_e_node_idx.permute(1, 0).long().detach().to(cur_dev)
            bi_e_label = torch.cat([edge_label, edge_label], dim=0).to(cur_dev).float()
            bi_e_node_idx = bi_e_node_idx.to(cur_dev)
            bi_e_rel_q = bi_e_rel_q.to(cur_dev)
            bi_e_label = torch.ones_like(bi_e_label).to(cur_dev)
            edge_w = edge_w.to(cur_dev)

            k = np.random.choice(range(init_qs.shape[1]))
            # k = 0
            gt_R = Es[0, :, :3, :3]
            gt_q = rot2quaternion(gt_R).to(cur_dev)
            start_node = start_nodes[k]
            gt_q = self.recover_pred_q(gt_q, start_node)
            init_q = init_qs[:, k, :]
            init_q = self.recover_pred_q(init_q, start_node)

            init_q = init_q.to(cur_dev)
            gt_q = gt_q.to(cur_dev)

            pred_q = init_q
            itr_losses = []
            for itr in range(max_itr):
                in_data = [pred_q, (bi_e_node_idx[0, :], bi_e_node_idx[1, :]), bi_e_rel_q, gt_q, bi_e_label]
                pred_q, loss1, _ = self.rot_avg_finenet(in_data)
                itr_losses.append(loss1)

            pred_q = self.recover_pred_q(pred_q, start_node)
            if max_itr > 1:
                return init_q, pred_q, gt_q, itr_losses, start_node

            return init_q, pred_q, gt_q, itr_losses[0], start_node

    def forward_init_cache(self, sample):
        img_lists, img_dims, Es, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt, e_rel_err, node_feat, edge_feat = sample
        N = out_graph.shape[1]
        E = edge_label.shape[1]

        with torch.cuda.device(self.dev_ids[0]):
            cur_dev = torch.cuda.current_device()
            node_feat = node_feat.to(cur_dev)
            edge_feat = edge_feat.to(cur_dev)

            # build bi-direct
            bi_e_node_idx, bi_e_rel_Rt = graph_util.bi_direct_edge(e_node_idx, e_rel_Rt)
            bi_e_rel_q = rot2quaternion(bi_e_rel_Rt).to(cur_dev).detach()
            bi_e_feat = torch.cat([edge_feat, edge_feat], dim=0)
            bi_e_node_idx = bi_e_node_idx.permute(1, 0).long().detach().to(cur_dev)
            bi_e_label = torch.cat([edge_label, edge_label], dim=0).to(cur_dev).float()

            # get the weight for each edge
            # with torch.no_grad():
            if not self.use_vae:
                # input_edge_feat = torch.cat([bi_e_feat.view(2 * E, -1), bi_e_rel_q.view(2*E, -1)], dim=-1)
                input_edge_feat = bi_e_feat.view(2*E, -1)
                ori_w, _, _ = self.appear_w.forward(node_feat.view(N, -1), input_edge_feat.view(2*E, -1), bi_e_node_idx)
            else:
                mu, log_var = self.appear_w.forward(node_feat.view(N, -1), bi_e_feat.view(2*E, -1), bi_e_node_idx)
                ori_w = self.appear_w.reparameterize(mu, log_var).unsqueeze(1)

            if torch.sum(torch.isnan(ori_w)).item() > 0:
                return None, None, None, None, None, None, None, None

            w = ori_w
            w_01, w_10 = w[:E], w[E:]
            avg_w = 0.5 * (w_01 + w_10)
            w = torch.cat([avg_w, avg_w])
            w = torch.sigmoid(0.8 * w)

            # run propagate --------------------------------------------------------------------------------------------
            start_nodes = graph_util.choose_anchor_node(N, bi_e_node_idx, w[:2*E, :], top_k=self.top_k)
            init_qs, node_levels = self.spt_gen.iter(N, bi_e_node_idx, bi_e_rel_q, w.view(2*E), start_nodes=start_nodes, iter_num=5)

            if self.use_vae:
                return init_qs, node_levels, ori_w, w, log_var, mu, start_nodes
            else:
                return init_qs, node_levels, ori_w, w, start_nodes


    def forward_init(self, sample):
        img_lists, img_dims, Es, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt, e_rel_err, node_feat, edge_feat = sample
        N = out_graph.shape[1]
        E = edge_label.shape[1]

        with torch.cuda.device(self.dev_ids[0]):
            cur_dev = torch.cuda.current_device()
            node_feat = node_feat.to(cur_dev)
            edge_feat = edge_feat.to(cur_dev)

            # build bi-direct
            bi_e_node_idx, bi_e_rel_Rt = graph_util.bi_direct_edge(e_node_idx, e_rel_Rt)
            bi_e_rel_q = rot2quaternion(bi_e_rel_Rt).to(cur_dev).detach()
            bi_e_feat = torch.cat([edge_feat, edge_feat], dim=0)
            bi_e_node_idx = bi_e_node_idx.permute(1, 0).long().detach().to(cur_dev)
            bi_e_label = torch.cat([edge_label, edge_label], dim=0).to(cur_dev).float()

            # get the weight for each edge
            # with torch.no_grad():
            if not self.use_vae:
                # input_edge_feat = torch.cat([bi_e_feat.view(2 * E, -1), bi_e_rel_q.view(2*E, -1)], dim=-1)
                input_edge_feat = bi_e_rel_q.view(2*E, -1)
                input_node_feat = torch.zeros((1,N,4)).to(cur_dev)
                ori_w, _, _ = self.appear_w.forward(input_node_feat.view(N, -1), input_edge_feat.view(2*E, -1), bi_e_node_idx)
            else:
                input_edge_feat = bi_e_rel_q.view(2*E, -1)
                input_node_feat = torch.zeros((1,N,4))
                mu, log_var = self.appear_w.forward(input_node_feat.view(N, -1), input_edge_feat.view(2*E, -1), bi_e_node_idx)
                ori_w = self.appear_w.reparameterize(mu, log_var).unsqueeze(1)

            if torch.sum(torch.isnan(ori_w)).item() > 0:
                return None, None, None, None, None, None, None, None

            w = ori_w
            w_01, w_10 = w[:E], w[E:]
            avg_w = 0.5 * (w_01 + w_10)
            w = torch.cat([avg_w, avg_w])
            w = torch.sigmoid(0.6 * w)

            # run propagate --------------------------------------------------------------------------------------------
            gt_R = Es[0, :, :3, :3]
            ori_gt_q = rot2quaternion(gt_R).to(cur_dev)

            start_nodes = graph_util.choose_anchor_node(N, bi_e_node_idx, w[:2*E, :], top_k=self.top_k)
            init_qs, node_levels = self.spt_gen.iter(N, bi_e_node_idx, bi_e_rel_q, w.view(2*E),
                                                     start_nodes=start_nodes, iter_num=5,
                                                     cpu_only=True if self.top_k < 26 else False)
            # init_qs, dim: (N, K, 4)
            # node_levels, dim: (N, K)
            init_qs = init_qs.to(cur_dev)
            node_levels = node_levels.to(cur_dev)

            # rel loss for all propagate -------------------------------------------------------------------------------
            # init_rel_loss, choosed_start_node = self.init_rel_loss_batch_(init_qs, ori_gt_q, bi_e_node_idx)
            init_rel_loss = self.init_rel_loss_l1_batch_(init_qs, ori_gt_q, bi_e_node_idx[:, :E])
            init_rel_loss = init_rel_loss.mean()

            # abs loss for all propagate -------------------------------------------------------------------------------
            abs_loss = self.init_abs_loss_batch(init_qs, ori_gt_q.to(cur_dev), start_nodes)
            # abs_loss = 0

            # # run score net ------------------------------------------------------------------------------------------
            # propagate_scores = self.score_net.forward(init_qs, node_levels, bi_e_node_idx, bi_e_rel_q)       # (K)
            # propagate_scores = F.softmax(propagate_scores * 100.0)
            #
            # init_qs = init_qs * propagate_scores.view(1, self.top_k, 1).expand(N, self.top_k, 1)            # (N, K, 4)
            # init_q = torch.sum(init_qs, dim=1, keepdim=False)                                               # (N, 4)
            # init_q = F.normalize(init_q, p=2, dim=1)

            choosed_start_node = start_nodes[0]
            # gt_R = graph_util.rel_ref_2_E_(choosed_start_node, Es)

            gt_q = self.recover_pred_q(ori_gt_q, choosed_start_node)

            # measure the initial relative pose err
            init_q = init_qs[:, 0, :]
            init_q = self.recover_pred_q(init_q, choosed_start_node)
            # init_rel_loss = self.init_rel_loss(init_q, gt_q, bi_e_node_idx)

        if self.use_vae:
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            return init_q, gt_q, ori_w, w, node_levels, init_rel_loss, abs_loss, choosed_start_node, KLD

        return init_q, gt_q, ori_w, w, node_levels, init_rel_loss, abs_loss, choosed_start_node

    """ Loss Functions -------------------------------------------------------------------------------------------------
    """
    def rot_err(self, pred_q, gt_q, invalid_nodes=None):
        pred_residual_q = qmul(inv_q(gt_q), pred_q)
        pred_residual_q = F.normalize(pred_residual_q, p=2, dim=-1)
        val2 = pred_residual_q.data.cpu().numpy()
        # todo: check
        if invalid_nodes is not None:
            vals = []
            for n in range(val2.shape[0]):
                if n not in invalid_nodes:
                    vals.append(val2[n])
            vals = np.asarray(vals)
        else:
            vals = val2
        theta = 2.0 * np.arccos(np.abs(vals[:, 0])) * 180.0 / np.pi
        return theta

    def recover_pred_q(self, pred_q, root):
        N = pred_q.shape[0]
        ref_ = pred_q[root: root + 1, :]
        inv_ref_ = inv_q(ref_)
        inv_ref_ext_ = inv_ref_.view(1, 4).repeat(N, 1)
        ref_to_q = qmul(pred_q, inv_ref_ext_)
        return ref_to_q

    def recover_pred_q_(self, inv_pred_q, pred_q, root):
        N = inv_pred_q.shape[0]
        ref_pred_q = pred_q[root:root + 1, :]   # dim: (1, 4)
        inv_ref_pred_q = inv_q(ref_pred_q)
        inv_ref_pred_q_exp = inv_ref_pred_q.repeat(N, 1)
        new_pred_q = qmul(pred_q, inv_ref_pred_q_exp)
        return new_pred_q

    def smooth_l1_loss(self, input, beta, alpha=0.05, size_average=True):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])
        nn = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
        nn = nn * beta

        cond = nn < alpha
        loss = torch.where(cond, 0.5 * nn ** 2 / alpha, nn - 0.5 * alpha)
        return loss.mean() if size_average else loss.sum()

    def l1_loss(self, input, weight=1.0, size_average=True):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])
        nn = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
        nn = nn * weight
        loss = nn
        return loss.mean() if size_average else loss.sum()

    def l2_loss(self, input, weight=1.0, size_average=True):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])
        nn = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
        nn = nn ** 2
        nn = nn * weight
        loss = nn
        return loss.mean() if size_average else loss.sum()

    # N = pred.shape[0]
    # ref_ = pred[root: root +1, :].view(1, 4).repeat(N, 1)
    # q_ref_to_p = qmul(pred, inv_q(ref_))
    # return q_ref_to_p

    def init_abs_loss_batch(self, init_qs, ori_gt_, start_nodes):
        N, K = init_qs.shape[:2]

        # select ref nodes with start_nodes on ground-truth
        ref_gt_q = [ori_gt_[start_nodes[k], :].view(1, 4) for k in range(K)]                            # dim: (K, 4)
        ref_gt_q = torch.cat(ref_gt_q, dim=0)
        inv_ref_gt_q = inv_q(ref_gt_q)

        # select ref nodes with start_nodes on init_qs
        ref_init_q = [init_qs[start_nodes[k], k, :].view(1, 4) for k in range(K)]                       # dim: (K, 4)
        ref_init_q = torch.cat(ref_init_q, dim=0)
        inv_ref_init_q = inv_q(ref_init_q)

        # abs loss for all propagate -------------------------------------------------------------------------------

        # convert gt_q to ref
        ori_gt_q_ext = ori_gt_.view(N, 1, 4).repeat(1, K, 1)                                           # dim: (N, K, 4)
        inv_ref_gt_q_ext = inv_ref_gt_q.view(1, K, 4).repeat(N, 1, 1)                                  # dim: (N, K, 4)
        trans_ref_gt = qmul(ori_gt_q_ext.view(-1, 4), inv_ref_gt_q_ext.view(-1, 4)).view(N, K, 4)      # dim: (N, K, 4)

        init_qs = init_qs.view(N, K, 4)                                                                # dim: (N, K, 4)
        inv_ref_init_q_ext = inv_ref_init_q.view(1, K, 4).repeat(N, 1, 1)
        trans_ref_init = qmul(init_qs.view(-1, 4), inv_ref_init_q_ext.view(-1, 4)).view(N, K, 4)       # dim: (N * K, 4)

        init_residual_q = qmul(inv_q(trans_ref_init.view(-1, 4)), trans_ref_gt.view(-1, 4)).view(N*K, 4)
        init_residual_q = F.normalize(init_residual_q, p=2, dim=-1).view(N, K, 4)

        loss_abs_init = FN.smooth_l1_loss(init_residual_q[:, 0:].view(-1, 4), size_average=True)

        return loss_abs_init

    def init_rel_loss(self, init_q, gt_q, edge_index):
        # todo: check
        loss1 = qmul(inv_q(edge_model(gt_q, edge_index)), edge_model(init_q, edge_index))
        loss1 = F.normalize(loss1, p=2, dim=-1)
        return self.smooth_l1_loss(loss1, 1.0, edge_index)

    def init_rel_loss_batch_(self, init_qs, gt_q, edge_index):
        N, K, _ = init_qs.shape
        inv_gt_rel_q = inv_q(edge_model(gt_q, edge_index))
        total_loss = 0
        min_k, min_k_loss = 0, 10000.0
        for k in range(K):
            loss1 = qmul(inv_gt_rel_q, edge_model(init_qs[:, k, :], edge_index))
            loss1 = F.normalize(loss1, p=2, dim=-1)
            loss1 = my_smooth_l1_loss_(loss1, 1.0, edge_index, gt_q.size)

            if loss1.item() < min_k_loss:
                min_k_loss = loss1.item()
                min_k = k

            total_loss += loss1
        return total_loss/K, min_k

    def init_rel_loss_l1_batch_(self, init_qs, ori_gt_q, e_node_idx):
        K = init_qs.shape[1]
        E = e_node_idx.shape[1]
        init_rel_qs = edge_model_batch(init_qs, e_node_idx[:, :E]).view(E * K, 4)  # dim: (E, K, 4)
        rel_gt_q = edge_model(ori_gt_q, e_node_idx[:, :E])
        rel_gt_q_ext = rel_gt_q[:E, :].view(E, 1, 4).repeat(1, K, 1)
        rel_err_q = qmul(inv_q(init_rel_qs), rel_gt_q_ext).view(E, K, 4)
        rel_err_q = F.normalize(rel_err_q, p=2, dim=-1).view(E, K, 4)
        init_rel_loss = huber_l1_loss(rel_err_q.view(E * K, 4), 1.0)
        return init_rel_loss

    """ Train Routines -------------------------------------------------------------------------------------------------
    """
    def _prepare_train(self):
        self.appear_w.train()
        self.rot_avg_finenet.train()
        self.score_net.train()

    def overfit(self, train_dataset, epoch, num_workers=2):
        self._prepare_train()
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                                  num_workers=num_workers, pin_memory=False, drop_last=True)

        n_samples = len(train_dataset)
        for e_i in range(0, epoch):
            total_rel_loss = 0
            total_init_loss = 0
            total_pred_loss = 0
            total_w_loss = 0
            total_kl_loss = 0

            init_theta = []
            pred_theta = []
            no_training = 0
            nan_counting = 0

            p_bar = tqdm(total=n_samples, desc='Epoch=%d' % e_i)
            for s_i, sample in enumerate(data_loader):

                ds, idx, img_name, _, img_dims, Es, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt, e_rel_err, _, _ = sample
                self.optimizer.zero_grad()

                # init_q, pred_q, gt_q, ori_w, w, node_levels, loss1, start_node

                if self.use_vae:
                    init_q, pred_q, gt_q, _, w, node_levels, rel_loss, start_node, KLD_loss = self.forward_once(sample[3:])
                else:
                    init_q, pred_q, gt_q, _, w, node_levels, rel_loss, start_node = self.forward_once(sample[3:])
                if init_q is None:
                    print('Meet NaN')
                    nan_counting += 1
                    continue

                # out_res = self.forward_appear(sample[2:])
                # edge_label = edge_label.view(-1)
                # bi_edge_label = torch.cat([edge_label, edge_label], dim=0).float()
                # bce_loss = self.bce_loss(out_res.view(-1), bi_edge_label.to(out_res.device))
                # bce_loss += 0.1*self.bce_loss(out_init_res.view(-1), bi_edge_label.to(out_res.device))

                # regulizer: make sure w is close to 0 when rel > 20 ---------------------------------------------------
                mask = (1 - edge_label)
                mask = torch.cat([mask, mask], dim=0).reshape(*w.shape)
                loss_w = (w.to(mask.device) * mask).sum()

                # loss function (pred) ---------------------------------------------------------------------------------
                node_mask = (node_levels > 0.1).float().to(pred_q.device).reshape(pred_q.shape[0], 1)

                pred_residual_q = qmul(inv_q(gt_q), pred_q)
                pred_residual_q = F.normalize(pred_residual_q, p=2, dim=-1)
                loss_abs = FN.smooth_l1_loss(pred_residual_q[:, 0:])

                # loss function (init) ---------------------------------------------------------------------------------
                init_residual_q = qmul(inv_q(gt_q), init_q)
                init_residual_q = F.normalize(init_residual_q, p=2, dim=-1)
                loss_abs_init = FN.smooth_l1_loss(init_residual_q[:, 0:])

                # init_err = self.rot_err(init_q, gt_q)
                # pred_err = self.rot_err(pq3, gt_q)
                #
                # print(["E:%d Itr:%d/%d" % (e_i, s_i, n_samples),
                #        "init_err:{0:.2f}".format(np.mean(init_err)),
                #        "opt_err:{0:.2f}".format(np.mean(pred_err)),
                #        "init_err_median:{0:.2f}".format(np.median(init_err)),
                #        "opt_err_median:{0:.2f}".format(np.median(pred_err))
                #        ])

                if torch.isnan(rel_loss).item():
                    continue
                else:
                    no_training += 1
                    loss = rel_loss + 2.0 *loss_abs + 5.0*loss_abs_init
                    loss.backward()
                    self.optimizer.step()

                if epoch % 1 == 0:
                    total_rel_loss = total_rel_loss + rel_loss.item()
                    total_pred_loss = total_pred_loss + loss_abs.item()
                    total_init_loss = total_init_loss + loss_abs_init.item()
                    total_w_loss = total_w_loss + loss_w.item()
                    # total_kl_loss = total_kl_loss + KLD_loss.item()

                    val2 = init_residual_q.cpu().clone().detach().numpy()
                    pred_val2 = pred_residual_q.data.cpu().clone().detach().numpy()
                    init_theta = np.concatenate((init_theta, 2.0 * np.arccos(np.abs(val2[:, 0])) * 180.0 / np.pi))
                    pred_theta = np.concatenate((pred_theta, 2.0 * np.arccos(np.abs(pred_val2[:, 0])) * 180.0 / np.pi))

                p_bar.update(1)

            total_rel_loss = total_rel_loss / no_training
            total_pred_loss = total_pred_loss / no_training
            total_init_loss = total_init_loss / no_training
            total_w_loss = total_w_loss / no_training
            # total_kl_loss = total_kl_loss / no_training

            if epoch % 1 == 0:
                print([e_i, "rel_loss:{0:.6f}".format(total_rel_loss),
                       "pred_loss:{0:.2f}".format(total_pred_loss),
                       "init_loss:{0:.2f}".format(total_init_loss),
                       "w_loss:{0:.2f}".format(total_w_loss),
                       # "kld_loss{0:.2f}".format(np.median(total_kl_loss)),
                       "deg_mean:{0:.2f}".format(np.mean(pred_theta)),
                       "init_deg_mean:{0:.2f}".format(np.mean(init_theta)),
                       "deg_std:{0:.2f}".format(np.std(pred_theta)),
                       "init_deg_std:{0:.2f}".format(np.std(init_theta)),
                       "deg_median{0:.2f}".format(np.median(pred_theta)),
                       "init_deg_median{0:.2f}".format(np.median(init_theta)),
                       'NAN: %d' % (nan_counting)
                       ])

                if self.logger is not None:
                    writer = self.logger.get_tensorboard_writer()
                    writer.add_scalar("rel_loss", total_rel_loss, e_i)
                    writer.add_scalar("pred_loss", total_pred_loss, e_i)
                    writer.add_scalar("init_loss", total_init_loss, e_i)
                    writer.add_scalar("w_loss", total_w_loss, e_i)

                    writer.add_scalar("init_median", np.median(init_theta), e_i)
                    writer.add_scalar("pred_median", np.median(pred_theta), e_i)
                    writer.add_scalar("init_mean", np.mean(init_theta), e_i)
                    writer.add_scalar("pred_mean", np.mean(pred_theta), e_i)

            if epoch % 1 == 0:
                self.save_checkpoint(epoch=epoch-1, itr=1)

    def overfit_init(self, train_dataset, epoch, num_workers=2, ignored_ds=None):
        print('[Trainbox] Overfiting')
        if ignored_ds is not None:
            print('Ignored Dataset:')
            print(ignored_ds)

        self._prepare_train()
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                                  num_workers=num_workers, pin_memory=False, drop_last=True)

        n_samples = len(train_dataset)
        min_init_loss = 10000.0
        for e_i in range(0, epoch):
            total_rel_loss = 0
            total_init_loss = 0
            total_pred_loss = 0
            total_w_loss = 0
            total_kl_loss = 0

            init_theta = []
            pred_theta = []
            no_training = 0
            nan_counting = 0

            p_bar = tqdm(total=n_samples, desc='Epoch=%d' % e_i)
            for s_i, sample in enumerate(data_loader):

                ds, idx, img_name, _, img_dims, Es, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt, e_rel_err, _, _ = sample
                if ignored_ds is not None and ds[0] in ignored_ds:
                    continue

                self.optimizer.zero_grad()

                # init_q, pred_q, gt_q, ori_w, w, node_levels, loss1, start_nodevim

                if self.use_vae:
                    init_q, gt_q, _, w, node_levels, init_rel_loss, abs_loss, start_node, KLD_loss = self.forward_init(sample[3:])
                else:
                    init_q, gt_q, _, w, node_levels, init_rel_loss, abs_loss, start_node = self.forward_init(sample[3:])
                if init_q is None:
                    print('Meet NaN')
                    nan_counting += 1
                    continue

                # regulizer: make sure w is close to 0 when rel > 20 ---------------------------------------------------
                mask = (1 - edge_label)
                mask = torch.cat([mask, mask], dim=0).view(*w.shape)
                loss_w_ = (w.to(mask.device) * mask)
                loss_w_sum = loss_w_.sum()
                loss_w = torch.norm(loss_w_)

                # # loss function (init) ---------------------------------------------------------------------------------
                init_residual_q = qmul(inv_q(gt_q), init_q)
                init_residual_q = F.normalize(init_residual_q, p=2, dim=-1)
                loss_abs_init = FN.smooth_l1_loss(init_residual_q[:, 0:])

                if torch.isnan(init_rel_loss).item():
                    continue
                else:
                    no_training += 1
                    loss = init_rel_loss + 0.3 * abs_loss + 0.0005 * loss_w
                    loss.backward()
                    self.optimizer.step()

                if epoch % 1 == 0:
                    total_rel_loss = total_rel_loss + init_rel_loss.item()
                    total_init_loss = total_init_loss + abs_loss.item()
                    total_w_loss = total_w_loss + loss_w_sum.item()
                    # total_kl_loss = total_kl_loss + KLD_loss.item()

                    val2 = init_residual_q.cpu().clone().detach().numpy()
                    init_theta = np.concatenate((init_theta, 2.0 * np.arccos(np.abs(val2[:, 0])) * 180.0 / np.pi))

                p_bar.update(1)

            total_rel_loss = total_rel_loss / no_training
            total_pred_loss = total_pred_loss / no_training
            total_init_loss = total_init_loss / no_training
            total_w_loss = total_w_loss / no_training
            # total_kl_loss = total_kl_loss / no_training

            if min_init_loss > total_init_loss:
                # save best
                self.save_checkpoint(epoch=epoch, itr=998)
                min_init_loss = total_init_loss

            if epoch % 1 == 0:
                print([e_i, "rel_loss:{0:.6f}".format(total_rel_loss),
                       "pred_loss:{0:.2f}".format(total_pred_loss),
                       "init_loss:{0:.2f}".format(total_init_loss),
                       "w_loss:{0:.2f}".format(total_w_loss),
                       # "kld_loss{0:.2f}".format(np.median(total_kl_loss)),
                       # "deg_mean:{0:.2f}".format(np.mean(pred_theta)),
                       "init_deg_mean:{0:.2f}".format(np.mean(init_theta)),
                       # "deg_std:{0:.2f}".format(np.std(pred_theta)),
                       "init_deg_std:{0:.2f}".format(np.std(init_theta)),
                       # "deg_median{0:.2f}".format(np.median(pred_theta)),
                       "init_deg_median{0:.2f}".format(np.median(init_theta)),
                       'NAN: %d' % (nan_counting)
                       ])

                if self.logger is not None:
                    writer = self.logger.get_tensorboard_writer()
                    writer.add_scalar("rel_loss", total_rel_loss, e_i)
                    writer.add_scalar("pred_loss", total_pred_loss, e_i)
                    writer.add_scalar("init_loss", total_init_loss, e_i)
                    writer.add_scalar("w_loss", total_w_loss, e_i)

                    writer.add_scalar("init_median", np.median(init_theta), e_i)
                    # writer.add_scalar("pred_median", np.median(pred_theta), e_i)
                    writer.add_scalar("init_mean", np.mean(init_theta), e_i)
                    # writer.add_scalar("pred_mean", np.mean(pred_theta), e_i)

            if epoch % 1 == 0:
                self.save_checkpoint(epoch=epoch-1, itr=e_i-1)

    def opt_init_overfit(self, train_dataset, epoch, num_workers=2, filtering_edge_thres=-1):
        self._prepare_eval()
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                                  num_workers=num_workers, pin_memory=False, drop_last=True)

        n_samples = len(train_dataset)
        print("[Trainbox] Running Init")
        p_bar = tqdm(total=n_samples, desc='Init')
        init_theta = np.asarray([]).astype(np.float32)
        for sample in data_loader:
            N,E, init_qs, gt_q, bi_e_node_idx, bi_e_rel_q, init_err, name, start_nodes = sample
            init_theta = np.concatenate((init_theta, init_err.detach().cpu().numpy().ravel()))
        # print info ---------------------------------------------------------------------------------------------------
        print(['[Init]',
               "init_deg_mean:{0:.2f}".format(np.mean(init_theta)),
               "init_deg_std:{0:.2f}".format(np.std(init_theta)),
               "init_deg_median:{0:.2f}".format(np.median(init_theta))
               ])

        # train the finenet ---------------------------------------------------------------------------------------------
        optimizer = torch.optim.Adam([
            {'params': self.rot_avg_finenet.parameters(), 'lr': 1.0e-5}
        ], lr=self.train_params.START_LR)
        self.rot_avg_finenet.train()

        for e_i in range(epoch):
            total_rel_loss = 0
            total_init_loss = 0
            total_pred_loss = 0
            total_w_loss = 0
            total_kl_loss = 0

            init_theta = []
            pred_theta = []
            no_training = 0
            nan_counting = 0
            p_bar = tqdm(total=n_samples, desc='Epoch=%d' % e_i)
            for d_i, sample in enumerate(data_loader):
                optimizer.zero_grad()

                if filtering_edge_thres != -1:
                    sample = filtering(sample, r_thres=filtering_edge_thres)

                N,E, init_qs, gt_q, bi_e_node_idx, bi_e_rel_q, init_err, name, start_nodes = sample
                N = N[0]
                init_qs = init_qs[0]
                qt_q = gt_q[0]
                bi_e_node_idx=bi_e_node_idx[0]
                bi_e_rel_q = bi_e_rel_q[0]
                init_err=init_err[0]
                name=name[0]
                start_nodes = start_nodes[0]
                init_q, pred_q, gt_q, rel_loss_set, start_node = self.opt_init_forward_finenet(sample, max_itr=3)
                if init_q is None:
                    print("Meet NaN")
                    nan_counting += 1
                    continue

                rel_loss = rel_loss_set[-1]

                pred_residual_q = qmul(inv_q(gt_q), pred_q)
                pred_residual_q = F.normalize(pred_residual_q, p=2, dim=1)
                loss_abs = FN.smooth_l1_loss(pred_residual_q[:, 0:])

                init_residual_q = qmul(inv_q(gt_q), init_q)
                init_residual_q = F.normalize(init_residual_q, p=2, dim=1)
                loss_abs_init = FN.smooth_l1_loss(init_residual_q[:, 0:])

                if torch.isnan(rel_loss).item():
                    continue
                else:
                    no_training += 1
                    loss = 4 * rel_loss_set[-1] + 2 * rel_loss_set[-2] + rel_loss_set[-3] + 0.6 * loss_abs
                    loss.backward()
                    optimizer.step()

                if epoch % 1 == 0:
                    total_rel_loss = total_rel_loss + rel_loss.item()
                    total_pred_loss = total_pred_loss + loss_abs.item()
                    total_init_loss = total_init_loss + loss_abs_init.item()
                val2 = init_residual_q.cpu().clone().detach().numpy()
                pred_val2 = pred_residual_q.data.cpu().clone().detach().numpy()
                init_theta = np.concatenate((init_theta, 2.0 * np.arccos(np.abs(val2[:, 0])) * 180.0 / np.pi))
                pred_theta = np.concatenate((pred_theta, 2.0 * np.arccos(np.abs(pred_val2[:, 0])) * 180.0 / np.pi))

                p_bar.update(1)

            total_rel_loss = total_rel_loss / no_training
            total_pred_loss = total_pred_loss / no_training
            total_init_loss = total_init_loss / no_training
            total_w_loss = total_w_loss / no_training
            if epoch % 1 == 0:
                print([e_i, "rel_loss:{0:.6f}".format(total_rel_loss),
                       "pred_loss:{0:.2f}".format(total_pred_loss),
                       "init_loss:{0:.2f}".format(total_init_loss),
                       "w_loss:{0:.2f}".format(total_w_loss),
                       # "kld_loss{0:.2f}".format(np.median(total_kl_loss)),
                       "deg_mean:{0:.2f}".format(np.mean(pred_theta)),
                       "init_deg_mean:{0:.2f}".format(np.mean(init_theta)),
                       "deg_std:{0:.2f}".format(np.std(pred_theta)),
                       "init_deg_std:{0:.2f}".format(np.std(init_theta)),
                       "deg_median{0:.2f}".format(np.median(pred_theta)),
                       "init_deg_median{0:.2f}".format(np.median(init_theta)),
                       'NAN: %d' % (nan_counting)
                       ])

                if self.logger is not None:
                    writer = self.logger.get_tensorboard_writer()
                    writer.add_scalar("rel_loss", total_rel_loss, e_i)
                    writer.add_scalar("pred_loss", total_pred_loss, e_i)
                    writer.add_scalar("init_loss", total_init_loss, e_i)
                    writer.add_scalar("w_loss", total_w_loss, e_i)

                    writer.add_scalar("init_median", np.median(init_theta), e_i)
                    writer.add_scalar("pred_median", np.median(pred_theta), e_i)
                    writer.add_scalar("init_mean", np.mean(init_theta), e_i)
                    writer.add_scalar("pred_mean", np.mean(pred_theta), e_i)

            if e_i % 10 == 0:
                self.save_checkpoint(epoch=epoch - 1, itr=e_i)

    def fix_init_overfit(self, train_dataset, epoch, num_workers=2, filtering_edge_thres=-1):
        self._prepare_eval()
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                                  num_workers=num_workers, pin_memory=False, drop_last=True)

        init_result = []
        n_samples = len(train_dataset)
        print("[Trainbox] Running Init")
        p_bar = tqdm(total=n_samples, desc='Init')
        init_theta = []
        for s_i, sample in enumerate(data_loader):
            if filtering_edge_thres != -1:
                sample = filtering(sample, r_thres=filtering_edge_thres)

            ds, idx, img_name, _, img_dims, Es, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt, e_rel_err, _, _ = sample

            with torch.no_grad():
                if self.use_vae:
                    pass
                else:
                    init_qs, node_levels, ori_w, w, start_nodes = self.forward_init_cache(sample[3:])
                    init_result.append((init_qs, node_levels, ori_w, w, start_nodes))

                    # measure the initial relative pose err
                    choosed_start_node = start_nodes[0]
                    gt_q = self.recover_pred_q(rot2quaternion(Es[0, :, :3, :3]), choosed_start_node)
                    init_q = self.recover_pred_q(init_qs[:, 0, :], choosed_start_node).to(gt_q.device)
                    init_residual_q = qmul(inv_q(gt_q), init_q)
                    init_residual_q = F.normalize(init_residual_q, p=2, dim=-1)
                    val2 = init_residual_q.cpu().clone().detach().numpy()
                    init_theta = np.concatenate((init_theta, 2.0 * np.arccos(np.abs(val2[:, 0])) * 180.0 / np.pi))

                with torch.cuda.device(self.dev_ids[0]):
                    torch.cuda.empty_cache()

            p_bar.update(1)

        # print info ---------------------------------------------------------------------------------------------------
        print(['[Init]',
               "init_deg_mean:{0:.2f}".format(np.mean(init_theta)),
               "init_deg_std:{0:.2f}".format(np.std(init_theta)),
               "init_deg_median{0:.2f}".format(np.median(init_theta))
               ])

        # train the finenet ---------------------------------------------------------------------------------------------
        optimizer = torch.optim.Adam([
            {'params': self.rot_avg_finenet.parameters(), 'lr': 2.0e-4}
        ], lr=self.train_params.START_LR)
        self.rot_avg_finenet.train()

        for e_i in range(epoch):
            total_rel_loss = 0
            total_init_loss = 0
            total_pred_loss = 0
            total_w_loss = 0
            total_kl_loss = 0

            init_theta = []
            pred_theta = []
            no_training = 0
            nan_counting = 0
            p_bar = tqdm(total=n_samples, desc='Epoch=%d' % e_i)
            for d_i, sample in enumerate(data_loader):
                optimizer.zero_grad()

                if filtering_edge_thres != -1:
                    sample = filtering(sample, r_thres=filtering_edge_thres)

                ds, idx, img_name, _, img_dims, Es, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, \
                edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt, e_rel_err, _, _ = sample

                if self.use_vae:
                    init_qs, node_levels, ori_w, w, log_var, mu, start_nodes = init_result[d_i]
                    pass
                else:
                    init_qs, node_levels, ori_w, w, start_nodes = init_result[d_i]
                init_q, pred_q, gt_q, rel_loss_set, start_node = self.forward_finenet(sample[3:], init_qs, start_nodes, w, max_itr=2)
                if init_q is None:
                    print("Meet NaN")
                    nan_counting += 1
                    continue
                if self.use_vae:
                    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                rel_loss = rel_loss_set[-1]
                # rel_loss = rel_loss_set

                mask = (1 - edge_label)
                mask = torch.cat([mask, mask], dim=0).view(*w.shape)
                loss_w_ = (w.to(mask.device) * mask)
                loss_w_sum = loss_w_.sum()
                loss_w = torch.norm(loss_w_)

                pred_residual_q = qmul(inv_q(gt_q), pred_q)
                pred_residual_q = F.normalize(pred_residual_q, p=2, dim=1)
                loss_abs = FN.smooth_l1_loss(pred_residual_q[:, 0:])

                init_residual_q = qmul(inv_q(gt_q), init_q)
                init_residual_q = F.normalize(init_residual_q, p=2, dim=1)
                loss_abs_init = FN.smooth_l1_loss(init_residual_q[:, 0:])

                if torch.isnan(rel_loss).item():
                    continue
                else:
                    no_training += 1
                    loss = 2 * rel_loss_set[-1] + 1*rel_loss_set[-2] + 0.5 * loss_abs
                    # loss = 1 * rel_loss_set + 0.3 * loss_abs
                    loss.backward()
                    optimizer.step()

                if epoch % 1 == 0:
                    total_rel_loss = total_rel_loss + rel_loss.item()
                    total_pred_loss = total_pred_loss + loss_abs.item()
                    total_init_loss = total_init_loss + loss_abs_init.item()
                    total_w_loss = total_w_loss + loss_w.item()
                val2 = init_residual_q.cpu().clone().detach().numpy()
                pred_val2 = pred_residual_q.data.cpu().clone().detach().numpy()
                init_theta = np.concatenate((init_theta, 2.0 * np.arccos(np.abs(val2[:, 0])) * 180.0 / np.pi))
                pred_theta = np.concatenate((pred_theta, 2.0 * np.arccos(np.abs(pred_val2[:, 0])) * 180.0 / np.pi))

                p_bar.update(1)

            total_rel_loss = total_rel_loss / no_training
            total_pred_loss = total_pred_loss / no_training
            total_init_loss = total_init_loss / no_training
            total_w_loss = total_w_loss / no_training
            if epoch % 1 == 0:
                print([e_i, "rel_loss:{0:.6f}".format(total_rel_loss),
                       "pred_loss:{0:.2f}".format(total_pred_loss),
                       "init_loss:{0:.2f}".format(total_init_loss),
                       "w_loss:{0:.2f}".format(total_w_loss),
                       # "kld_loss{0:.2f}".format(np.median(total_kl_loss)),
                       "deg_mean:{0:.2f}".format(np.mean(pred_theta)),
                       "init_deg_mean:{0:.2f}".format(np.mean(init_theta)),
                       "deg_std:{0:.2f}".format(np.std(pred_theta)),
                       "init_deg_std:{0:.2f}".format(np.std(init_theta)),
                       "deg_median{0:.2f}".format(np.median(pred_theta)),
                       "init_deg_median{0:.2f}".format(np.median(init_theta)),
                       'NAN: %d' % (nan_counting)
                       ])

                if self.logger is not None:
                    writer = self.logger.get_tensorboard_writer()
                    writer.add_scalar("rel_loss", total_rel_loss, e_i)
                    writer.add_scalar("pred_loss", total_pred_loss, e_i)
                    writer.add_scalar("init_loss", total_init_loss, e_i)
                    writer.add_scalar("w_loss", total_w_loss, e_i)

                    writer.add_scalar("init_median", np.median(init_theta), e_i)
                    writer.add_scalar("pred_median", np.median(pred_theta), e_i)
                    writer.add_scalar("init_mean", np.mean(init_theta), e_i)
                    writer.add_scalar("pred_mean", np.mean(pred_theta), e_i)

            if epoch % 1 == 0:
                self.save_checkpoint(epoch=epoch - 1, itr=epoch-1)

    """ Validation Routines --------------------------------------------------------------------------------------------
    """
    def _prepare_eval(self):
        self.appear_w.eval()
        self.rot_avg_finenet.eval()

    def _valid_loop(self, valid_loader, cur_train_epoch, cur_train_itr):

        avg_loss = []
        avg_acc = []

        for valid_batch_idx, valid_sample in enumerate(valid_loader):
            img_names, img_lists, img_dims, Es, Cs, Ks, out_graph, sub2id, id2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, rel = valid_sample
            pass

            if valid_batch_idx % self.train_params.MAX_VALID_BATCHES_NUM:
                break

        avg_loss = np.asarray(avg_loss)
        avg_acc = np.asarray(avg_acc)

        return {'Loss(Valid)/avg_loss': np.mean(avg_loss),
                'Accuracy(Valid)/avg_acc': np.mean(avg_acc)}