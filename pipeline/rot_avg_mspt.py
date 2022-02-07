import numpy as np
import torch, os
from torch.utils.data import DataLoader
import evaluator.rotmap as iccv_rot_compare
from core_dl.train_params import TrainParameters
from exp.rot_avg_multi_propagate_trainbox import LocalGlobalVLADTrainBox
from exp.make_test_dataset import make_dataset
from core_3dv.quaternion import *
import graph_utils.utils as graph_utils
from torch.autograd.variable import Variable
from pipeline.supergraph_helper import *
import argparse
torch.manual_seed(666)


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", default=1, type=int)
parser.add_argument("--dataset",  default="./test_config/yfcc_80nodes_petra_jordan.json")
args = parser.parse_args()

""" Configuration ------------------------------------------------------------------------------------------------------
"""
enable_sigmoid_norm = False                     # enable sigmoid normalization

filter_edges_by_weight = False                  # filter edges by weight, use only for 
                                                # large-scale graphs if GPU memory is limited

selected_topk = 50                              # select top-k for MSP

max_init_itr = 10                               # max optimization iterations for initial pose 
                                                # (forward, backward MSP using Adam)

max_final_itr = 10                              # max optimization iterations for final pose 
                                                # (forward, backward MSP + FineNet using Adam)


""" Load pre-trained model ---------------------------------------------------------------------------------------------
"""
train_params = TrainParameters()
train_params.DEV_IDS = [args.gpu_id, args.gpu_id]
train_params.VERBOSE_MODE = False
box = LocalGlobalVLADTrainBox(train_params=train_params, top_k=20, ckpt_path_dict={
    'vlad': './models/netvlad_vgg16.tar',
    'ckpt': "./models/yfcc_80nodes.pth.tar"
})
box._prepare_eval()

""" Dataset ------------------------------------------------------------------------------------------------------------
"""
test_dataset_json = args.dataset

test_dataset = make_dataset(test_dataset_json, load_img=False, load_node_edge_feat=True)
dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=True)


""" Test Script -----------------------------------------------------------------------------------------------------------
"""
n_sub_graphs = len(test_dataset)

# cache the edge w, rel_q etc... ------------------------------------------------------------------------------------------
node_Es_dict = dict()
node_ori2id = dict()

edge_map2idx = dict()
edge_w, edge_sig_w = [], []
edge_rel_R = []
edge_node_idx = []
edge_meta = []

for s_i, sample in enumerate(dataloader):
    ds, idx, img_name, _, img_dims, Es, Ks, out_graph, sub2id, id2sub, _, ori2sub, \
        covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt, e_rel_err, n_f, e_f = sample
    N = out_graph.shape[1]
    E = edge_label.shape[1]

    # run AGF to get initial weight for each edge -------------------------------------------------------------------------
    with torch.no_grad():
        w, sig_w, bi_e_node_idx, bi_e_rel_q = box.forward_get_w(sample[3:], sigmoid=enable_sigmoid_norm)

    # cache node
    for n in range(N):
        idx = id2sub[n].item()
        if idx not in node_Es_dict:
            node_Es_dict[idx] = Es[0, n]

        ori_idx = ori2sub[n].item()
        node_ori2id[ori_idx] = idx

    # cache w
    w = w.detach().cpu().numpy().ravel()
    sig_w = sig_w.detach().cpu().numpy().ravel()
    if enable_sigmoid_norm:
        w = sig_w

    for ei in range(len(e_node_idx)):
        n1, n2 = e_node_idx[ei][0].item(), e_node_idx[ei][1].item()
        n1_, n2_ = id2sub[n1], id2sub[n2]

        key = (n1_.item(), n2_.item())
        if key not in edge_map2idx:
            edge_map2idx[key] = len(edge_w)
            edge_w.append([w[ei], w[ei + E]])
            edge_sig_w.append([sig_w[ei], sig_w[ei + E]])
            edge_rel_R.append(e_rel_Rt[ei].view(1, 3, 4))
            edge_node_idx.append(np.asarray((n1_.item(), n2_.item())))
            edge_meta.append({'rel_err': e_rel_err[0, ei]})
        else:
            idx = edge_map2idx[key]
            edge_w[idx] += ([w[ei], w[ei + E]])
            edge_sig_w[idx] += ([sig_w[ei], sig_w[ei + E]])

node_Es = np.zeros((len(node_Es_dict), 3, 4))
for n, E in node_Es_dict.items():
    node_Es[n] = E
node_Es = torch.from_numpy(node_Es).float()

# show statistic
N = node_Es.shape[0]
E = len(edge_w)
print('[Summary] Nodes: %d, Edges: %d' % (N, E))

# hard-code: reduce top_k for large-scale graph if gpu memory is limited
if E >= 60000:
    selected_topk = 20                          
if N >= 100000:
    selected_topk = 2

""" Generating average w -----------------------------------------------------------------------------------------------
"""
edge_avg_w = np.zeros(2 * len(edge_w))
max_e_shape = 0
w_means = []
for ei in range(len(edge_w)):
    e_w = np.asarray(edge_w[ei])
    e_sig_w = np.asarray(edge_sig_w[ei])
    e_err = edge_meta[ei]['rel_err']

    e_sig_w_mean = e_sig_w.mean()
    e_w_mean = e_w.mean()
    e_w_std = e_w.std()

    e_w_shape = e_w.shape[0]

    if e_w_shape > max_e_shape:
        max_e_shape = e_w_shape

    edge_avg_w[ei] = e_w_mean
    edge_avg_w[ei + E] = e_w_mean

w_means = np.asarray(w_means)
edge_avg_w = torch.from_numpy(edge_avg_w).float()

# build bi-direct graph
bi_e_node_idx, bi_e_rel_Rt = graph_utils.bi_direct_edge(edge_node_idx, edge_rel_R)
bi_e_rel_q = rot2quaternion(bi_e_rel_Rt).detach()
bi_e_node_idx = bi_e_node_idx.permute(1, 0).long().detach()

''' Select edge with w > 0.6 -------------------------------------------------------------------------------------------------
'''
if filter_edges_by_weight:
    sel_idx = torch.where(edge_avg_w > 0.95)[0].long()
    bi_e_node_idx = bi_e_node_idx[:, sel_idx]
    bi_e_rel_q = bi_e_rel_q[sel_idx]
    edge_avg_w = edge_avg_w[sel_idx]
    E = sel_idx.shape[0] // 2
    print('[Filtering Edges] After filtering: %d' % E)

''' Select top_k nodes -------------------------------------------------------------------------------------------------------
'''
start_nodes = graph_utils.choose_anchor_node(N, bi_e_node_idx, edge_avg_w[:2 * E], top_k=selected_topk)
ori_gt_q = rot2quaternion(node_Es[:, :3, :3])

''' Optimize the init orientation --------------------------------------------------------------------------------------------
'''
# create optimizer
with torch.cuda.device(train_params.DEV_IDS[0]):
    cur_dev = torch.cuda.current_device()
    opt_w_var = Variable(edge_avg_w.clone(), requires_grad=True)
    ref_w = edge_avg_w.clone().detach() if enable_sigmoid_norm is True else torch.sigmoid(0.6 * opt_w_var.clone()).detach()
    sub_optimzer = torch.optim.Adam([{'params': opt_w_var, 'lr': 1.5}])
    bi_e_node_idx = bi_e_node_idx.to(cur_dev)
    bi_e_rel_q = (bi_e_rel_q.to(cur_dev))
    bi_e_label = torch.ones_like(opt_w_var)
    ori_gt_q = ori_gt_q.to(cur_dev)

    pre_ref_w = None
    print('[Optimizing Initial Orientation]')
    for itr in range(max_init_itr):
        sub_optimzer.zero_grad()

        if enable_sigmoid_norm is False:
            opt_w = torch.sigmoid(0.6 * opt_w_var)          # tip: factor 0.6 makes faster convergency
        else:
            opt_w = opt_w_var

        # init ---------------------------------------------------------------------------------------------------------
        init_qs, _ = box.spt_gen.iter(N, bi_e_node_idx, bi_e_rel_q, opt_w.view(2 * E),
                                      start_nodes=start_nodes,
                                      iter_num=20, cpu_only=False)
        init_qs = init_qs.to(cur_dev)

        # optimize initial pose ----------------------------------------------------------------------------------------
        K = init_qs.shape[1]
        init_rel_qs = edge_model_batch(init_qs, bi_e_node_idx[:, :E]).view(E * K, 4)            # dim: (E, K, 4)
        e_rel_q_ext = bi_e_rel_q[:E, :].view(E, 1, 4).repeat(1, K, 1)
        rel_err_q = qmul(inv_q(init_rel_qs), e_rel_q_ext).view(E, K, 4)
        rel_err_q = F.normalize(rel_err_q, p=2, dim=-1).view(E, K, 4)
        ref_w_ext = ref_w[:E].clone().detach().view(E, 1).repeat(1, K).to(cur_dev)
        init_loss = my_smooth_l1_loss(rel_err_q.view(E * K, 4), ref_w_ext.view(E * K), alpha=0.002)
        init_loss = init_loss.view(E, K)
        init_loss = torch.mean(init_loss, dim=0).view(-1)
        
        # evaluate the predicted initial pose --------------------------------------------------------------------------
        best_id = 0                                                                             # fix best_id = 0
        choosed_start_node = start_nodes[best_id]

        init_R = quaternion2rot(init_qs[:, best_id, :].clone().detach().cpu())
        gt_R = quaternion2rot(ori_gt_q.clone().detach().cpu())
        e_mean, e_median, e_var = iccv_rot_compare.compare_rot_graph(init_R.detach().cpu(), gt_R.detach().cpu())
        print('[Init] Itr: %d: mean: %.2f median: %.2f' % (itr, e_mean, e_median))
                
        # backward the loss --------------------------------------------------------------------------------------------
        init_loss.sum().backward(retain_graph=True)
        sub_optimzer.step()

        ref_w = torch.sigmoid(0.6 * opt_w_var).clone().detach()


""" Optimizing Final Pose ----------------------------------------------------------------------------------------------
"""
print('[Optimizing Final Orientation]:')

# select best k from MSP by checking the loss
best_k_id = 0
best_k_loss = 99999.0
for k in range(init_qs.shape[1]):
    choosed_start_node = start_nodes[k]
    init_q_ = transform_q(init_qs[:, k, :], choosed_start_node)
    gt_q_ = transform_q(ori_gt_q, choosed_start_node)
    init_err = box.rot_err(init_q_.cpu(), gt_q_.cpu())

    # measure the loss
    pred_rel_q = edge_model(init_q_, bi_e_node_idx[:, :E])
    rel_err_q = qmul(inv_q(pred_rel_q), bi_e_rel_q[:E])
    rel_err_q = F.normalize(rel_err_q, p=2, dim=1)
    l1_loss = my_smooth_l1_loss_new(rel_err_q,
                                    ref_w[:E].to(cur_dev),
                                    bi_e_node_idx[:, :E], init_q_.size, alpha=0.02).mean()

    if l1_loss.item() < best_k_loss:
        best_k_loss = l1_loss.item()
        best_k_id = k

# create optimizer and optimizing FineNet
with torch.cuda.device(train_params.DEV_IDS[0]):
    cur_dev = torch.cuda.current_device()
    opt_w_var = Variable(opt_w_var.clone(), requires_grad=True)
    opt_w_var_ref = opt_w_var.clone().detach()

    sub_optimzer = torch.optim.Adam([{'params': opt_w_var, 'lr': 1.0}])

    for itr in range(max_final_itr):
        sub_optimzer.zero_grad()

        if enable_sigmoid_norm is False:
            opt_w = torch.sigmoid(0.6 * opt_w_var)
        else:
            opt_w = opt_w_var

        # init ---------------------------------------------------------------------------------------------------------
        init_qs, _ = box.spt_gen.iter(N, bi_e_node_idx, bi_e_rel_q, opt_w.view(2 * E),
                                      start_nodes=start_nodes[:1],
                                      iter_num=20, cpu_only=False)
        init_qs = init_qs.to(cur_dev)
        init_q = init_qs[:, 0, :]
        pred_q = init_q

        # measure the loss
        gt_rel_q = edge_model(ori_gt_q, bi_e_node_idx[:, :E])
        init_rel_q = edge_model(init_q, bi_e_node_idx[:, :E])
        rel_err_q = qmul(inv_q(init_rel_q), bi_e_rel_q[:E])
        rel_err_q = F.normalize(rel_err_q, p=2, dim=1)
        l1_init_loss = my_smooth_l1_loss_new(rel_err_q,
                                        ref_w[:E].to(cur_dev),
                                        bi_e_node_idx[:, :E], pred_q.size, alpha=0.05).mean()

        total_loss = 0
        pre_loss = 0
        pre_pred_q = None
        best_loss = 99999
        
        for sub_itr in range(2):
            # run FineNet twice
            in_data = [pred_q.to(cur_dev),
                       (bi_e_node_idx[0, :], bi_e_node_idx[1, :]), bi_e_rel_q.to(cur_dev), torch.zeros_like(ori_gt_q).to(cur_dev),
                       bi_e_label.to(cur_dev)]
            pred_q, _, _ = box.rot_avg_finenet(in_data)

            # measure the loss
            pred_rel_q = edge_model(pred_q, bi_e_node_idx[:, :E])
            rel_err_q = qmul(inv_q(pred_rel_q), bi_e_rel_q[:E])
            rel_err_q = F.normalize(rel_err_q, p=2, dim=1)
            l1_loss = my_smooth_l1_loss_new(rel_err_q,
                                            ref_w[:E].to(cur_dev),
                                            bi_e_node_idx[:, :E], pred_q.size, alpha=0.05).mean()

            pre_pred_q = pred_q
            pre_loss = l1_loss.item()

        # evaluate final orientation error
        final_R = quaternion2rot(pred_q.clone().detach().cpu())
        gt_R = quaternion2rot(ori_gt_q.clone().detach().cpu())
        e_mean, e_median, e_var = iccv_rot_compare.compare_rot_graph(final_R.detach().cpu(), gt_R.detach().cpu())
        print('[Final] Itr: %d: mean: %.2f median: %.2f' % (itr, e_mean, e_median))
            
        # early stop    
        total_loss = l1_loss
        pre_loss = l1_loss.item()
        if l1_loss.item() - best_loss < 0:
            best_loss = l1_loss.item()
            best_q = pred_q
        else:
            break
        
        total_loss += 0.001 * torch.norm(opt_w_var_ref - opt_w_var).mean()
        total_loss.backward(retain_graph=True)
        sub_optimzer.step()
