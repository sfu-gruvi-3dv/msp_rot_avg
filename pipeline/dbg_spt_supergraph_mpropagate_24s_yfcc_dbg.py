import numpy as np
from numpy.core.fromnumeric import argpartition
import torch, os, random
import sys
import pickle
sys.path.append("/mnt/Tango/pg/pg_akt_rot_avg")
from core_dl.train_params import TrainParameters
from exp.rot_avg_multi_propagate_trainbox import LocalGlobalVLADTrainBox
from exp.make_test_dataset import make_dataset
from torch.utils.data import DataLoader
from dbg.dbg_spt_visualizer import SptDbgVisualizer
import core_3dv.camera_operator as cam_opt
import core_3dv.camera_operator_gpu as cam_opt_gpu

from core_3dv.quaternion import *
import matplotlib.pyplot as plt
import graph_utils.utils as graph_utils
from graph_utils.graph_node import Node
from torch.autograd.variable import Variable
from torch_scatter import scatter
import core_math.transfom as trans
from dbg.dbg_plot import *
from dbg.supergraph_helper import *
from dbg.get_host_dbg_dir import get_dbg_dir
import torchgeometry.core.conversions as convert
import argparse
torch.manual_seed(666)

parser = argparse.ArgumentParser()
parser.add_argument("--dev", default=1, type=int)
parser.add_argument("--config",  default="./test_config/yfcc_2_180nodes_dump_feat_15.json")
parser.add_argument("--dump_init", action="store_true")
parser.add_argument("--output_file",type=str)
parser.add_argument("--name",default="big_ben_2")

args = parser.parse_args()

""" Load pre-trained model ---------------------------------------------------------------------------------------------
"""
train_params = TrainParameters()
train_params.DEV_IDS = [args.dev,args.dev]
train_params.VERBOSE_MODE = True
box = LocalGlobalVLADTrainBox(train_params=train_params, top_k=10, ckpt_path_dict={
    'vlad': '/mnt/Tango/pg/pg_akt_old/cache/netvlad_vgg16.tar',

    # Part 1:
    'ckpt': "/mnt/Exp_4/rot_avg_yfcc_logs/Oct28_21-23-16_cs-gruvi-24s/checkpoints/iter_000002.pth.tar"
})
box._prepare_eval()

""" Dataset ------------------------------------------------------------------------------------------------------------
"""
# test_json = './test_config/yfcc_2_80nodes_dump_feat_11.json'
test_json = args.config

valid_dataset = make_dataset(test_json, load_img=False, load_node_edge_feat=True)
dataloader = DataLoader(valid_dataset, batch_size=1, num_workers=0, shuffle=True)

use_pre_sigmoid = False

""" Now Test -----------------------------------------------------------------------------------------------------------
"""
n_sub_graphs = len(valid_dataset)

# cache the edge w, rel_q etc...
node_Es_dict = dict()
node_ori2id = dict()

edge_map2idx = dict()
edge_w, edge_sig_w = [], []
edge_rel_R = []
edge_node_idx = []
edge_meta = []

for s_i, sample in enumerate(dataloader):
    ds, idx, img_name, _, img_dims, Es, Ks, out_graph, sub2id, id2sub, _, ori2sub, covis_mat, e_node_idx, edge_label, e_keypt_n1, e_keypt_n2, e_rel_Rt, e_rel_err, n_f, e_f = sample
    N = out_graph.shape[1]
    E = edge_label.shape[1]

    with torch.no_grad():
        w, sig_w, bi_e_node_idx, bi_e_rel_q = box.forward_get_w(sample[3:], sigmoid=use_pre_sigmoid)

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
    if use_pre_sigmoid:
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

N = node_Es.shape[0]
E = len(edge_w)
print('Nodes: %d, Edges: %d' % (N, E))

""" Generating average w -----------------------------------------------------------------------------------------------
"""
edge_avg_w = np.zeros(2 * len(edge_w))
edge_avg_sig_w = np.zeros(2 * len(edge_w))

max_e_shape = 0
edge_gt_w = np.zeros(2 * len(edge_w))
errs = np.zeros(2 * len(edge_w))
w_means = []
for ei in range(len(edge_w)):
    e_w = np.asarray(edge_w[ei])
    e_sig_w = np.asarray(edge_sig_w[ei])

    e_err = edge_meta[ei]['rel_err']

    e_sig_w_mean = e_sig_w.mean()
    e_w_mean = e_w.mean()
    e_w_std = e_w.std()

    errs[ei] = e_err
    errs[ei + E] = e_err
    edge_gt_w[ei] = 1.0 if e_err < 20 else 0.0
    edge_gt_w[ei + E] = 1.0 if e_err < 20 else 0.0

    e_w_shape = e_w.shape[0]

    if e_w_shape > max_e_shape:
        max_e_shape = e_w_shape

    # edge_avg_w[ei] = e_w[::2].mean()
    # edge_avg_w[ei + E] = e_w[1::2].mean()

    edge_avg_w[ei] = e_w_mean
    edge_avg_w[ei + E] = e_w_mean
    edge_avg_sig_w[ei] = e_sig_w_mean
    edge_avg_sig_w[ei + E] = e_sig_w_mean

w_means = np.asarray(w_means)
edge_avg_w = torch.from_numpy(edge_avg_w).float()
edge_avg_sig_w = torch.from_numpy(edge_avg_sig_w).float()

edge_gt_w = torch.from_numpy(edge_gt_w).float()
errs = torch.from_numpy(errs).float()

# debug
if not use_pre_sigmoid:
    e = torch.sigmoid(0.6 * edge_avg_w[:E].detach()).cpu().numpy()
else:
    e = edge_avg_w[:E].detach().cpu().numpy()
err_np = errs[:E].detach().cpu().numpy()
plt.clf()
polt_w_with_rel_err_bar(e,
                        err_np,
                        save_path=os.path.join(get_dbg_dir(), 'w_wrt_rel_err.pdf'), ecolor='black')
plot_w_histo(e , err_np, thres=[80, 180], save_path=os.path.join(get_dbg_dir(), 'w_histo80180_b.pdf'))
plot_w_histo(e , err_np, thres=[40, 80], save_path=os.path.join(get_dbg_dir(), 'w_histo4080_b.pdf'))
plot_w_histo(e , err_np, thres=[0, 5], save_path=os.path.join(get_dbg_dir(), 'w_histo05_b.pdf'))

bi_e_node_idx, bi_e_rel_Rt = graph_utils.bi_direct_edge(edge_node_idx, edge_rel_R)
bi_e_rel_q = rot2quaternion(bi_e_rel_Rt).detach()
bi_e_node_idx = bi_e_node_idx.permute(1, 0).long().detach()

''' Select edge with w > 0.6 -------------------------------------------------------------------------------------------
'''
sel_idx = torch.where(edge_avg_sig_w > 0.2)[0].long()
bi_e_node_idx = bi_e_node_idx[:, sel_idx]
bi_e_rel_q = bi_e_rel_q[sel_idx]
edge_avg_w = edge_avg_w[sel_idx]
errs = errs[sel_idx]
E = sel_idx.shape[0] // 2
print('New Edges: %d' % E)

''' Select top_k nodes -------------------------------------------------------------------------------------------------
'''
selected_topk = 50
if N >= 1500:
    selected_topk = 10
if N >= 3000:
    selected_topk = 2
start_nodes = graph_utils.choose_anchor_node(N, bi_e_node_idx, edge_avg_sig_w[:2 * E], top_k=selected_topk)
ori_gt_q = rot2quaternion(node_Es[:, :3, :3])

''' Optimize the init pose ---------------------------------------------------------------------------------------------
'''
max_itr = 10

# create optimizer
with torch.cuda.device(train_params.DEV_IDS[0]):
    cur_dev = torch.cuda.current_device()
    opt_w_var = Variable(edge_avg_w.clone(), requires_grad=True)
    ref_w = edge_avg_w.clone().detach() if use_pre_sigmoid is True else torch.sigmoid(0.6 * opt_w_var.clone()).detach()
    # ref_w = edge_avg_sig_w.clone().detach()
    sub_optimzer = torch.optim.Adam([{'params': opt_w_var, 'lr': 1.5}])
    bi_e_node_idx = bi_e_node_idx.to(cur_dev)
    bi_e_rel_q = (bi_e_rel_q.to(cur_dev))
    bi_e_label = torch.ones_like(opt_w_var)
    ori_gt_q = ori_gt_q.to(cur_dev)

    pre_ref_w = None

    for itr in range(max_itr):
        sub_optimzer.zero_grad()

        if use_pre_sigmoid is False:
            opt_w = torch.sigmoid(0.6 * opt_w_var)
        else:
            opt_w = opt_w_var

        # init ---------------------------------------------------------------------------------------------------------
        init_qs, _ = box.spt_gen.iter(N, bi_e_node_idx, bi_e_rel_q, opt_w.view(2 * E),
                                      start_nodes=start_nodes,
                                      iter_num=20, cpu_only=False)
        init_qs = init_qs.to(cur_dev)

        # optimize initial pose ----------------------------------------------------------------------------------------
        K = init_qs.shape[1]
        init_rel_qs = edge_model_batch(init_qs, bi_e_node_idx[:, :E]).view(E * K, 4)  # dim: (E, K, 4)
        e_rel_q_ext = bi_e_rel_q[:E, :].view(E, 1, 4).repeat(1, K, 1)
        rel_err_q = qmul(inv_q(init_rel_qs), e_rel_q_ext).view(E, K, 4)
        rel_err_q = F.normalize(rel_err_q, p=2, dim=-1).view(E, K, 4)
        ref_w_ext = ref_w[:E].clone().detach().view(E, 1).repeat(1, K).to(cur_dev)
        init_loss = my_smooth_l1_loss(rel_err_q.view(E * K, 4), ref_w_ext.view(E * K), alpha=0.002)
        init_loss = init_loss.view(E, K)
        init_loss = torch.mean(init_loss, dim=0).view(-1)

        best_id = 0
        choosed_start_node = start_nodes[best_id]
        ori_init_q = init_qs[:, best_id, :]
        init_q = transform_q(ori_init_q, choosed_start_node)
        gt_q = transform_q(ori_gt_q, choosed_start_node)
        init_err = box.rot_err(init_q.cpu(), gt_q.cpu())
        print('Init_Itr %d: best_node:%d mean: %.2f median: %.2f' % (itr, best_id, np.mean(init_err), np.median(init_err)))

        # backward the loss --------------------------------------------------------------------------------------------
        init_loss.sum().backward(retain_graph=True)
        sub_optimzer.step()

        ref_w = torch.sigmoid(0.6 * opt_w_var).clone().detach()


    with torch.no_grad():
        sub_optimzer.zero_grad()

        if use_pre_sigmoid is False:
            opt_w = torch.sigmoid(0.6 * opt_w_var)
        else:
            opt_w = opt_w_var

        # init ---------------------------------------------------------------------------------------------------------
        init_qs, _ = box.spt_gen.iter(N, bi_e_node_idx, bi_e_rel_q, opt_w.view(2 * E),
                                      start_nodes=start_nodes,
                                      iter_num=20, cpu_only=False)
        init_qs = init_qs.to(cur_dev)

        # optimize initial pose ----------------------------------------------------------------------------------------
        K = init_qs.shape[1]
        init_rel_qs = edge_model_batch(init_qs, bi_e_node_idx[:, :E]).view(E * K, 4)  # dim: (E, K, 4)
        e_rel_q_ext = bi_e_rel_q[:E, :].view(E, 1, 4).repeat(1, K, 1)
        rel_err_q = qmul(inv_q(init_rel_qs), e_rel_q_ext).view(E, K, 4)
        rel_err_q = F.normalize(rel_err_q, p=2, dim=-1).view(E, K, 4)
        ref_w_ext = ref_w[:E].clone().detach().view(E, 1).repeat(1, K).to(cur_dev)
        init_loss = my_smooth_l1_loss(rel_err_q.view(E * K, 4), ref_w_ext.view(E * K), alpha=0.002)
        init_loss = init_loss.view(E, K)
        init_loss = torch.mean(init_loss, dim=0).view(-1)

        best_id = 0
        choosed_start_node = start_nodes[best_id]
        ori_init_q = init_qs[:, best_id, :]
        init_q = transform_q(ori_init_q, choosed_start_node)
        gt_q = transform_q(ori_gt_q, choosed_start_node)
        init_err = box.rot_err(init_q.cpu(), gt_q.cpu())
        print('Init_Itr %d: best_node:%d mean: %.2f median: %.2f' % (itr, best_id, np.mean(init_err), np.median(init_err)))

        # backward the loss --------------------------------------------------------------------------------------------
        ref_w = torch.sigmoid(0.6 * opt_w_var).clone().detach()

        output_init_qs = init_qs.detach().cpu()
        output_gt_q = gt_q.detach().cpu()
        output_bi_e_node_idx = bi_e_node_idx.detach().cpu()
        output_bi_e_rel_q = bi_e_rel_q.detach().cpu()
        output_dict = dict()
        output_dict["init_qs"] = output_init_qs
        output_dict["gt_q"] = output_gt_q
        output_dict["bi_e_node_idx"] = output_bi_e_node_idx
        output_dict["bi_e_rel_q"] = output_bi_e_rel_q
        output_dict["init_err"] = init_err
        output_dict["name"] = args.name
        output_dict["start_nodes"] = start_nodes
        output_dict["N"] = N
        output_dict["E"] = E
        if args.output_file is not None:
            with open(args.output_file, "wb") as fout:
                pickle.dump(output_dict, fout)
    

# debug
err_np = errs[:E].detach().cpu().numpy()
if not use_pre_sigmoid:
    t = torch.sigmoid(0.6 * opt_w_var[:E].detach()).cpu().numpy()
else:
    t = opt_w_var[:E].detach().cpu().numpy()

plt.clf()
polt_w_with_rel_err_bar(e,
                            err_np,
                            save_path=os.path.join(get_dbg_dir(), 'w_wrt_rel_err.pdf'), ecolor='black')
polt_w_with_rel_err_bar(t , err_np, save_path=os.path.join(get_dbg_dir(), 'w_wrt_rel_err_after.pdf'), ecolor='blue')
plot_w_histo(t , err_np, thres=[80, 180], save_path=os.path.join(get_dbg_dir(), 'w_histo80180.pdf'))
plot_w_histo(t , err_np, thres=[40, 80], save_path=os.path.join(get_dbg_dir(), 'w_histo4080.pdf'))
plot_w_histo(t , err_np, thres=[0, 5], save_path=os.path.join(get_dbg_dir(), 'w_histo05.pdf'))


""" Optimizing Final Pose ----------------------------------------------------------------------------------------------
"""
print('Start optimizing Final Pose')
max_itr = 30

# create optimizer
with torch.cuda.device(train_params.DEV_IDS[0]):
    cur_dev = torch.cuda.current_device()
    opt_w_var = Variable(opt_w_var.clone(), requires_grad=True)
    opt_w_var_ref = opt_w_var.clone().detach()

    # ref_w = edge_avg_w.clone().detach() if use_pre_sigmoid is True else torch.sigmoid(0.6 * opt_w_var.clone()).detach()

    sub_optimzer = torch.optim.Adam([{'params': opt_w_var, 'lr': 1.0}])
    gt_q = transform_q(ori_gt_q, choosed_start_node)

    for itr in range(max_itr):
        sub_optimzer.zero_grad()

        if use_pre_sigmoid is False:
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
        gt_rel_q = edge_model(gt_q, bi_e_node_idx[:, :E])
        init_rel_q = edge_model(init_q, bi_e_node_idx[:, :E])
        rel_err_q = qmul(inv_q(init_rel_q), bi_e_rel_q[:E])
        rel_err_q = F.normalize(rel_err_q, p=2, dim=1)
        init_rel_errs = box.rot_err(gt_rel_q, init_rel_q)
        l1_init_loss = my_smooth_l1_loss_new(rel_err_q,
                                        ref_w[:E].to(cur_dev),
                                        bi_e_node_idx[:, :E], pred_q.size, alpha=0.05).mean()

        init_q_ = transform_q(init_q, choosed_start_node)
        init_errs = box.rot_err(init_q_.clone().detach().cpu(), gt_q.cpu())
        print('Init_Itr %d: %.2f, %.2f, loss=%f' % (
            itr, np.mean(init_errs), np.median(init_errs), l1_init_loss.item()))

        total_loss = 0
        pre_loss = 0
        pre_pred_q = None
        for sub_itr in range(10):
            in_data = [pred_q.to(cur_dev),
                       (bi_e_node_idx[0, :], bi_e_node_idx[1, :]), bi_e_rel_q.to(cur_dev), gt_q.to(cur_dev),
                       bi_e_label.to(cur_dev)]
            pred_q, _, _ = box.rot_avg_finenet(in_data)

            # measure the loss
            pred_rel_q = edge_model(pred_q, bi_e_node_idx[:, :E])
            rel_err_q = qmul(inv_q(pred_rel_q), bi_e_rel_q[:E])
            rel_err_q = F.normalize(rel_err_q, p=2, dim=1)
            l1_loss = my_smooth_l1_loss_new(rel_err_q,
                                            ref_w[:E].to(cur_dev),
                                            bi_e_node_idx[:, :E], pred_q.size, alpha=0.05).mean()

            # determine if the iteration has to stop
            # diff_loss = pre_loss - l1_loss.item()
            # if (pre_loss != 0 and diff_loss < 0) or abs(diff_loss) < 1e-4:
            #     pred_q = pre_pred_q
            #     break

            rel_q_errs = box.rot_err(gt_rel_q, pred_rel_q)
            pred_q_ = transform_q(pred_q.clone(), choosed_start_node)
            pred_errs = box.rot_err(pred_q_.clone().detach().cpu(), gt_q.cpu())
            print('- Pred_Itr %d: Sub_itr%d: %.2f, %.2f, loss=%f' % (
            itr, sub_itr, np.mean(pred_errs), np.median(pred_errs), l1_loss.item()))

            pre_pred_q = pred_q
            pre_loss = l1_loss.item()

        # Final Q
        pred_q_ = transform_q(pred_q.clone(), choosed_start_node)
        pred_errs = box.rot_err(pred_q_.clone().detach().cpu(), gt_q.cpu())
        print('Final_Pred_Itr %d: %.2f, %.2f, loss=%f' % (
            itr, np.mean(pred_errs), np.median(pred_errs), l1_loss.item()))
        
        Ebest = iccv_rot_compare.compare_rot_graph(init_R.detach().cpu(), gt_R.detach().cpu())
        print(Ebest)
        
        total_loss = l1_loss
        # total_loss += 0.01 * torch.norm(opt_w_var_ref - opt_w_var).mean()
        total_loss.backward(retain_graph=True)
        sub_optimzer.step()

exit(0)
