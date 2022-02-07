import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch_geometric
import numpy as np
from core_3dv.quaternion import *


def weightedAverageQuaternion_(Q, w):
    w = w.view(-1)
    M = Q.shape[0]
    weightSum = torch.sum(w) + 1e-5

    A = torch.cat([(w[i] * torch.ger(Q[i, :], Q[i, :])).view(1, -1) for i in range(M)], dim=0)
    A = torch.sum(A, dim=0).view(4, 4)
    A = (1.0 / weightSum) * A

    return A

def softmax_(X: torch.Tensor, N=8.0, dim=-1):
    # exp_x = torch.exp(X * N)
    # Z = torch.sum(exp_x, dim=dim) + 1e-5
    # softmax = exp_x / Z
    softmax = nn.functional.softmax(X * N, dim=dim)

    return softmax

# torch.autograd.set_detect_anomaly(True)

class MyBadFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return inp

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        res = grad_out
        return res



class SptPropagate(nn.Module):
    def __init__(self, top_k):
        super(SptPropagate, self).__init__()
        self.top_k = top_k

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'src_ids': edges.src['node_id'], 'q': edges.src['q'], 'l': edges.src['l'], 'w': edges.data['w'], 'rel_q': edges.data['rel_q']}

    def reduce_func(self, nodes):

        # assume we have M neighbors in N nodes:
        neighbor_q = nodes.mailbox['q']                             # dim: (N, M, K*4)
        neighbor_level = nodes.mailbox['l']                         # dim: (N, M, K)
        neighbor_w = nodes.mailbox['w']                             # dim: (N, M)
        neighbor_rel_q = nodes.mailbox['rel_q']                     # dim: (N, M, 4)
        # neighbor_idx = nodes.mailbox['src_ids']                     # dim: (N, M)

        N = neighbor_w.shape[0]
        M = neighbor_w.shape[1]
        K = neighbor_level.shape[-1]
        neighbor_q = neighbor_q.view(N, M, K, 4)

        # cur_idx = nodes.data['node_id']                           # dim: (N)
        cur_level = nodes.data['l']                                 # dim: (N, K)
        cur_q = nodes.data['q']                                     # dim: (N, K, 4)
        cur_w = torch.ones((N, 1)).to(cur_level.device)             # dim: (N, 1)

        # todo: check
        accu_qs = qmul(neighbor_rel_q.view(N, M, 1, 4).expand(N, M, K, 4).contiguous(), neighbor_q.view(-1, 4))
        accu_qs = accu_qs.view(N, M, K, 4)

        # accu_qs = []
        # for k in range(K):
        #     accu_q = qmul(neighbor_rel_q.view(-1, 4), neighbor_q[:, :, k, :].view(-1, 4))           # dim: (N, M, 4)
        #     accu_qs.append(accu_q.view(N, M, -1, 4))
        # accu_qs = torch.cat(accu_qs, dim=2)

        cur_q = cur_q.view(N, -1, K, 4)                                # dim: (N, M, K, 4)
        cur_w = cur_w.view(N, 1)
        cur_level = cur_level.view(N, -1, K)

        in_q = torch.cat([cur_q, accu_qs], dim=1)                    # dim: (N, M + 1, K*4)
        in_w = torch.cat([cur_w, neighbor_w], dim=1)                 # dim: (N, M + 1)
        in_l = torch.cat([cur_level, neighbor_level], dim=1)         # dim: (N, M + 1, K)

        # compute the normalized w
        cost = in_w.view(N, M + 1, 1) * in_l
        cost = cost.permute(0, 2, 1)                                 # dim: (N, K, M + 1)
        q_w = softmax_(cost, N=8.0, dim=-1)                         # dim: (N, K, M + 1)

        in_q = in_q.view(N, M + 1, K, 4).permute(0, 2, 1, 3)         # dim: (N, K, M + 1, 4)
        q = q_w.view(N, K, M + 1, 1) * in_q
        output_qs = torch.sum(q, dim=2)                              # dim: (N, K, 4)

        in_l = in_l.permute(0, 2, 1).view(N, K, M + 1)
        output_levels = torch.sum((q_w * in_l), dim=-1)              # dim: (N, K)

        return {'out_q': output_qs, 'out_levels': output_levels}

    def forward(self, graph: dgl.DGLGraph, node_levels, node_q, edge_rel_q, edge_w):
        """

        Args:
            graph:
            node_levels: node levels, dim: (N, K)
            node_q: dim: (N, K, 4)
            edge_rel_q: dim: (E, 4)
            edge_w: dim: (E)

        Returns:

        """
        cur_dev = node_levels.device
        N = node_levels.shape[0]
        K = node_levels.shape[1]
        E = edge_rel_q.shape[0]

        graph.ndata['node_id'] = torch.from_numpy(np.arange(0, N)).to(cur_dev)
        graph.ndata['l'] = node_levels.view(N, -1)                                                    # dim: (N, K)
        graph.ndata['q'] = node_q.view(N, -1)                                                         # dim: (N, K * 4)
        graph.edata['w'] = edge_w                                                                     # dim: (E)
        graph.edata['rel_q'] = edge_rel_q                                                             # dim: (E, 4)

        graph.update_all(self.message_func, self.reduce_func)
        q = graph.ndata.pop('out_q').view(N*K, 4)
        q = F.normalize(q, p=2, dim=1).view(N, K, 4)
        out_levels = graph.ndata.pop('out_levels').view(N, K)

        return q, out_levels

    def iter(self, node_num, edge_idx, edge_rel_q, edge_w, start_nodes:np.ndarray, iter_num=4, cpu_only=False):

        cur_dev = torch.cuda.current_device()
        if cpu_only:
            cur_dev = 'cpu'

        K = start_nodes.shape[0]
        N = node_num

        # build node levels
        node_levels = torch.ones(node_num, K) * 1e-1                                            # dim: (N, K)
        for k in range(K):
            start_node = start_nodes[k]
            node_levels[start_node, k] = 1.0

        # build DGL graph
        edge_idx = edge_idx.cpu().long().to(cur_dev)
        graph = dgl.DGLGraph().to(cur_dev)
        graph.add_nodes(node_num)
        graph.add_edges(edge_idx[0, :].view(-1), edge_idx[1:, ].view(-1))

        # build init pose
        in_init_q = rot2quaternion(torch.stack([torch.eye(3) for i in range(node_num)])).cpu()      # dim: (N, 4)
        in_init_qs = in_init_q.view(N, 1, 4).expand(-1, K, -1).contiguous().to(cur_dev)             # dim: (N, K, 4)

        out = in_init_qs
        out_levels = node_levels.to(cur_dev)
        edge_rel_q = edge_rel_q.to(cur_dev)
        edge_w = edge_w.to(cur_dev)

        pre_levels = None
        for i in range(iter_num):
            out, out_levels = self.forward(graph,
                                           out_levels,
                                           out,
                                           edge_rel_q.to(edge_w.device).detach(),
                                           edge_w)
            # early stop
            if pre_levels is None:
                pre_levels = out_levels
            else:
                diff = torch.norm(pre_levels - out_levels).mean()
                # print('Propagate Iter:%d, diff=%f' % (i, diff.item()))
                if diff < 1e-1:
                    break
                pre_levels = out_levels

        return out, out_levels

    # """ Only for debug
    # """
    # def prepare_iter(self, node_num, edge_idx, edge_rel_q, edge_w, start_node):
    #     # build node levels
    #     node_levels = torch.ones(node_num) * 1e-1
    #     node_levels[start_node] = 1.0
    #
    #     # build DGL graph
    #     edge_idx = edge_idx.cpu().long()
    #     graph = dgl.DGLGraph()
    #     graph.add_nodes(node_num)
    #     graph.add_edges(edge_idx[0, :].view(-1), edge_idx[1:, ].view(-1))
    #     in_init_q = rot2quaternion(torch.stack([torch.eye(3) for i in range(node_num)])).cpu()
    #
    #     out = in_init_q
    #     out_levels = node_levels.cpu()
    #     edge_rel_q = edge_rel_q.cpu()
    #     edge_w = edge_w.cpu()
    #
    #     return graph, out_levels, out, edge_rel_q, edge_w