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
    def __init__(self):
        super(SptPropagate, self).__init__()

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'src_ids': edges.src['node_id'], 'q': edges.src['q'], 'l': edges.src['l'], 'w': edges.data['w'], 'rel_q': edges.data['rel_q']}

    def reduce_func(self, nodes):

        neighbor_q = nodes.mailbox['q']
        neighbor_level = nodes.mailbox['l']
        neighbor_w = nodes.mailbox['w']
        neighbor_rel_q = nodes.mailbox['rel_q']
        neighbor_idx = nodes.mailbox['src_ids']

        cur_idx = nodes.data['node_id']
        cur_level = nodes.data['l']
        cur_q = nodes.data['q']
        cur_w = torch.ones_like(cur_level)
        N = cur_level.shape[0]

        accu_q = qmul(neighbor_rel_q.view(-1, 4), neighbor_q.view(-1, 4))
        accu_q = accu_q.view(N, -1, 4)
        cur_q = cur_q.view(N, -1, 4)
        cur_w = cur_w.view(N, 1)
        cur_level = cur_level.view(N, 1)

        # if torch.any(cur_level == 1.0).sum().item() == 1.0:
        #     a = 4
        #     print(a)

        in_q = torch.cat([cur_q, accu_q], dim=1)
        in_w = torch.cat([cur_w, neighbor_w], dim=1)
        in_l = torch.cat([cur_level, neighbor_level], dim=1)
        # in_l_decay = torch.ones_like(in_l) * 0.8
        # in_l_decay[:, 0] = 1.0

        # compute the
        cost = in_w * in_l
        q_w = softmax_(cost, N=10.0, dim=1)

        indices_array = torch.cat([cur_idx.view(N, -1), neighbor_idx], dim=1)
        q_idx = torch.argmax(q_w, dim=1, keepdim=True)
        sel_indices = []
        for n in range(N):
            s = indices_array[n, q_idx[n][0]]
            sel_indices.append(s)
        sel_indices = torch.stack(sel_indices).view(N)

        q = q_w.view(N, -1, 1) * in_q
        output_qs = torch.sum(q, dim=1)

        # if torch.any(cur_idx == 6).sum().item() == 1.0:
        #     a = 4
        #     print(q_w)

        # output_qs = []
        # for n in range(N):
        #     output_q = weightedAverageQuaternion_(in_q[n], q_w[n])
        #     A = output_q.detach().numpy()
        #     print(np.linalg.cond(A))
        #     output_qs.append(output_q)
        # output_qs = torch.stack(output_qs, dim=0)
        output_levels = torch.sum((q_w * in_l), dim=1)

        return {'out_q': output_qs, 'out_levels': output_levels, 'sel_idx': sel_indices}

    def forward(self, graph: dgl.DGLGraph, node_levels, node_q, edge_rel_q, edge_w):

        N = node_q.shape[0]
        E = edge_rel_q.shape[0]

        graph.ndata['node_id'] = torch.from_numpy(np.arange(0, N))
        graph.ndata['l'] = node_levels
        graph.ndata['q'] = node_q
        graph.edata['w'] = edge_w
        graph.edata['rel_q'] = edge_rel_q

        graph.update_all(self.message_func, self.reduce_func)
        q = graph.ndata.pop('out_q').view(N, 4)

        # try:
        #     eig_v, eig_vec = torch.symeig(A, eigenvectors=True)
        # except Exception:
        #     print(A)
        #     print(edge_rel_q)
        q = F.normalize(q, p=2, dim=1)

        return q, graph.ndata.pop('out_levels')

    def forward_dbg(self, graph: dgl.DGLGraph, node_levels, node_q, edge_rel_q, edge_w):

        N = node_q.shape[0]
        E = edge_rel_q.shape[0]

        graph.ndata['node_id'] = torch.from_numpy(np.arange(0, N))
        graph.ndata['l'] = node_levels
        graph.ndata['q'] = node_q
        graph.edata['w'] = edge_w
        graph.edata['rel_q'] = edge_rel_q

        graph.update_all(self.message_func, self.reduce_func)
        q = graph.ndata.pop('out_q').view(N, 4)
        selected_index = graph.ndata.pop('sel_idx').view(N)

        # try:
        #     eig_v, eig_vec = torch.symeig(A, eigenvectors=True)
        # except Exception:
        #     print(A)
        #     print(edge_rel_q)
        q = F.normalize(q, p=2, dim=1)

        return q, graph.ndata.pop('out_levels'), selected_index

    def iter(self, node_num, edge_idx, edge_rel_q, edge_w, start_node, iter_num=4):

        # build node levels
        node_levels = torch.ones(node_num) * 1e-1
        node_levels[start_node] = 1.0

        # build DGL graph
        edge_idx = edge_idx.cpu().long()
        graph = dgl.DGLGraph()
        graph.add_nodes(node_num)
        graph.add_edges(edge_idx[0, :].view(-1), edge_idx[1:, ].view(-1))

        in_init_q = rot2quaternion(torch.stack([torch.eye(3) for i in range(node_num)])).cpu()

        out = in_init_q
        out_levels = node_levels.cpu()
        edge_rel_q = edge_rel_q.cpu()
        edge_w = edge_w.cpu()
        # edge_w = MyBadFn.apply(edge_w)

        for i in range(iter_num):
            out, out_levels = self.forward(graph,
                                           out_levels,
                                           out,
                                           edge_rel_q.to(edge_w.device).detach(),
                                           edge_w)

        return out, out_levels

    """ Only for debug
    """
    def prepare_iter(self, node_num, edge_idx, edge_rel_q, edge_w, start_node):
        # build node levels
        node_levels = torch.ones(node_num) * 1e-1
        node_levels[start_node] = 1.0

        # build DGL graph
        edge_idx = edge_idx.cpu().long()
        graph = dgl.DGLGraph()
        graph.add_nodes(node_num)
        graph.add_edges(edge_idx[0, :].view(-1), edge_idx[1:, ].view(-1))

        in_init_q = rot2quaternion(torch.stack([torch.eye(3) for i in range(node_num)])).cpu()

        out = in_init_q
        out_levels = node_levels.cpu()
        edge_rel_q = edge_rel_q.cpu()
        edge_w = edge_w.cpu()

        return graph, out_levels, out, edge_rel_q, edge_w