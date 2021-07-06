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
    softmax = nn.functional.softmax(X * N, dim=dim)
    return softmax

class SptLevelPropagate(nn.Module):
    def __init__(self):
        super(SptLevelPropagate, self).__init__()

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'src_ids': edges.src['node_id'], 'q': edges.src['q'], 'l': edges.src['l'], 'rel_q': edges.data['rel_q'], 'w': edges.data['w']}

    def reduce_func(self, nodes):

        neighbor_q = nodes.mailbox['q']
        neighbor_level = nodes.mailbox['l']
        neighbor_rel_q = nodes.mailbox['rel_q']
        neighbor_w = nodes.mailbox['w']
        neighbor_level = neighbor_level * neighbor_w
        neighbor_level = neighbor_level.squeeze(-1)

        cur_level = nodes.data['l']
        cur_q = nodes.data['q']
        N = cur_level.shape[0]

        accu_q = qmul(neighbor_rel_q.view(-1, 4), neighbor_q.view(-1, 4))
        accu_q = accu_q.view(N, -1, 4)
        cur_q = cur_q.view(N, -1, 4)
        cur_level = cur_level.view(N, 1)

        in_q = torch.cat([cur_q, accu_q], dim=1)
        in_l = torch.cat([cur_level, neighbor_level], dim=1)
        q_w = softmax_(in_l, N=16.0, dim=1)
        q = q_w.view(N, -1, 1) * in_q
        output_qs = torch.sum(q, dim=1)

        # output_qs = []
        # for n in range(N):
        #     output_q = weightedAverageQuaternion_(in_q[n], q_w[n])
        #     output_qs.append(output_q)
        # output_qs = torch.stack(output_qs, dim=0)

        return {'out_q': output_qs}

    def forward(self, graph: dgl.DGLGraph, node_levels, node_q, edge_rel_q, edge_w):

        N = len(graph.nodes)
        E = len(graph.edges)

        graph.ndata['node_id'] = torch.from_numpy(np.arange(0, N))
        graph.ndata['l'] = node_levels
        graph.ndata['q'] = node_q
        graph.edata['w'] = edge_w
        graph.edata['rel_q'] = edge_rel_q

        graph.update_all(self.message_func, self.reduce_func)
        q = graph.ndata.pop('out_q').view(N, 4)

        return q

    def iter(self, node_num, edge_idx, edge_rel_q, node_levels, edge_w, iter_num=4):
        E = edge_idx.shape[1]

        node_levels = node_levels.view(node_num, 1)
        edge_w = edge_w.view(E, -1).cpu()

        # build DGL graph
        edge_idx = edge_idx.long()
        graph = dgl.DGLGraph()
        graph.add_nodes(node_num)
        graph.add_edges(edge_idx[0, :].view(-1), edge_idx[1:, ].view(-1))

        in_init_q = rot2quaternion(torch.stack([torch.eye(3) for i in range(node_num)])).cpu()

        out = in_init_q
        out_levels = node_levels.cpu()
        edge_rel_q = edge_rel_q.cpu()

        for i in range(iter_num):
            out = self.forward(graph,
                               out_levels,
                               out,
                               edge_rel_q.detach(), edge_w)

        out = F.normalize(out, p=2, dim=1)
        return out

