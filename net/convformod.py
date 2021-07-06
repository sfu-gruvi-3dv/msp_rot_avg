import torch
import torch.nn as nn
import torch.nn.functional as F
from net.gat_base_net import GATLayer
from net.multihead_gat_net import MultiHeadGATLayer
from dgl.nn.pytorch import GATConv
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
import matplotlib.pyplot as plt
import numpy as np
# class GATConv2(GATConv):
#     def __init__(self,in_feats=512, out_feats=512, num_heads=n_head, residual=True):
#         super(GATConv2, self).__init__()
        
def forward2(self, graph, feat):
    r"""Compute graph attention network layer.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat : torch.Tensor or pair of torch.Tensor
        If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
        :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
        If a pair of torch.Tensor is given, the pair must contain two tensors of shape
        :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

    Returns
    -------
    torch.Tensor
        The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
        is the number of heads, and :math:`D_{out}` is size of output feature.
    """
    graph = graph.local_var()
    if isinstance(feat, tuple):
        h_src = self.feat_drop(feat[0])
        h_dst = self.feat_drop(feat[1])
        feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
        feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
    else:
        h_src = h_dst = self.feat_drop(feat)
        feat_src = feat_dst = self.fc(h_src).view(
            -1, self._num_heads, self._out_feats)
    # NOTE: GAT paper uses "first concatenation then linear projection"
    # to compute attention scores, while ours is "first projection then
    # addition", the two approaches are mathematically equivalent:
    # We decompose the weight vector a mentioned in the paper into
    # [a_l || a_r], then
    # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
    # Our implementation is much efficient because we do not need to
    # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
    # addition could be optimized with DGL's built-in function u_add_v,
    # which further speeds up computation and saves memory footprint.
    el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
    er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
    graph.srcdata.update({'ft': feat_src, 'el': el})
    graph.dstdata.update({'er': er})
    # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
    graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
    e = self.leaky_relu(graph.edata.pop('e'))
    # compute softmax
    attn_alpha = self.attn_drop(edge_softmax(graph, e))
    graph.edata['a'] = attn_alpha
    # message passing
    graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                     fn.sum('m', 'ft'))
    rst = graph.dstdata['ft']
    # residual
    rstbef = rst
    if self.res_fc is not None:
        resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
        rst = rst + resval
#         rst = resval
    # activation
    if self.activation:
        rst = self.activation(rst)
        rstbef = self.activation(rstbef)
    return rst, attn_alpha, rstbef
    
class MultiGATBaseConvs(nn.Module):

    def __init__(self, input_feat_channel=512):
        super(MultiGATBaseConvs, self).__init__()
        n_head = 16
        self.l1 = GATConv(in_feats=input_feat_channel, out_feats=512, num_heads=n_head, residual=True)
        self.l2 = GATConv(in_feats=n_head*512, out_feats=512, num_heads=n_head, residual=True)
        self.l3 = GATConv(in_feats=n_head*512, out_feats=512, num_heads=n_head, residual=True)
        self.l4 = GATConv(in_feats=n_head*512, out_feats=512, num_heads=1, residual=True)
        self.l1.forward = MethodType(forward2, self.l1)
        self.l2.forward = MethodType(forward2, self.l2)
        self.l3.forward = MethodType(forward2, self.l3)
        self.l4.forward = MethodType(forward2, self.l4)
        
        
#         self.l = GATConv(in_feats=[n_head, 512], out_feats=[1, 512], num_heads=1)
        
    def forward(self, graph, feat):
        N = feat.shape[0]
#         print(feat.shape)
        x,_,_ = self.l1.forward(graph, feat)
        x1 = F.relu(x)
#         print(x.shape)
        x,_,_ = self.l2.forward(graph, x1.view(N, -1))
        x = F.relu(x)
        
        x,_,_ = self.l3.forward(graph, x.view(N, -1))
        x = F.relu(x)      
        
        x, attn, bef = self.l4.forward(graph, x.view(N, -1))
        x = F.relu(x)
#         bef = F.relu(bef)
#         print(x1-x)
        
        diff = (x1 - x).detach().cpu().numpy()
        plt.hist(np.asarray(diff).ravel(), bins=100)
        plt.ylabel('Freq.')
        plt.show()
        
        return x.view(N, -1), attn, bef.view(N, -1)