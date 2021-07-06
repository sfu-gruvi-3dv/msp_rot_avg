import numpy as np
import socket, shutil, os, pickle, sys, torch
from core_3dv.essential_mat import *
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
# from torch_geometric.utils import scatter_
from torch_geometric.nn import GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch_scatter

def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e9 if name == 'max' else 0

    out = op(src, index, 0, None, dim_size)  # , fill_value)
    if isinstance(out, tuple):
        out = out[0]

    if name == 'max':
        out[out == fill_value] = 0

    return out


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def inv_q(q):
    """
    Inverse quaternion(s) q .
    """
    assert q.shape[-1] == 4
    original_shape = q.shape
    return torch.stack((q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]), dim=1).view(original_shape)


class EdgeConvRot(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super(EdgeConvRot, self).__init__(aggr='mean', flow="target_to_source")  # "Max" aggregation.
        # print(2 * in_channels + edge_channels)
        self.mlp = Seq(Linear(2 * in_channels + edge_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # print(x_i.shape)
        # print(x_j.shape)
        # print(edge_attr.shape)
        W = torch.cat([torch.cat([x_i, x_j], dim=1), edge_attr], dim=1)  # tmp has shape [E, 2 * in_channels]
        # print("W:", W.shape)
        W = self.mlp(W)
        return W

    def propagate(self, edge_index, size, x, edge_attr):
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        edge_out = self.message(x_i, x_j, edge_attr)
        out = scatter_(self.aggr, edge_out, edge_index[i], dim_size=size[i])
        return out, edge_out


def node_model(x, batch):
    # print(batch.shape)
    out, inverse_indices = torch.unique_consecutive(batch, return_inverse=True)
    quat_vals = x[inverse_indices]
    q_ij = qmul(x, inv_q(quat_vals[batch]))
    return q_ij


def edge_model(x, edge_index):
    row, col = edge_index
    q_ij = qmul(x[col], inv_q(x[row]))
    return q_ij


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Linear(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


class EdgePred(torch.nn.Module):
    def __init__(self, in_channels, edge_channels):
        super(EdgePred, self).__init__()
        self.mlp = Seq(Linear(2 * in_channels + edge_channels, 8),
                       ReLU(),
                       Linear(8, 1))

    def forward(self, xn, edge_index, edge_attr):
        row, col = edge_index
        xn = torch.cat([xn[row], xn[col], edge_attr], dim=1)
        xn = self.mlp(xn)
        return torch.sigmoid(xn)


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn1, nn2):
        super(GlobalSAModule, self).__init__()
        self.nn1 = nn1
        self.nn2 = nn2

    def forward(self, x, batch):
        xn = self.nn1(x)
        #  xn = F._max_pool1d(xn, x.size(1))
        # xn = scatter_('mean', xn, batch)
        # xn = xn[batch]
        xn = torch.cat([xn, x], dim=1)
        #   print(xn.shape)
        #  x = xn.unsqueeze(0).repeat(x.size(0), 1, 1)
        #  batch = torch.arange(x.size(0), device=batch.device)
        return self.nn2(xn)


def update_attr(x, edge_index, edge_attr):
    row, col = edge_index
    x_i = x[row]
    x_j = inv_q(x[col])
    #   print(x_i.shape)
    #  print(x_j.shape)
    # print(edge_attr.shape)
    W = qmul(edge_attr, x_i)
    W = qmul(x_j, W)
    return W


def smooth_l1_loss(input, beta=0.05, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])

    n = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
    #   cond = n < 5*beta
    #  n = torch.where(cond, n, 0*n)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def my_smooth_l1_loss(input, beta, alpha=0.05):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])
    nn = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
    beta = torch.squeeze(beta)
    nn = torch.mul(nn, beta)
    # print([nn.shape, beta.shape])
    # nn = nn + alpha*(torch.ones(beta.shape, dtype=torch.float, device=input.device) - beta)
    cond = nn < alpha
    loss = torch.where(cond, 0.5 * nn ** 2 / alpha, nn - 0.5 * alpha)
    return loss.mean()


def my_smooth_l1_loss_new(input, beta, edge_index, size, alpha=0.05):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])
    nn = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
    beta = torch.squeeze(beta)
    nn = torch.mul(nn, beta)

    cond = nn < alpha
    loss = torch.where(cond, 0.5 * nn ** 2 / alpha, nn - 0.5 * alpha)
    loss = scatter_('mean', loss, edge_index[0], dim_size=size(0))

    return loss.mean()

import torch.nn as nn
class Net(torch.nn.Module):
    def __init__(self, node_feat_num, edge_feat_num):
        super(Net, self).__init__()
        self.no_features = 64  # More features for large dataset
        self.conv1 = EdgeConvRot(node_feat_num, edge_feat_num, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, 704, self.no_features)
        self.conv3 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv4 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)

        self.lin1 = Linear(self.no_features, 4)
        self.edge_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

        self.m = torch.nn.Sigmoid()

    def forward(self, x_org, x_rot, edge_index, edge_attr, edge_rot):

        node_feat = torch.cat([x_org, x_rot], dim=1)
        edge_feat = torch.cat([edge_attr, edge_rot], dim=1)
        edge_index = edge_index.long()
        E = edge_index.shape[1]

        # edge_attr_mod = edge_attr  # update_attr(x_org, edge_index, edge_attr[:, :4])
        x1, edge_x1 = self.conv1(node_feat, edge_index, edge_feat)
        # x1 = F.relu(x1)
        edge_x1 = F.relu(edge_x1)
        # print("x1", x1.shape)
        x2, edge_x2 = self.conv2(x1, edge_index, torch.cat([edge_attr, edge_x1], dim=1))
        x2 = F.relu(x2)
        edge_x2 = F.relu(edge_x2)

        x3, edge_x3 = self.conv3(torch.cat([x2, x1], dim=1), edge_index, torch.cat([edge_x2, edge_x1], dim=1))
        x3 = F.relu(x3)
        edge_x3 = F.relu(edge_x3)

        x4, edge_x4 = self.conv4(torch.cat([x3, x2], dim=1), edge_index, torch.cat([edge_x3, edge_x2], dim=1))
        x4 = F.relu(x4)

        x = self.lin1(x4)
        pred_edge_res = self.edge_fc(edge_x4[:int(E/2), :] + edge_x4[int(E/2):, :])
        # print(x_org.shape)
        # x = qmul(x, x_rot)
        # x = x_rot + x
        x = F.normalize(x, p=2, dim=1)

        # x = x + x_org  # qmul(x, x_org)

        #     loss1 = inv_q(edge_model(data.y, edge_index)) - edge_model(x, edge_index)
        #         loss1 = qmul(inv_q(edge_model(data.y, edge_index)), edge_model(x, edge_index))
        #         loss1 = F.normalize(loss1, p=2, dim=1)
        #        loss1 = qmul(inv_q(edge_attr[:, :4]), edge_model(x, edge_index))
        #         loss1 = my_smooth_l1_loss_new(loss1[:, 0:], beta, edge_index, x_org.size)
        return x, pred_edge_res  # , beta   # node_model(x, batch),