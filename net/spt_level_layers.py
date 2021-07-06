import torch
import torch.nn as nn
import torch.nn.functional as F
from net.FineNet import *
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import torch
import torch.nn as nn
import torch.nn.functional as F
from net.FineNet import *
from torch_geometric.nn import SAGPooling, GATConv


class QPropagate(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super(QPropagate, self).__init__(aggr='mean', flow="target_to_source")  # "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels + edge_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        W = torch.cat([torch.cat([x_i, x_j], dim=1), edge_attr], dim=1)  # tmp has shape [E, 2 * in_channels]
        W = self.mlp(W)
        return W

    def propagate(self, edge_index, size, x, edge_attr):
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        edge_out = self.message(x_i, x_j, edge_attr)
        # out = scatter_(self.aggr, edge_out, edge_index[i], dim_size=size[i])
        out = scatter(edge_out, edge_index[i], dim=0, dim_size=size[i], reduce=self.aggr)
        return out, edge_out

class LevelFusion(nn.Module):

    def __init__(self, in_node_feat, in_edge_feat, inplace=True, num_opt=4):
        super(LevelFusion, self).__init__()
        self.no_features = 128
        self.reducer = nn.Linear(in_features=in_node_feat, out_features=self.no_features)
        self.conv1 = EdgeConvRot(self.no_features, in_edge_feat, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, self.no_features, self.no_features)
        self.conv3 = EdgeConvRot(2*self.no_features, 2*self.no_features, self.no_features)
        self.conv3b = EdgeConvRot(2*self.no_features, 2*self.no_features, self.no_features)

        self.conv3_sub_pre = EdgeConvRot(self.no_features, self.no_features, self.no_features * 2)
        self.conv3_sub_pooling = SAGPooling(in_channels=2*self.no_features, GNN=GATConv)
        self.conv3_sub1 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)

        self.conv4 = EdgeConvRot(2*self.no_features, self.no_features, self.no_features)
        # self.conv5 = EdgeConvRot(self.no_features, self.no_features, self.no_features)
        self.lin1 = Linear(self.no_features, 1)
        self.w_lin = Linear(self.no_features, 1)

        self.inplace = inplace

    def forward(self, node_feat, edge_feat, edge_index):

        node_feat_reduced = self.reducer.forward(node_feat)

        x1, edge_x1 = self.conv1(node_feat_reduced, edge_index, edge_feat)
        x1 = F.relu(x1, inplace=self.inplace)
        edge_x1 = F.relu(edge_x1, inplace=self.inplace)

        x2, edge_x2 = self.conv2(x1, edge_index, edge_x1)
        x2 = F.relu(x2, inplace=self.inplace)
        edge_x2 = F.relu(edge_x2, inplace=self.inplace)

        x3, edge_x3 = self.conv3(torch.cat([x2, x1], dim=1), edge_index, torch.cat([edge_x2, edge_x1], dim=1))
        x3 = F.relu(x3, inplace=self.inplace)
        edge_x3 = F.relu(edge_x3, inplace=self.inplace)

        x3, edge_x3 = self.conv3(torch.cat([x3, x2], dim=1), edge_index, torch.cat([edge_x3, edge_x2], dim=1))
        x3 = F.relu(x3, inplace=self.inplace)
        edge_x3 = F.relu(edge_x3, inplace=self.inplace)

        # Branch sub
        x3_s1, edge_x3_s1 = self.conv3_sub_pre(x3, edge_index, edge_x3)
        x3_s1, edge_x3_s1 = F.relu(x3_s1, inplace=self.inplace), F.relu(edge_x3_s1, inplace=self.inplace)

        x3_s1_pool, edge_s1_index_pool, edge_x3_s1_pool, batch, s1_to_ori, scores = self.conv3_sub_pooling.forward(
            x3_s1, edge_index, edge_attr=edge_x3_s1)
        x3_s2, edge_x3_s2 = self.conv3_sub1(x3_s1_pool, edge_s1_index_pool, edge_x3_s1_pool)
        x3_s2, edge_x3_s2 = F.relu(x3_s2, inplace=self.inplace), F.relu(edge_x3_s2, inplace=self.inplace)
        #
        x3_2 = x3.clone()
        x3_2[s1_to_ori] = x3[s1_to_ori] + x3_s2

        x4, edge_x4 = self.conv4(torch.cat([x3_2, x3], dim=1), edge_index, edge_x3)
        x4, edge_x4 = F.relu(x4, inplace=self.inplace), F.relu(edge_x4, inplace=self.inplace)
        # x5, edge_x5 = self.conv5(x4, edge_index, edge_x4)
        # x5, edge_x5 = F.relu(x5, inplace=self.inplace), F.relu(edge_x5, inplace=self.inplace)

        x = self.lin1(x4)
        x = torch.sigmoid(x)

        w = self.w_lin(edge_x4)
        w = torch.sigmoid(w)
        return x, w

def soft_argmax(a, beta=2.0, dim=-1):
    N, D = a.shape
    prob_volume = torch.softmax(beta * a, dim=1)                    # dim: (N, D)
    soft_2d = torch.arange(0, D).float().to(a.device)
    pred_depth = torch.sum(prob_volume * soft_2d.view(1, D), dim=-1)
    return pred_depth