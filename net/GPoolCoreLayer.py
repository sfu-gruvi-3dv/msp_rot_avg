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

class PoolingFineNetWithAppearance(torch.nn.Module):

    def __init__(self, in_node_feat, in_edge_feat):
        super(PoolingFineNetWithAppearance, self).__init__()
        self.no_features = 64  # More features for large dataset
        self.conv1 = EdgeConvRot(in_node_feat + 4, in_edge_feat + 4, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, self.no_features + 4, self.no_features)
        self.conv3 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)

        self.conv3_sub_pre = EdgeConvRot(self.no_features, self.no_features, self.no_features * 2)
        self.conv3_sub_pooling = SAGPooling(in_channels=2*self.no_features, GNN=GATConv)
        self.conv3_sub1 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features*2)
        self.conv3_sub2 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)

        # self.conv3_sub_pooling = SAGPooling(in_channels=self.no_features, GNN=GATConv)
        # self.conv3_sub_pre = EdgeConvRot(self.no_features, self.no_features, self.no_features * 2)
        self.conv3_subsub_pooling = SAGPooling(in_channels=2*self.no_features, GNN=GATConv)
        self.conv3_subsub = EdgeConvRot(2 * self.no_features, 2 * self.no_features, 2*self.no_features)
        self.conv3_subsub2 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, 2*self.no_features)

        self.conv4 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv5 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.lin1 = Linear(self.no_features, 4)
        self.lin2 = Linear(self.no_features, 1)

        self.m = torch.nn.Sigmoid()

    def forward(self, data):
        x_org, edge_index, edge_attr, gt_q, beta, node_feat, edge_feat = data
        N = x_org.shape[0]
        E = edge_attr.shape[0]

        edge_attr_mod = update_attr(x_org, edge_index, edge_attr[:, :4])
        edge_feat = edge_feat.view(E, -1)
        node_feat = node_feat.view(N, -1)

        x1, edge_x1 = self.conv1(torch.cat([node_feat, x_org], dim=1), edge_index, torch.cat([edge_feat, edge_attr_mod], dim=1))
        x1 = F.relu(x1)
        edge_x1 = F.relu(edge_x1)

        x2, edge_x2 = self.conv2(x1, edge_index, torch.cat([edge_attr_mod, edge_x1], dim=1))
        x2 = F.relu(x2)
        edge_x2 = F.relu(edge_x2)

        x3, edge_x3 = self.conv3(torch.cat([x2, x1], dim=1), edge_index, torch.cat([edge_x2, edge_x1], dim=1))
        x3 = F.relu(x3)
        edge_x3 = F.relu(edge_x3)

        # Branch sub
        x3_s1, edge_x3_s1 = self.conv3_sub_pre(x3, edge_index, edge_x3)
        x3_s1, edge_x3_s1 = F.relu(x3_s1), F.relu(edge_x3_s1)

        x3_s1_pool, edge_s1_index_pool, edge_x3_s1_pool, batch, s1_to_ori, scores = self.conv3_sub_pooling.forward(x3_s1, edge_index, edge_attr=edge_x3_s1)
        x3_s2, edge_x3_s2 = self.conv3_sub1(x3_s1_pool, edge_s1_index_pool, edge_x3_s1_pool)
        x3_s2, edge_x3_s2 = F.relu(x3_s2), F.relu(edge_x3_s2)

        # Branch x3b
        x3_ss_pool, edge_ss_index_pool, edge_x3_ss_pool, batch1, ss_to_s1, _ = self.conv3_subsub_pooling.forward(x3_s2, edge_s1_index_pool, edge_attr=edge_x3_s2)
        x3_ss1, edge_x3_ss1 = self.conv3_subsub.forward(x3_ss_pool, edge_ss_index_pool, edge_x3_ss_pool)
        x3_ss1, edge_x3_ss1 = F.relu(x3_ss1), F.relu(edge_x3_ss1)

        x3_ss2, edge_x3_ss2 = self.conv3_subsub.forward(x3_ss1, edge_ss_index_pool, edge_x3_ss1)
        x3_ss2, edge_x3_ss2 = F.relu(x3_ss2), F.relu(edge_x3_ss2)
        x3_s2_ = x3_s2.clone()
        x3_s2_[ss_to_s1, :] = x3_s2_[ss_to_s1, :] + x3_ss2

        x3_s3, edge_x3_s3 = self.conv3_sub2(x3_s2_, edge_s1_index_pool, edge_x3_s2)
        x3_s3, edge_x3_s3 = F.relu(x3_s3), F.relu(edge_x3_s3)
        x3[s1_to_ori] = x3[s1_to_ori] + x3_s3

        # End route
        x4, edge_x4 = self.conv4(torch.cat([x3, x2], dim=1), edge_index, torch.cat([edge_x3, edge_x2], dim=1))
        x4, edge_x4 = F.relu(x4), F.relu(edge_x4)
        x5, edge_x5 = self.conv4(torch.cat([x4, x3], dim=1), edge_index, torch.cat([edge_x4, edge_x3], dim=1))
        x5, edge_x5 = F.relu(x5), F.relu(edge_x5)

        x = self.lin1(x5)
        # x = x + x_org  # qmul(x, x_org)
        x = qmul(x, x_org)
        x = F.normalize(x, p=2, dim=1)

        out_res = self.lin2(edge_x5)

        #     loss1 = inv_q(edge_model(data.y, edge_index)) - edge_model(x, edge_index)
        loss1 = qmul(inv_q(edge_model(gt_q, edge_index)), edge_model(x, edge_index))
        loss1 = F.normalize(loss1, p=2, dim=1)
        #        loss1 = qmul(inv_q(edge_attr[:, :4]), edge_model(x, edge_index))
        loss1 = my_smooth_l1_loss_new(loss1[:, 0:], beta, edge_index, x_org.size)

        return x, loss1, beta, out_res, (x1, x2, x3, x4, x5), (edge_x1, edge_x2, edge_x3, edge_x4, edge_x5)  # node_model(x, batch),


class PoolingFineNet(torch.nn.Module):

    def __init__(self):
        super(PoolingFineNet, self).__init__()
        self.no_features = 32  # More features for large dataset
        self.conv1 = EdgeConvRot(4, 4, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, self.no_features + 4, self.no_features)
        self.conv3 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)

        self.conv3_sub_pre = EdgeConvRot(self.no_features, self.no_features, self.no_features * 2)
        self.conv3_sub_pooling = SAGPooling(in_channels=2*self.no_features, GNN=GATConv)
        self.conv3_sub1 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features*2)
        self.conv3_sub2 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)

        # self.conv3_sub_pooling = SAGPooling(in_channels=self.no_features, GNN=GATConv)
        # self.conv3_sub_pre = EdgeConvRot(self.no_features, self.no_features, self.no_features * 2)
        self.conv3_subsub_pooling = SAGPooling(in_channels=2*self.no_features, GNN=GATConv)
        self.conv3_subsub = EdgeConvRot(2 * self.no_features, 2 * self.no_features, 2*self.no_features)
        self.conv3_subsub2 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, 2*self.no_features)

        self.conv4 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv5 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.lin1 = Linear(self.no_features, 4)

        self.m = torch.nn.Sigmoid()

    def forward(self, data):
        x_org, edge_index, edge_attr, gt_q, beta = data
        edge_attr_mod = update_attr(x_org, edge_index, edge_attr[:, :4])

        x1, edge_x1 = self.conv1(x_org, edge_index, edge_attr_mod)
        x1 = F.relu(x1)
        edge_x1 = F.relu(edge_x1)

        x2, edge_x2 = self.conv2(x1, edge_index, torch.cat([edge_attr_mod, edge_x1], dim=1))
        x2 = F.relu(x2)
        edge_x2 = F.relu(edge_x2)

        x3, edge_x3 = self.conv3(torch.cat([x2, x1], dim=1), edge_index, torch.cat([edge_x2, edge_x1], dim=1))
        x3 = F.relu(x3)
        edge_x3 = F.relu(edge_x3)

        # Branch sub
        x3_s1, edge_x3_s1 = self.conv3_sub_pre(x3, edge_index, edge_x3)
        x3_s1, edge_x3_s1 = F.relu(x3_s1), F.relu(edge_x3_s1)

        x3_s1_pool, edge_s1_index_pool, edge_x3_s1_pool, batch, s1_to_ori, scores = self.conv3_sub_pooling.forward(x3_s1, edge_index, edge_attr=edge_x3_s1)
        x3_s2, edge_x3_s2 = self.conv3_sub1(x3_s1_pool, edge_s1_index_pool, edge_x3_s1_pool)
        x3_s2, edge_x3_s2 = F.relu(x3_s2), F.relu(edge_x3_s2)

        # Branch x3b
        x3_ss_pool, edge_ss_index_pool, edge_x3_ss_pool, batch1, ss_to_s1, _ = self.conv3_subsub_pooling.forward(x3_s2, edge_s1_index_pool, edge_attr=edge_x3_s2)
        x3_ss1, edge_x3_ss1 = self.conv3_subsub.forward(x3_ss_pool, edge_ss_index_pool, edge_x3_ss_pool)
        x3_ss1, edge_x3_ss1 = F.relu(x3_ss1), F.relu(edge_x3_ss1)

        x3_ss2, edge_x3_ss2 = self.conv3_subsub.forward(x3_ss1, edge_ss_index_pool, edge_x3_ss1)
        x3_ss2, edge_x3_ss2 = F.relu(x3_ss2), F.relu(edge_x3_ss2)
        x3_s2_ = x3_s2.clone()
        x3_s2_[ss_to_s1, :] = x3_s2_[ss_to_s1, :] + x3_ss2

        x3_s3, edge_x3_s3 = self.conv3_sub2(x3_s2_, edge_s1_index_pool, edge_x3_s2)
        x3_s3, edge_x3_s3 = F.relu(x3_s3), F.relu(edge_x3_s3)
        x3[s1_to_ori] = x3[s1_to_ori] + x3_s3

        # End route
        x4, edge_x4 = self.conv4(torch.cat([x3, x2], dim=1), edge_index, torch.cat([edge_x3, edge_x2], dim=1))
        x4, edge_x4 = F.relu(x4), F.relu(edge_x4)
        x5, edge_x5 = self.conv4(torch.cat([x4, x3], dim=1), edge_index, torch.cat([edge_x4, edge_x3], dim=1))
        x5, edge_x5 = F.relu(x5), F.relu(edge_x5)

        x = self.lin1(x5)
        # x = x + x_org  # qmul(x, x_org)
        x = qmul(x, x_org)

        x = F.normalize(x, p=2, dim=1)
        #     loss1 = inv_q(edge_model(data.y, edge_index)) - edge_model(x, edge_index)
        loss1 = qmul(inv_q(edge_model(gt_q, edge_index)), edge_model(x, edge_index))
        loss1 = F.normalize(loss1, p=2, dim=1)
        #        loss1 = qmul(inv_q(edge_attr[:, :4]), edge_model(x, edge_index))
        loss1 = my_smooth_l1_loss_new(loss1[:, 0:], beta, edge_index, x_org.size)

        return x, loss1, beta, (x1, x2, x3, x4, x5), (edge_x1, edge_x2, edge_x3, edge_x4, edge_x5)  # node_model(x, batch),

class AppearancePoolFusion(nn.Module):

    def __init__(self, in_node_feat, in_edge_feat, inplace=True, num_opt=4):
        super(AppearancePoolFusion, self).__init__()
        self.no_features = 128
        self.conv1 = EdgeConvRot(in_node_feat, in_edge_feat, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, self.no_features, self.no_features)
        self.conv3 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)

        self.conv3_sub_pre = EdgeConvRot(self.no_features, self.no_features, self.no_features * 2)
        self.conv3_sub_pooling = SAGPooling(in_channels=2*self.no_features, GNN=GATConv)
        self.conv3_sub1 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features*2)
        self.conv3_sub2 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)

        # self.conv3_sub_pooling = SAGPooling(in_channels=self.no_features, GNN=GATConv)
        # self.conv3_sub_pre = EdgeConvRot(self.no_features, self.no_features, self.no_features * 2)
        self.conv3_subsub_pooling = SAGPooling(in_channels=2*self.no_features, GNN=GATConv)
        self.conv3_subsub = EdgeConvRot(2 * self.no_features, 2 * self.no_features, 2*self.no_features)
        self.conv3_subsub2 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, 2*self.no_features)

        self.conv4 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv5 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.lin1 = Linear(self.no_features, num_opt)

        self.inplace = inplace

    def forward(self, node_feat, edge_index, edge_feat):

        x1, edge_x1 = self.conv1(node_feat, edge_index, edge_feat)
        x1 = F.relu(x1, inplace=self.inplace)
        edge_x1 = F.relu(edge_x1, inplace=self.inplace)

        x2, edge_x2 = self.conv2(x1, edge_index, edge_x1)
        x2 = F.relu(x2, inplace=self.inplace)
        edge_x2 = F.relu(edge_x2, inplace=self.inplace)

        x3, edge_x3 = self.conv3(torch.cat([x2, x1], dim=1), edge_index, torch.cat([edge_x2, edge_x1], dim=1))
        x3 = F.relu(x3, inplace=self.inplace)
        edge_x3 = F.relu(edge_x3, inplace=self.inplace)

        # Branch sub
        x3_s1, edge_x3_s1 = self.conv3_sub_pre(x3, edge_index, edge_x3)
        x3_s1, edge_x3_s1 = F.relu(x3_s1, inplace=self.inplace), F.relu(edge_x3_s1, inplace=self.inplace)

        x3_s1_pool, edge_s1_index_pool, edge_x3_s1_pool, batch, s1_to_ori, scores = self.conv3_sub_pooling.forward(
            x3_s1, edge_index, edge_attr=edge_x3_s1)
        x3_s2, edge_x3_s2 = self.conv3_sub1(x3_s1_pool, edge_s1_index_pool, edge_x3_s1_pool)
        x3_s2, edge_x3_s2 = F.relu(x3_s2, inplace=self.inplace), F.relu(edge_x3_s2, inplace=self.inplace)

        # Branch x3b
        x3_ss_pool, edge_ss_index_pool, edge_x3_ss_pool, batch1, ss_to_s1, _ = self.conv3_subsub_pooling(x3_s2,
                                                                                                         edge_s1_index_pool,
                                                                                                         edge_attr=edge_x3_s2)
        x3_ss1, edge_x3_ss1 = self.conv3_subsub.forward(x3_ss_pool, edge_ss_index_pool, edge_x3_ss_pool)
        x3_ss1, edge_x3_ss1 = F.relu(x3_ss1, inplace=self.inplace), F.relu(edge_x3_ss1, inplace=self.inplace)
        x3_ss2, edge_x3_ss2 = self.conv3_subsub.forward(x3_ss1, edge_ss_index_pool, edge_x3_ss1)
        x3_ss2, edge_x3_ss2 = F.relu(x3_ss2, inplace=self.inplace), F.relu(edge_x3_ss2, inplace=self.inplace)
        x3_s2_2 = x3_s2.clone()
        x3_s2_2[ss_to_s1] = x3_s2[ss_to_s1] + x3_ss2

        x3_s3, edge_x3_s3 = self.conv3_sub2(x3_s2_2, edge_s1_index_pool, edge_x3_s2)
        x3_s3, edge_x3_s3 = F.relu(x3_s3, inplace=self.inplace), F.relu(edge_x3_s3, inplace=self.inplace)
        # x3_s3_global = gmp(x3_s3, batch)
        x3_2 = x3.clone()
        x3_2[s1_to_ori] = x3[s1_to_ori] + x3_s3

        x4, edge_x4 = self.conv4(torch.cat([x3_2, x2], dim=1), edge_index, torch.cat([edge_x3, edge_x2], dim=1))
        x4, edge_x4 = F.relu(x4, inplace=self.inplace), F.relu(edge_x4, inplace=self.inplace)
        x5, edge_x5 = self.conv4(torch.cat([x4, x3_2], dim=1), edge_index, torch.cat([edge_x4, edge_x3], dim=1))
        x5, edge_x5 = F.relu(x5, inplace=self.inplace), F.relu(edge_x5, inplace=self.inplace)

        x = self.lin1(x5)

        return x

class AppearanceFusion(nn.Module):

    def __init__(self, in_node_feat, in_edge_feat):
        super(AppearanceFusion, self).__init__()
        self.no_features = 128
        self.conv1 = EdgeConvRot(in_node_feat, in_edge_feat, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, self.no_features, self.no_features)
        self.conv3 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv4 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv5 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv6 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.lin1 = Linear(self.no_features, 1)

    def forward(self, node_feat, edge_feat, edge_idx):

        x1, edge_x1 = self.conv1(node_feat, edge_idx, edge_feat)
        x1 = F.relu(x1)
        edge_x1 = F.relu(edge_x1)

        x2, edge_x2 = self.conv2(x1, edge_idx, edge_x1)
        x2 = F.relu(x2)
        edge_x2 = F.relu(edge_x2)

        x3, edge_x3 = self.conv3(torch.cat([x2, x1], dim=1), edge_idx, torch.cat([edge_x2, edge_x1], dim=1))
        x3 = F.relu(x3)
        edge_x3 = F.relu(edge_x3)

        x4, edge_x4 = self.conv4(torch.cat([x3, x2], dim=1), edge_idx, torch.cat([edge_x3, edge_x2], dim=1))
        x4 = F.relu(x4)
        edge_x4 = F.relu(edge_x4)

        x5, edge_x5 = self.conv5(torch.cat([x4, x3], dim=1), edge_idx, torch.cat([edge_x4, edge_x3], dim=1))
        x5 = F.relu(x5)
        edge_x5 = F.relu(edge_x5)

        x6, edge_x6 = self.conv6(torch.cat([x5, x4], dim=1), edge_idx, torch.cat([edge_x5, edge_x4], dim=1))
        x6 = F.relu(x6)
        edge_x6 = F.relu(edge_x6)

        out = self.lin1(edge_x6)

        return out, x6, edge_x6

class AppearanceFusionVAE(nn.Module):

    def __init__(self, in_node_feat, in_edge_feat):
        super(AppearanceFusionVAE, self).__init__()
        self.no_features = 128
        self.conv1 = EdgeConvRot(in_node_feat, in_edge_feat, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, self.no_features, self.no_features)
        self.conv3 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv4 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv5 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv6 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.lin1 = Linear(self.no_features, 2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, node_feat, edge_feat, edge_idx):

        x1, edge_x1 = self.conv1(node_feat, edge_idx, edge_feat)
        x1 = F.relu(x1)
        edge_x1 = F.relu(edge_x1)

        x2, edge_x2 = self.conv2(x1, edge_idx, edge_x1)
        x2 = F.relu(x2)
        edge_x2 = F.relu(edge_x2)

        x3, edge_x3 = self.conv3(torch.cat([x2, x1], dim=1), edge_idx, torch.cat([edge_x2, edge_x1], dim=1))
        x3 = F.relu(x3)
        edge_x3 = F.relu(edge_x3)

        x4, edge_x4 = self.conv4(torch.cat([x3, x2], dim=1), edge_idx, torch.cat([edge_x3, edge_x2], dim=1))
        x4 = F.relu(x4)
        edge_x4 = F.relu(edge_x4)

        x5, edge_x5 = self.conv5(torch.cat([x4, x3], dim=1), edge_idx, torch.cat([edge_x4, edge_x3], dim=1))
        x5 = F.relu(x5)
        edge_x5 = F.relu(edge_x5)

        x6, edge_x6 = self.conv6(torch.cat([x5, x4], dim=1), edge_idx, torch.cat([edge_x5, edge_x4], dim=1))
        x6 = F.relu(x6)
        edge_x6 = F.relu(edge_x6)

        out = self.lin1(edge_x6)

        return out[:, 0], out[:, 1]

class AppearanceFineNet(nn.Module):

    def __init__(self, in_channel):
        super(AppearanceFineNet, self).__init__()
        self.no_features = 128  # More features for large dataset
        self.conv1 = EdgeConvRot(4 + in_channel, 4 + in_channel, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, self.no_features + 4, self.no_features)
        self.conv3 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv4 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.lin1 = Linear(self.no_features, 4)
        self.lin2 = Linear(self.no_features, 1)

    def forward(self, data):
        x_org, edge_index, edge_attr, gt_q, beta, node_feat, edge_feat = data

        N = x_org.shape[0]
        E = edge_attr.shape[0]

        edge_attr_mod = update_attr(x_org, edge_index, edge_attr[:, :4])
        edge_feat = edge_feat.view(E, -1)
        node_feat = node_feat.view(N, -1)

        x1, edge_x1 = self.conv1(torch.cat([node_feat, x_org], dim=1), edge_index, torch.cat([edge_feat, edge_attr_mod], dim=1))
        x1 = F.relu(x1)
        edge_x1 = F.relu(edge_x1)

        x2, edge_x2 = self.conv2(x1, edge_index, torch.cat([edge_attr_mod, edge_x1], dim=1))
        x2 = F.relu(x2)
        edge_x2 = F.relu(edge_x2)

        x3, edge_x3 = self.conv3(torch.cat([x2, x1], dim=1), edge_index, torch.cat([edge_x2, edge_x1], dim=1))
        x3 = F.relu(x3)
        edge_x3 = F.relu(edge_x3)

        x4, edge_x4 = self.conv4(torch.cat([x3, x2], dim=1), edge_index, torch.cat([edge_x3, edge_x2], dim=1))
        x4 = F.relu(x4)
        edge_x4 = F.relu(edge_x4)

        x = self.lin1(x4)
        # x = x + x_org
        x = qmul(x, x_org)

        x = F.normalize(x, p=2, dim=1)
        #     loss1 = inv_q(edge_model(data.y, edge_index)) - edge_model(x, edge_index)
        loss1 = qmul(inv_q(edge_model(gt_q, edge_index)), edge_model(x, edge_index))
        loss1 = F.normalize(loss1, p=2, dim=1)
        #        loss1 = qmul(inv_q(edge_attr[:, :4]), edge_model(x, edge_index))
        loss1 = my_smooth_l1_loss_new(loss1[:, 0:], beta, edge_index, x_org.size)

        out_res = self.lin2(edge_x4)

        return x, loss1, beta, out_res, (x1, x2, x3, x4), (edge_x1, edge_x2, edge_x3, edge_x4)  # node_model(x, batch),

""" 
"""
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

def update_attr_batch(x, edge_index, edge_attr):
    """

    Args:
        x:  dim: (N, K, 4)
        edge_index:
        edge_attr: dim: (E, 4)

    Returns:

    """
    N, K, _ = x.shape
    E = edge_attr.shape[0]
    edge_attr = edge_attr.view(E, 1, 4).repeat(1, K, 1)           # dim: (E, K, 4)
    row, col = edge_index
    x_i = x[row, :]                                               # dim: (E, K, 4)
    x_j = inv_q(x[col, :].view(-1, 4)).view(E, K, 4)              # dim: (E, K, 4)

    W = qmul(edge_attr.view(-1, 4), x_i.view(-1, 4))
    W = qmul(x_j.view(-1, 4), W.view(-1, 4))

    return W.view(E, K , 4)

class ScoreNetwork(nn.Module):

    def __init__(self, top_k=4):
        super(ScoreNetwork, self).__init__()
        self.no_features = 128
        self.input_node_feat = 4 * top_k + top_k
        self.dropout_ratio = 0.6

        self.conv1 = EdgeConvRot(self.input_node_feat, 4, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, self.no_features, self.no_features)
        self.conv2_pooling = SAGPooling(in_channels=self.no_features, GNN=GATConv)

        self.conv3 = EdgeConvRot(self.no_features, self.no_features, self.no_features)
        self.conv3_pooling = SAGPooling(in_channels=self.no_features, GNN=GATConv)

        self.conv4 = EdgeConvRot(self.no_features, self.no_features, self.no_features)
        self.conv4_pooling = SAGPooling(in_channels=self.no_features, GNN=GATConv)

        self.lin1 = torch.nn.Linear(self.no_features*2, self.no_features)
        self.lin2 = torch.nn.Linear(self.no_features, self.no_features//2)
        self.lin3 = torch.nn.Linear(self.no_features//2, top_k)

    def forward(self, node_feat, node_level, edge_index, edge_feat):

        N = node_feat.shape[0]
        K = node_feat.shape[1]

        E = edge_feat.shape[0]
        # edge_feat_ = update_attr_batch(node_feat, edge_index, edge_feat).view(-1, 4)

        node_feat = node_feat.view(N, -1)

        node_feat = torch.cat([node_feat, node_level], dim=1)

        x1, edge_x1 = self.conv1(node_feat, edge_index, edge_feat)
        x1, edge_x1 = F.relu(x1), F.relu(edge_x1)

        x2, edge_x2 = self.conv2(x1, edge_index, edge_x1)
        x2, edge_x2 = F.relu(x2), F.relu(edge_x2)

        x2_pool, edge_x2_index_pool, edge_x2_pool, batch, p2_to_x2, _ = self.conv2_pooling.forward(
            x2, edge_index, edge_attr=edge_x2)
        l1 = torch.cat([gmp(x2_pool, batch), gap(x2_pool, batch)], dim=1)

        x3, edge_x3 = self.conv3(x2_pool, edge_x2_index_pool, edge_x2_pool)
        x3, edge_x3 = F.relu(x3), F.relu(edge_x3)
        x3_pool, edge_x3_index_pool, edge_x3_pool, batch, p3_to_x3, _ = self.conv3_pooling.forward(
            x3, edge_x2_index_pool, edge_attr=edge_x3)
        l2 = torch.cat([gmp(x3_pool, batch), gap(x3_pool, batch)], dim=1)

        x4, edge_x4 = self.conv4(x3_pool, edge_x3_index_pool, edge_x3_pool)
        x4, edge_x4 = F.relu(x4), F.relu(edge_x4)
        x4_pool, edge_x4_index_pool, edge_x4_pool, batch, p4_to_x4, _ = self.conv4_pooling.forward(
            x4, edge_x3_index_pool, edge_attr=edge_x4)
        l3 = torch.cat([gmp(x4_pool, batch), gap(x4_pool, batch)], dim=1)
        l = l1 + l2 + l3

        x = F.relu(self.lin1(l))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x