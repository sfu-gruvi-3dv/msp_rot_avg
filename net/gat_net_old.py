import torch
import torch.nn as nn
import torch.nn.functional as F
from net.gat_base_net import GATLayer
from net.multihead_gat_net import MultiHeadGATLayer
from dgl.nn.pytorch import GATConv


class MultiGATBaseConvs(nn.Module):

    def __init__(self, input_feat_channel=512, n_head=16):
        super(MultiGATBaseConvs, self).__init__()
        self.n_head = n_head
        self.l1 = GATConv(in_feats=input_feat_channel, out_feats=512, num_heads=n_head, residual=True)
        self.l2 = GATConv(in_feats=n_head*512, out_feats=512, num_heads=n_head, residual=True)
        self.l3 = GATConv(in_feats=n_head*512, out_feats=512, num_heads=n_head, residual=True)
        self.l4 = GATConv(in_feats=n_head*512, out_feats=512, num_heads=1, residual=True)
        
#         self.l = GATConv(in_feats=[n_head, 512], out_feats=[1, 512], num_heads=1)
        
    def forward(self, graph, feat):
        N = feat.shape[0]
#         print(feat.shape)
        x = self.l1.forward(graph, feat)
        x1 = F.relu(x)
#         print(x.shape)
        x = self.l2.forward(graph, x1.view(N, -1))
        x = F.relu(x)
        
        x = self.l3.forward(graph, x.view(N, -1))
        x = F.relu(x)      
        
        x = self.l4.forward(graph, x.view(N, -1))
        x = F.relu(x)
        
        return x.view(N, -1)
    
class GATBase(nn.Module):

    def __init__(self, input_feat_channel=512):
        super(GATBase, self).__init__()
        self.l1 = GATLayer(input_feat_channel, 512)
        self.l2 = GATLayer(512, 512)
        self.l3 = GATLayer(512, 512)
        self.l4 = GATLayer(512, 512)

    def forward(self, graph, feat):
        x = self.l1.forward(graph, feat)
        x = self.l2.forward(graph, x)
        x = self.l3.forward(graph, x)
        x = self.l4.forward(graph, x)
        return x

class EdgeClassifier(nn.Module):

    def __init__(self, input_feat_channel=512):
        super(EdgeClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_feat_channel, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.fc.forward(x)

class EdgeClassifier2(nn.Module):

    def __init__(self, input_feat_channel, input_local_feat_channel):
        super(EdgeClassifier2, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_feat_channel, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(input_local_feat_channel, 256),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, gat_cat_feats, local_eg_feats):
        x1 = self.fc.forward(gat_cat_feats)
        x2 = self.fc1.forward(local_eg_feats)
        return self.fc2.forward(torch.cat([x1, x2], dim=1))


class MultiGATBase(nn.Module):

    def __init__(self, input_feat_channel=512):
        super(MultiGATBase, self).__init__()
        n_head = 16
        self.l1 = MultiHeadGATLayer(input_feat_channel, 512, num_heads=n_head)
        self.l2 = MultiHeadGATLayer(512*n_head, 512, num_heads=n_head)
        self.l3 = MultiHeadGATLayer(512*n_head, 512, num_heads=n_head)
        self.l4 = MultiHeadGATLayer(512*n_head, 512, num_heads=1)
        
    def forward(self, graph, feat):
        x = self.l1.forward(graph, feat)
        x = self.l2.forward(graph, x)
        x = self.l3.forward(graph, x)
        x = self.l4.forward(graph, x)
        return x
