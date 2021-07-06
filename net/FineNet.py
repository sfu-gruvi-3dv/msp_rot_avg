#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data
import h5py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import hdf5storage

# no_measurements = 1200 

# data_path = './' # os.getcwd() 
# datasetR = [] smooth_l1_loss


# In[4]:


# no_measurements = 1200
# datasetTrain = [] 
# filename = data_path+'data/gt_graph_random_large_outliers_updated_nocleannet.h5' 
# for item in range(99,1299): 
#     x = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/x', filename=filename, options=None), dtype=torch.float)
#     xt = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/xt', filename=filename, options=None), dtype=torch.float)
#     o = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/o', filename=filename, options=None), dtype=torch.float)
#  #   onode = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/onode', filename=filename, options=None), dtype=torch.float)
#  #   omarker = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/omarker', filename=filename, options=None), dtype=torch.float)
#     edge_index = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/edge_index', filename=filename, options=None), dtype=torch.long)
#     edge_attr = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/edge_feature', filename=filename, options=None), dtype=torch.float)
#     y = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/y', filename=filename, options=None), dtype=torch.float)
#     datasetTrain.append(Data(x=x, xt=xt, o=o, y=y, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)) 
#     print([item-99, datasetTrain[item-99]])


# # In[3]:


# no_measurements = 100
# datasetTest = [] 

# filename = data_path+'data/gt_graph_random_large_outliers_updated_nocleannet.h5' 
# #filename = data_path+'data/gt_graph_random_large_outliers_real_updated.h5' 
# for item in range(no_measurements): 
#     x = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/x', filename=filename, options=None), dtype=torch.float)
#     xt = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/xt', filename=filename, options=None), dtype=torch.float)
#     o = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/o', filename=filename, options=None), dtype=torch.float)
#  #   onode = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/onode', filename=filename, options=None), dtype=torch.float)
#  #   omarker = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/omarker', filename=filename, options=None), dtype=torch.float)
#     edge_index = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/edge_index', filename=filename, options=None), dtype=torch.long)
#     edge_attr = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/edge_feature', filename=filename, options=None), dtype=torch.float)
#     y = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/y', filename=filename, options=None), dtype=torch.float)
#     datasetTest.append(Data(x=x, xt=xt, o=o.t(), y=y, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)) 
#     print([item, datasetTest[item]])


# In[5]:


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


# In[6]:


from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import MessagePassing

class EdgeConvRot(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super(EdgeConvRot, self).__init__(aggr='mean', flow="target_to_source") #  "Max" aggregation.
        self.mlp = Seq(Linear(2*in_channels+edge_channels, out_channels),
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
        #out = scatter_(self.aggr, edge_out, edge_index[i], dim_size=size[i])
        out = scatter(edge_out, edge_index[i], dim=0,dim_size=size[i], reduce=self.aggr)
        return out, edge_out 


# In[29]:
import torch
import torch.nn.functional as F
#from torch_geometric.utils import scatter_
from torch_scatter import scatter
from torch_geometric.nn import GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

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
        self.mlp = Seq(Linear(2*in_channels+edge_channels, 8),
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
    
 


# In[30]:


def update_attr(x, edge_index, edge_attr):
    row, col = edge_index
    x_i = x[row]
    x_j = inv_q(x[col])
 #   print(x_i.shape)
  #  print(x_j.shape)
   # print(edge_attr.shape) 
    W=qmul(edge_attr, x_i) 
    W=qmul(x_j, W) 
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


def weighted_smooth_l1_loss(weight, input, beta=0.05, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])

    n = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    loss *= weight.reshape(*loss.shape)
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


def my_smooth_l1_loss_(input, beta, edge_index, size, alpha=0.05):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0])
    nn = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1)
    # beta = torch.squeeze(beta)
    nn = nn * beta

    cond = nn < alpha
    loss = torch.where(cond, 0.5 * nn ** 2 / alpha, nn - 0.5 * alpha)
    loss = scatter(loss, edge_index[0], dim=0, dim_size=size(0), reduce="mean")
    # out = scatter(edge_out, edge_index[i], dim=0,dim_size=size[i], reduce=self.aggr)

    return loss.mean()

def my_smooth_l1_loss_new(input, beta, edge_index, size, alpha=0.05):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    nn0 = torch.min(1.0 - input[:, 0], 1.0 + input[:, 0]) 
    nn = torch.abs(nn0) + torch.sum(torch.abs(input[:, 1:]), dim=1) 
    # beta = torch.squeeze(beta) 
    nn = nn * beta.view(*nn.shape)
    
    cond = nn < alpha
    loss = torch.where(cond, 0.5 * nn ** 2 / alpha, nn - 0.5 * alpha)   
    loss = scatter(loss, edge_index[0], dim=0, dim_size=size(0), reduce="mean")
    #out = scatter(edge_out, edge_index[i], dim=0,dim_size=size[i], reduce=self.aggr)

    return loss.mean()

class Net(torch.nn.Module):
    def __init__(self): 
        super(Net, self).__init__() 
        self.no_features = 32   # More features for large dataset 
        self.conv1 = EdgeConvRot(4, 4, self.no_features) 
        self.conv2 = EdgeConvRot(self.no_features, self.no_features+4, self.no_features)  
        self.conv3 = EdgeConvRot(2*self.no_features, 2*self.no_features, self.no_features) 
        self.conv4 = EdgeConvRot(2*self.no_features, 2*self.no_features, self.no_features) 

        self.lin1 = Linear(self.no_features, 4) 
        
        self.m = torch.nn.Sigmoid()

    def forward(self, data):
        x_org, edge_index, edge_attr, gt_q, beta = data
        edge_attr_mod = update_attr(x_org, edge_index, edge_attr[:, :4])
        x1, edge_x1 = self.conv1(x_org, edge_index,edge_attr_mod)
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
        x = self.lin1(x4) 
        x = x + x_org #qmul(x, x_org) 
        x = F.normalize(x, p=2, dim=1)
        #     loss1 = inv_q(edge_model(data.y, edge_index)) - edge_model(x, edge_index)
        loss1 = qmul(inv_q(edge_model(gt_q, edge_index)), edge_model(x, edge_index)) 
        loss1 = F.normalize(loss1, p=2, dim=1)
        #     loss1 = qmul(inv_q(edge_attr[:, :4]), edge_model(x, edge_index))
        loss1 = my_smooth_l1_loss_new(loss1[:, 0:], beta, edge_index, x_org.size)

        return x, loss1, beta   # node_model(x, batch),


class RA_FineNet(torch.nn.Module):
    def __init__(self):
        super(RA_FineNet, self).__init__()
        self.no_features = 32  # More features for large dataset
        self.conv1 = EdgeConvRot(4, 4, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, self.no_features + 4, self.no_features)
        self.conv3 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv4 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)

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

        x4, edge_x4 = self.conv4(torch.cat([x3, x2], dim=1), edge_index, torch.cat([edge_x3, edge_x2], dim=1))
        x4 = F.relu(x4)
        edge_x4 = F.relu(edge_x4)

        x = self.lin1(x4)
        x = x + x_org  # qmul(x, x_org)

        x = F.normalize(x, p=2, dim=1)
        #     loss1 = inv_q(edge_model(data.y, edge_index)) - edge_model(x, edge_index)
        loss1 = qmul(inv_q(edge_model(gt_q, edge_index)), edge_model(x, edge_index))
        loss1 = F.normalize(loss1, p=2, dim=1)
        #        loss1 = qmul(inv_q(edge_attr[:, :4]), edge_model(x, edge_index))
        loss1 = my_smooth_l1_loss_new(loss1[:, 0:], beta, edge_index, x_org.size)

        return x, loss1, beta, (x1, x2, x3, x4), (edge_x1, edge_x2, edge_x3, edge_x4)  # node_model(x, batch),

class EdgeMessnger(torch.nn.Module):
    def __init__(self, in_node_f_channel, in_edge_f_channel, ra_edge_f_channel, ra_node_f_channel):
        super(EdgeMessnger, self).__init__()
        self.no_features = 512  # More features for large dataset

        self.conv1 = EdgeConvRot(in_node_f_channel, in_edge_f_channel, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features + ra_node_f_channel, self.no_features + ra_edge_f_channel, self.no_features)
        self.conv3 = EdgeConvRot(self.no_features + ra_node_f_channel, 2 * self.no_features + ra_edge_f_channel, self.no_features)
        self.conv4 = EdgeConvRot(self.no_features + ra_node_f_channel, 2 * self.no_features + ra_edge_f_channel, self.no_features)
        self.lin1 = Linear(self.no_features, 1)

    def forward(self, node_feat, edge_index, e_feat, ra_node_feat, ra_edge_feat):

        # level 1
        x1, edge_x1 = self.conv1(node_feat, edge_index, e_feat)
        x1 = F.relu(x1)
        edge_x1 = F.relu(edge_x1)

        # level 2
        ra_x2_edge_feat = ra_edge_feat[1]
        ra_x2_feat = ra_node_feat[1]
        x2, edge_x2 = self.conv2(torch.cat([x1, ra_x2_feat], dim=1), edge_index, torch.cat([ra_x2_edge_feat, edge_x1], dim=1))
        x2 = F.relu(x2)
        edge_x2 = F.relu(edge_x2)

        # level 3
        ra_x3_edge_feat = ra_edge_feat[2]
        ra_x3_feat = ra_node_feat[2]
        x3, edge_x3 = self.conv3(torch.cat([x2, ra_x3_feat], dim=1), edge_index, torch.cat([ra_x3_edge_feat, edge_x2, edge_x1], dim=1))
        x3 = F.relu(x3)
        edge_x3 = F.relu(edge_x3)

        # level 4
        ra_x4_edge_feat = ra_edge_feat[3]
        ra_x4_feat = ra_node_feat[3]
        x4, edge_x4 = self.conv4(torch.cat([x3, ra_x4_feat], dim=1), edge_index, torch.cat([ra_x4_edge_feat, edge_x3, edge_x2], dim=1))

        edge_x4 = F.relu(edge_x4)

        out = self.lin1(edge_x4)
        return out


class Net_outlier_det(torch.nn.Module):
    def __init__(self, no_feats=32):
        super(Net_outlier_det, self).__init__()
        self.no_features = no_feats  # More features for large dataset
        self.conv1 = EdgeConvRot(4, 4, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, self.no_features + 4, self.no_features)
        self.conv3 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv4 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)

        self.lin1 = Linear(self.no_features, 4)
        self.lin2 = Linear(self.no_features, 1)

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
        x4, edge_x4 = self.conv4(torch.cat([x3, x2], dim=1), edge_index, torch.cat([edge_x3, edge_x2], dim=1))
        x4 = F.relu(x4)
        x = self.lin1(x4)
        x = x + x_org  # qmul(x, x_org)
        x = F.normalize(x, p=2, dim=1)

        out_res = self.lin2(edge_x4)

        #     loss1 = inv_q(edge_model(data.y, edge_index)) - edge_model(x, edge_index)
        loss1 = qmul(inv_q(edge_model(gt_q, edge_index)), edge_model(x, edge_index))
        loss1 = F.normalize(loss1, p=2, dim=1)
        #        loss1 = qmul(inv_q(edge_attr[:, :4]), edge_model(x, edge_index))
        loss1 = my_smooth_l1_loss_new(loss1[:, 0:], beta, edge_index, x_org.size)

        return x, loss1, beta  # node_model(x, batch),

class FineNetWithW(torch.nn.Module):
    def __init__(self, no_feats=32):
        super(FineNetWithW, self).__init__()
        self.no_features = no_feats  # More features for large dataset

        self.conv1 = EdgeConvRot(4, 5, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, self.no_features + 5, self.no_features)
        self.conv3 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv4 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)

        self.lin1 = Linear(self.no_features, 4)

        self.m = torch.nn.Sigmoid()

    def forward(self, data):
        x_org, edge_index, edge_attr, edge_w, gt_q, beta = data
        edge_attr_mod = update_attr(x_org, edge_index, edge_attr[:, :4])
        E = edge_attr_mod.shape[0]
        edge_attr_mod = torch.cat([edge_attr_mod, edge_w.view(E, 1)], dim=1)

        x1, edge_x1 = self.conv1(x_org, edge_index, edge_attr_mod)
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
        x = self.lin1(x4)
        x = x + x_org
        x = F.normalize(x, p=2, dim=1)

        loss1 = qmul(inv_q(edge_model(gt_q, edge_index)), edge_model(x, edge_index))
        loss1 = F.normalize(loss1, p=2, dim=1)
        loss1 = my_smooth_l1_loss_new(loss1[:, 0:], beta, edge_index, x_org.size)

        return x, loss1, beta  # node_model(x, batch),

# In[31]:


# PATH = 'checkpoint/graph_random_fine_updated_another.pth' 
# import time
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# training_exmpl = 0.8
# model = Net().to(device) 
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
# for g in optimizer.param_groups:
#     g['lr'] = 0.00005


# # In[ ]:


# PATH = 'checkpoint/graph_random_fine_updated.pth' 
# import numpy as np 

# train_loader = DataLoader(datasetTrain, batch_size=1, shuffle=True,num_workers=2)
# test_loader = DataLoader(datasetTest, batch_size=1, shuffle=False,num_workers=2)

# no_training = len(train_loader) 
# no_testing = len(test_loader) 

# model.train()
# best_loss = 200 
# t = time.time() 
# count = 0 
# val = 14688000
# for epoch in range(2500):
#     total_loss1 = 0 
#     total_loss2 = 0 
#     theta = [] 
#     for indx, data in enumerate(train_loader):
#         data_gpu = data.to(device)
#         optimizer.zero_grad()
#      #   print(data_gpu.y.shape)

#         out, loss1, beta = model(data_gpu)
#     #    out = data_gpu.x + out  
#    #     out = qmul(data_gpu.x, out) 
#       #  out = F.normalize(out, p=2, dim=1) 
#      #   print(out.shape)
#       #  loss1 = qmul(inv_q(data_gpu.edge_attr), edge_out)  
#       #  out = node_model(out, data_gpu.batch)
#       #  out = F.normalize(out, p=2, dim=1)
#         out = qmul(inv_q(data_gpu.y), out) 
#       #  loss1 = F.normalize(loss1, p=2, dim=1) 
#         out = F.normalize(out, p=2, dim=1) 

#     # F.smooth_l1_loss(10*edge_out, 10*data_gpu.edge_attr, size_average=None)  
#       #  loss2 = F.mse_loss(out, data_gpu.y, size_average=None) 
#    #    loss1 = loss1[:, 1:].abs().sum() #torch.nn.MSELoss(out, data_gpu.y)
#         loss2 = smooth_l1_loss(out[:, 0:]) 
 
#       #  loss2 = out[:, 1:].pow(2).sum() #(out - data_gpu.y).pow(2).sum() #torch.nn.MSELoss(out, data_gpu.y)
#         loss = loss1 + 0.25*loss2 
#         loss.backward()
#         optimizer.step()
#       #  print([indx, loss.item()])
#      #   time.sleep(0.05)
#         count = count + 1
#         if epoch % 2 == 0:
#             total_loss1 = total_loss1 + loss1.item() 
#             total_loss2 = total_loss2 + loss2.item()
#           #  out = qmul(inv_q(data_gpu.y), data_gpu.xt) 
#             val2 = out.data.cpu().numpy()
#             theta = np.concatenate((theta, 2.0*np.arccos(np.abs(val2[:, 0]))*180.0/np.pi ))
#     total_loss1 = total_loss1/no_training
#     total_loss2 = total_loss2/no_training
#     if epoch % 10 == 0:
#         total_loss1 = 0 
#         total_loss2 = 0 
#         for data in test_loader: 
#             data_gpu = data.to(device)
#             out, loss1, beta = model(data_gpu)
#             out = qmul(inv_q(data_gpu.y), out) 
#             out = F.normalize(out, p=2, dim=1) 
#             loss2 = smooth_l1_loss(out[:, 0:]) 
#             total_loss1 = total_loss1 + loss1.item() 
#             total_loss2 = total_loss2 + loss2.item()
#           #  out = qmul(inv_q(data_gpu.y), data_gpu.xt) 
#             val2 = out.data.cpu().numpy()
#             theta = np.concatenate((theta, 2.0*np.arccos(np.abs(val2[:, 0]))*180.0/np.pi ))
#         # print([count, total_loss1/no_training, total_loss2/no_training], time.time() - t) 
#         total_loss1 = total_loss1/no_testing
#         total_loss2 = total_loss2/no_testing
#     if epoch % 2 == 0: 
#         print([epoch, "{0:.6f}".format(total_loss1), "{0:.2f}".format(np.mean(theta)), "{0:.2f}".format(np.median(theta)), "{0:.6f}".format(total_loss2)], "{0:.3f}".format(time.time() - t))
#         if val > total_loss1/no_training : 
#             val = total_loss1/no_training 
#             torch.save({
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 }, PATH) 


# # In[25]:


# data_gpu.y.shape


# # In[ ]:





# # In[ ]:


# import numpy as np 
# import math 
# import h5py
# import torch 
# import time 
# data_path = './' # os.getcwd() 

# PATH = './checkpoint/graph_random_fine_updated.pth' 

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # device = torch.device('cpu')
# model = Net().to(device) 
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])

# test_loader = DataLoader(datasetTest, batch_size=1, shuffle=False)
# #model = best_model 
# #print(best_loss)
# #pred_rot = []
# model.eval()
# total_loss = 0 
# t = time.time() 
# count = 0 
# hf = h5py.File(data_path+'data/pred_syn_rot2.h5', 'w')
# #hf = h5py.File(data_path+'data/pred_synthetic_rot_another.h5', 'w')
# theta = [] 

# for data in test_loader: 
#     data_gpu = data.to(device)
#     pred, out, beta = model(data_gpu)
#    # data_gpu.x = pred
#    # pred, out, beta = model(data_gpu)
#    # data_gpu.x = pred
#   #  pred, out, beta = model(data_gpu)
#   #  pred = qmul(data_gpu.x, pred) 
#   #  pred = data_gpu.x + pred
#    # pred = F.normalize(pred, p=2, dim=1) 
#   #  pred = node_model(pred, data_gpu.batch)
#     pred = F.normalize(pred, p=2, dim=1)
        
#     out = qmul(inv_q(data_gpu.y), pred) 
#   #  out = qmul(inv_q(data_gpu.y), data_gpu.xt) 
#     out = F.normalize(out, p=2, dim=1) 
    
#     val2 = out.data.cpu().numpy()
#   #  print(val.size)
#     theta = np.concatenate((theta, 2.0*np.arccos(np.abs(val2[:, 0]))*180.0/np.pi )) 
    
#     loss = out[:, 1:].pow(2).sum() 
#  #   loss = (pred - data_gpu.y).pow(2).sum() 
#     total_loss = total_loss + loss.item() 
#     pred_rot = torch.cat([data_gpu.x, data_gpu.xt, pred, data_gpu.y], dim=1).data.cpu().numpy()
#     hf.create_dataset('/data/'+str(count+1), data=pred_rot)
#     count = count + 1 
#    # print([len(pred_rot), (time.time()-t)/len(pred_rot)])
# print(len(test_loader))
# hf.close()
# print([np.mean(theta), np.median(theta), (time.time() - t)/(test_loader.batch_size*len(test_loader))])


# # In[ ]:




