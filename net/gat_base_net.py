import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        # gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_normal_(self.fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        pass
        
    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, graph: dgl.DGLGraph, h):
        # equation (1)
        z = self.fc(h)
        graph.ndata['z'] = z
        # equation (2)
        graph.apply_edges(self.edge_attention)
        # equation (3) & (4)
        graph.update_all(self.message_func, self.reduce_func)
        return graph.ndata.pop('h')