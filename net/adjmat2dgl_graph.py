import torch
import torch.nn as nn
import numpy as np
import dgl
import networkx as nx

def build_graph(num_nodes: int, edge_node_pair, undirect=True, self_edge=True):

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)

    if self_edge is True:
        for n in range(num_nodes):
            g.add_edge(n, n)

    for edge in edge_node_pair:
        n1 = edge[0]
        n2 = edge[1]

        if undirect is True:
            g.add_edge(n1, n2)
            g.add_edge(n2, n1)
        else:
            g.add_edge(n1, n2)

    return g

def adjmat2graph(adj_mat, undirect=True, self_edge=True):
    adj_mat = adj_mat.cpu().numpy()
    N = adj_mat.shape[0]  # N nodes
    g = dgl.DGLGraph()
    g.add_nodes(N)

    for n in range(N):
        row = adj_mat[n, :]
        nnz_indices = np.nonzero(row)[0]
        if self_edge:
            g.add_edge(n, n)
        for idx in range(len(nnz_indices)):
            nnz_idx = nnz_indices[idx]
            if undirect is False and nnz_idx > n:
                g.add_edge(n, nnz_idx)
            elif undirect is True:
                g.add_edge(n, nnz_idx)

    return g

def gather_edge_label(adj_mat, self_edge=True):
    out_graph_np = adj_mat.cpu().numpy()
    N = out_graph_np.shape[0]

    edge_label = []
    for n in range(N):
        row = out_graph_np[n, :]
        nnz_indices = np.nonzero(row)[0]
        if self_edge:
            edge_label.append(1)
        for idx in range(len(nnz_indices)):
            nnz_idx = nnz_indices[idx]
            if nnz_idx > n:
                label = 1
                if row[nnz_idx] == 1:
                    label = 1
                elif row[nnz_idx] == -1:
                    label = 0
                edge_label.append(label)
    edge_label = np.asarray(edge_label).ravel()

    return torch.from_numpy(edge_label).float()

def gather_edge_feat(adj_mat, edge_classifier, node_feats, self_edge=True):
    # run edge classifier
    out_graph_np = adj_mat.cpu().numpy()
    N = out_graph_np.shape[0]

    edge_feats = []
    for n in range(N):
        row = out_graph_np[n, :]
        nnz_indices = np.nonzero(row)[0]
        if self_edge:
            h1 = node_feats[n:n + 1, :]
            h_concate = torch.cat([h1, h1], dim=0)
            edge_feats.append(h_concate)
        for idx in range(len(nnz_indices)):
            nnz_idx = nnz_indices[idx]
            if nnz_idx > n:
                # pair (n, nnz_idx)
                h1 = node_feats[n:n + 1, :]
                h2 = node_feats[nnz_idx:nnz_idx + 1, :]
                h_concate = torch.cat([h1, h2], dim=0)
                edge_feats.append(h_concate)

    edge_feats = torch.stack(edge_feats, dim=0)
    out_edge_res = edge_classifier.forward(edge_feats.view(edge_feats.shape[0], -1))
    return out_edge_res

def inv_adjmat(adj_mat, self_edge=False):
    adj_mat = adj_mat.cpu().numpy()
    N = adj_mat.shape[0]
    e_num_adjmat = adj_mat
    e_num_adjmat = e_num_adjmat.astype(np.int)
    num_edges = 0
    e_dict = {}
    for n in range(N):
        for n2 in range(n, N):
            if not adj_mat[n, n2] == 0:
                num_edges += 1
                e_num_adjmat[n, n2] = num_edges
                e_num_adjmat[n2, n] = num_edges
                e_dict[num_edges] = (n,n2)
    invmat = np.zeros([num_edges, num_edges])
    for e in range(1, num_edges+1):
        nodes = e_dict[e]
        r1 = e_num_adjmat[nodes[0]]
        r2 = e_num_adjmat[nodes[1]]
        for n in range(N):
            if r1[n] > 0:
                adj_e = r1[n]
                invmat[e-1, adj_e-1] = 1
            if r2[n] > 0:
                adj_e = r2[n]
                invmat[e-1, adj_e-1] = 1
    invmat = invmat - np.identity(num_edges)
    
    if self_edge:
        for i in range(invmat.shape[-1]):
            invmat[i, i] = 1
    # print("numedges:", num_edges)
    return invmat, num_edges

def gather_edge_feat2(adj_mat, node_feats):
    # run edge classifier
    out_graph_np = adj_mat.cpu().numpy()
    N = out_graph_np.shape[0]

    edge_feats = []
    for n in range(N):
        row = out_graph_np[n, :]
        nnz_indices = np.nonzero(row)[0]
        for idx in range(len(nnz_indices)):
            nnz_idx = nnz_indices[idx]
            if nnz_idx > n:
                # pair (n, nnz_idx)
                h1 = node_feats[n:n + 1, :]
                h2 = node_feats[nnz_idx:nnz_idx + 1, :]
                h_concate = torch.cat([h1, h2], dim=0)
                edge_feats.append(h_concate)

    edge_feats = torch.stack(edge_feats, dim=0)
    return edge_feats