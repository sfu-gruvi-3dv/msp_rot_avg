import torch
import numpy as np
import sys, os
import functools
import torch

from core_3dv.quaternion import *
from graph_utils.unionsetpy.unionset import *
def cmp(a,b):
    if a[3] > b[3]:
        return -1
    elif a[3] == b[3]:
        return 0
    else:
        return 1

def graph_add_edge(graph, edge):
    if edge[0] not in graph:
        graph[edge[0]] = []
    graph[edge[0]].append(edge)

def dfs_q(graph, spt_q,u,fa):
    for edge in graph[u]:
        if edge[1] == fa:
            continue
        v = edge[1]
        rel_q = edge[2]
        spt_q[v] = qmul(spt_q[u], rel_q)
        dfs_q(graph, spt_q, v, u)

def generate_spt_by_w(edge_idx, edge_rel_q, w, start_node):
    E = edge_rel_q.shape[0]
    edge_list = dict()
    graph = dict()
    for i in range(E):
        u,v = edge_idx[:,i]
        u = u.item()
        v = v.item()
        rel_q = edge_rel_q[i].cpu().numpy()
        e_w = w[i].item()
        if u not in edge_list:
            edge_list[u] = []
        if v not in edge_list:
            edge_list[v] = []
        # edge_list.append((u,v,rel_q,e_w))
        edge_list[u].append((u,v,rel_q,e_w))
        rel_q = torch.from_numpy(rel_q).view(1,4)
        rel_q_inv = inv_q(rel_q)
        edge_list[v].append((v,u,rel_q_inv.numpy(),e_w))
    
    node_num = len(edge_list)
    dist = dict()
    paths = dict()
    visted = dict()
    for key, val in edge_list.items():
        dist[key] = 1e18
        visted[key] = 0
    dist[start_node] = 0
    
    for i in range(node_num):
        nowminn = 1e18
        nownode= -1
        for j in edge_list.keys():
            if not visted[j] and (nownode==-1 or dist[j]<dist[nownode]):
                nownode = j
        visted[nownode] = 1
        for edge in edge_list[nownode]:
            u,v,rel_q,w = edge
            if not visted[v] and dist[v]>dist[u] + 1 - w:
                dist[v] = dist[u]+1 - w
                paths[v] = edge
    graph = dict()
    for key, edge in paths.items():
        rel_q = torch.from_numpy(edge[2]).view(1,4)
        rel_q_inv = inv_q(rel_q)
        graph_add_edge(graph, (edge[0],edge[1],rel_q[0,:],edge[3]))
        graph_add_edge(graph, (edge[1], edge[0], rel_q_inv[0,:], edge[3]))

    spt_q = dict()
    spt_q[start_node] = torch.zeros((1,4))
    spt_q[start_node][0,0] = 1 
    dfs_q(graph, spt_q, start_node,-1)
    return spt_q
        

def generate_mst_by_w(edge_idx, edge_rel_q, w):
    E = edge_rel_q.shape[0]
    pickedge_edge = []
    maxx_node = 0
    for i in range (edge_idx.shape[1]):
        u,v = edge_idx[:,i]
        u = u.item()
        v = v.item()
        maxx_node = u if u > maxx_node else maxx_node
        maxx_node = v if v > maxx_node else maxx_node

    us = unionset(maxx_node+1,int)
    edge_list = []
    graph = dict()
    for i in range(E):
        u,v = edge_idx[:,i]
        u = u.item()
        v = v.item()
        rel_q = edge_rel_q[i].cpu().numpy()
        e_w = w[i].item()
        edge_list.append((u,v,rel_q,e_w))
    edge_list = sorted(edge_list, key = functools.cmp_to_key( cmp))
    for edge in edge_list:
        if us.merge(edge[0],edge[1]):
            rel_q = torch.from_numpy(edge[2]).view(1,4)
            rel_q_inv = inv_q(rel_q)
            graph_add_edge(graph, (edge[0],edge[1],rel_q[0,:],edge[3]))
            graph_add_edge(graph, (edge[1], edge[0], rel_q_inv[0,:], edge[3]))
    
    spt_q = dict()
    start_node = list(graph.keys())[0]
    spt_q[start_node] = torch.zeros((1,4))
    spt_q[start_node][0,0] = 1 
    dfs_q(graph, spt_q, start_node,-1)
    return spt_q