import random
import copy

from abc import ABCMeta, abstractmethod
from graph_utils.mstpy.graph import pylib_graph
from graph_utils.mstpy.shortestpath import *
from graph_utils.mstpy.mlst_util import *

class MLST_base():
    __metaclass__ = ABCMeta
    """
    This is the base Maximum leaf spanning tree algorithm
    link: https://link.springer.com/article/10.1007/PL00009201
    """

    def __init__(self, graph:pylib_graph, debug=False):
        """This is the constructor function for MLST_base class

        Arguments:
            graph {[type]} -- [description]
        """
        super().__init__()
        self.graph = graph

        self.white_mark = [0 for x in range(self.graph.node_size)]
        self.gray_mark = [0 for x in range(self.graph.node_size)]
        self.black_mark = [0 for x in range(self.graph.node_size)]

        self.degree_list = self.graph.getDegreeList()
        self.debug = debug

    def MLST_base_one_loop(self):
        u,maxu = self.MLST_base_pick_gray_node()
        if self.debug:
            print("gray node: %d, max degree = %d" % (u, maxu))

        self.MLST_base_scan_gray_node(u)
        self.MLST_base_add_black_node(u)
        if self.debug:
            print("node u in black?" + str(u in self.black_list))
            print("node u in gray?" + str(u in self.gray_list))
            print("node u in white?" + str(u in self.white_list))

    def MLST_base_add_black_node(self, u:int):
        if u in self.gray_list:
            self.gray_list.remove(u)
            self.black_list.append(u)

    def MLST_base_scan_gray_node(self, u:int):
        edge_list = self.adj_list[u]

        for eid in edge_list:
            edge = self.graph.edge_list[eid]
            v = edge.u
            if v == u:
                v = edge.v
            if v in self.white_list:
                self.MLST_base_remove_from_white_list(v)
                self.gray_list.append(v)
                self.MLST_edge.append(eid)
                

    def MLST_base_pick_gray_node(self):
        res_node = 0
        res_max_white = -1
        for node in self.gray_list:
            if self.white_number[node] > res_max_white:
                res_node = node
                res_max_white = self.white_number[node]
        
        if res_max_white <= 0:
            for node in self.white_list:
                if self.white_number[node] > res_max_white:
                    res_node = node
                    res_max_white = self.white_number[node]

            self.MLST_base_remove_from_white_list(res_node)
            self.gray_list.append(res_node)

        return res_node, res_max_white

    def MLST_base_remove_from_white_list(self, v:int):
        self.white_list.remove(v)
        v_edges = self.adj_list[v]
        for eid in v_edges:
            edge = self.graph.edge_list[eid]
            vv = edge.u
            if vv == v:
                vv = edge.v
            self.white_number[vv] = self.white_number[vv] - 1

    def getMLST_base(self, start_node=None):
        if self.debug:
            print("Start to generate base MLST")

        if start_node == None:
            start_node = self.degree_list.index(max(self.degree_list))
        self.start_node = start_node
        if self.debug:
            print("Start Node ID = %d" % self.start_node)

        self.adj_list = self.graph.getAdjacencyList()

        self.white_number = self.degree_list.copy()
        self.black_list = []
        self.gray_list = []
        self.white_list = [i for i in range(self.graph.node_size)]
        self.marked_list = []
        self.MLST_edge = []

        self.MLST_base_remove_from_white_list(self.start_node)
        self.gray_list.append(self.start_node)

        while len(self.white_list) != 0:
            if self.debug:
                print("white node size = %d" % len(self.white_list))
            self.MLST_base_one_loop()

        return self.black_list, self.MLST_edge

    def getMLST_base_with_t_spanner(self, start_node=None, t=None):
        
        if t == None:
            t = 2.0

        graph_Ts = self.graph.getPorpertyCopy()
        for node in self.graph.node_list:
            graph_Ts.addNode(node)
        
        self.getMLST_base(start_node)

        for eid in self.MLST_edge:
            edge = self.graph.edge_list[eid]
            graph_Ts.addEdge(edge)

        for eid in range(len(self.graph.edge_list)):
            edge = self.graph.edge_list[eid]
            if eid in self.MLST_edge:
                continue
            if edge.u in self.black_list and \
                edge.v in self.black_list:
                graph_tmp = copy.deepcopy(graph_Ts)
                graph_tmp.addEdge(edge)
                shortest_Ti = shortestPathDijkstra(graph_tmp, edge.u)
                shortest_Ts = shortestPathDijkstra(graph_Ts, edge.u)
                if shortest_Ti[edge.v]*t < shortest_Ts[edge.v]:
                    graph_Ts = graph_tmp

        for eid in range(len(self.graph.edge_list)):
            edge = self.graph.edge_list[eid]
            if edge not in graph_Ts.edge_list:
                graph_tmp = copy.deepcopy(graph_Ts)
                graph_tmp.addEdge(edge)
                shortest_Ti = shortestPathDijkstra(graph_tmp, edge.u)
                shortest_Ts = shortestPathDijkstra(graph_Ts, edge.u)
                if shortest_Ti[edge.v]*t < shortest_Ts[edge.v]:
                    graph_Ts = graph_tmp

        return graph_Ts
                
