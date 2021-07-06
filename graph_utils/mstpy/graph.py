import numpy as np

from abc import ABCMeta, abstractmethod

from graph_utils.mstpy.edge_base import *
from graph_utils.mstpy.node_base import *

class pylib_graph():
    """This is graph class for pythonlib

    This class only contain the direct edge.
    
    members:
        node_list: list, the node list for the graph.
        edge_list: list, the edge list for the graph.
        node_type: node_base, the class for the node
        edge_type: edge_base, the class for the edge
        node_size: int, the size of nodes
        edge_size: int, the size of edges
        edge_directed: bool, 
            True for directed edge, False for undirected edge
            default: False

        self_loop: bool, self loop in graph, default: True
        multi_edge: bool, multi edge in pair of node, default: False
    """

    __metaclass__ = ABCMeta

    def __init__(self, edge_directed=False, node_type=None, edge_type=None):
        """init function

        Keyword Arguments:
            edge_directed {bool} -- [The directed or undirected edge]
            node_type {node_base} -- [The class of the node] (default: {node_base})
            edge_type {edge_base} -- [The class of the edge] (default: {edge_base})
        """
        super().__init__()

        if node_type == None:
            node_type = node_base
        if edge_type == None:
            edge_type = edge_base

        self.edge_directed = edge_directed

        self.node_type = node_type
        self.edge_type = edge_type

        self.node_list = []
        self.edge_list = []

        self.degree_list = []

        self.adj_list = []
        self.adj_mat = None

        self.edge_set = set()

        self.node_size = 0
        self.edge_size = 0

        self.self_loop = True
        self.multi_edge = False

    def setSelfLoop(self, self_loop=None):
        """Set the status of the self loop

        Keyword Arguments:
            self_loop {bool} -- [True for self loop] (default: {None})
        """
        if self_loop != None:
            self.self_loop = self_loop
    
    def setMultiEdge(self, multi_edge=None):
        """set the status of the multi edge

        Keyword Arguments:
            multi_edge {bool} -- [True for self loop] (default: {None})
        """
        if multi_edge != None:
            self.multi_edge = multi_edge

    def addNode(self, node:node_base):
        self.node_list.append(node)
        self.node_size = self.node_size+1
        self.adj_list.append([])
        self.degree_list.append(0)

    def addEdge(self, edge:edge_base):
        edge_pair = (edge.u, edge.v)
        flag = True
        if not self.multi_edge:
            flag = flag and not (edge_pair in self.edge_set)         
        if not self.edge_directed:
            edge_pair = (edge_pair[0], edge_pair[1])
            if not self.multi_edge:
                flag = flag and not (edge_pair in self.edge_set)  
        if not flag:
            return flag

        self.edge_list.append(edge)
        self.edge_size = self.edge_size + 1

        self.edge_set.add((edge.u, edge.v))
        self.adj_list[edge.u].append(self.edge_size - 1)
        self.degree_list[edge.u] = self.degree_list[edge.u] + 1

        if not self.edge_directed:
            self.edge_set.add((edge.v, edge.u))
            self.adj_list[edge.v].append(self.edge_size - 1)
            self.degree_list[edge.v] = self.degree_list[edge.v]  + 1

        return flag

    def getAdjacencyList(self):
        return self.adj_list
    
    def getAdjacenctMatrix(self):
        mat = np.zeros([self.node_size,self.node_size],type=np.int)
        for edge in self.edge_list:
            mat[edge.u][edge.v] = mat[edge.u][edge.v] + 1
            if not self.edge_directed:
                mat[edge.v][edge.u] = mat[edge.v][edge.u] + 1

        return mat

    def getDegreeList(self):
        return self.degree_list
    
    def getPorpertyCopy(self):
        res = pylib_graph()

        res.edge_directed = self.edge_directed

        res.node_type = self.node_type
        res.edge_type = self.edge_type

        res.self_loop = self.self_loop
        res.multi_edge = self.multi_edge
        return res