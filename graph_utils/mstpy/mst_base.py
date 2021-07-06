from abc import ABCMeta, abstractmethod
from graph_utils.unionsetpy.unionset import *
from graph_utils.mstpy.edge_base import edge_base
from graph_utils.mstpy.node_base import node_base

class mst_base():

    __metaclass__ = ABCMeta

    def __init__(self, node_type=None, edge_type=None):
        """
        vector_list: list, with node value
        edge_list: list, with edge value
        edge_type: str, 'list' for adjacent list, 'matrix' for adjacent matrix
        """

        if node_type == None:
            node_type = node_base
        if edge_type == None:
            edge_type = edge_base
        super().__init__()
        self.node_type = node_type
        self.edge_type = edge_type
        self.node_list = []
        self.edge_list = []
        
        self.mst = None


    def generate_mst(self):
        """
        The base function for mst
        Using Kruskal algorithm
        """
        self.unionset = unionset(len(self.node_list), int)
        self.edge_list.sort()
        self.mst = []
        for edge in self.edge_list:
            u = edge.u
            v = edge.v
            if self.unionset.merge(u,v):
                self.mst.append(edge)
        return self.mst
    