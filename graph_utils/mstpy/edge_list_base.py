from abc import ABCMeta, abstractmethod


class edge_list_base():
    def __init__(self, edge_list):
        super().__init__()
        self.edge_list = edge_list
        self.len = len(self.edge_list)
    
    def __getitem__(self, idx):
        return self.edge_list[self.edge_list.index(idx)]

