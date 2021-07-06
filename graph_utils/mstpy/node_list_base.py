from abc import ABCMeta, abstractmethod

def node_list_base():
    __metaclass__ = ABCMeta

    def __init__(self, node_list):
        super().__init__()
        self.len = len(node_list)
        self.node_list = node_list
    
    def append(self, new_node):
        self.node_list.append(new_node)
    
    def __getitem__(self, idx):
        return self.node_list[self.node_list.index(idx)]

    