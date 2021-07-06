from abc import ABCMeta, abstractmethod

class node_base():
    __metaclass__ = ABCMeta

    def __init__(self, id=0, value=None):
        super().__init__()
        self.value = value
        self.id = id
        