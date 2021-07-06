from abc import ABCMeta, abstractmethod

class edge_base():
    __metaclass__ = ABCMeta

    def __init__(self, u, v, dist=0, value=None):
        super().__init__()
        self.value = value
        self.dist = dist
        self.u = u
        self.v = v

    @abstractmethod
    def __lt__(self, other):
        """
        less operation used in sorting.
        you need to implement this function in your child edge class
        """
        # TODO: implement lt function
        return self.dist < other.dist
