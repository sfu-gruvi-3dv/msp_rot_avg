from abc import ABCMeta, abstractmethod

class unionset():
    """
    This is the unionset Class
    """
    __metaclass__ = ABCMeta

    def __init__(self, usize, val):
        super().__init__()
        self.value_type = val
        self.usize = usize
        self.father = [x for x in range(self.usize)]
        self.value = [self.value_type() for x in range(self.usize)]
        self.size = [1 for x in range(self.usize)]

    def resize(self, newsize):
        """
        resize the unionset and clear()
        """
        self.usize = newsize
        self.clear()
        
    def clear(self):
        """
        clear == reset
        reset the unionset
        """
        self.father = [x for x in range(self.usize)]
        self.value = [self.value_type() for x in range(self.usize)]
        self.size = [1 for x in range(self.usize)]

    def mergevalue(self, a,b):
        """
        merge the value of cluster a and b
        a and b should not be same cluster
        """
        if a == b:
            return
        self.value[a] = self.value[a] + self.value[b]
        self.size[a] = self.size[a] + self.size[b]

    def __getitem__(self, idx):
        """
        As same as getValue, return the value of the cluster which idx in
        """
        return self.value[idx]

    def merge(self, a,b):
        """
        merge a and b, return True if merge, otherwise False
    
        :Return:
            bool: `True` if merge, otherwise `False`
        """
        fa = self.getFather(a)
        fb = self.getFather(b)

        if fa != fb:
            if fa > fb:
                fa, fb = fb, fa
            self.mergevalue(fa,fb)
            self.father[fb] = fa
            return True
        return False

    def getFather(self, idx):
        """
        Get the father node of idx
        """
        root = idx

        while root != self.father[root]:
            root = self.father[root]
        while idx != root:
            self.father[idx], idx = root, self.father[idx]
        return root

    def getSize(self, idx):
        """
        get the size of cluster of idx
        """
        return self.size[self.getFather(idx)]

    def getValue(self, idx):
        """
        get the value of the cluster which idx in
        """
        return self.value[self.getFather(idx)]


if __name__ == "__main__":
    us = unionset(10,int)
    print("%d"% us.getFather(3))
    