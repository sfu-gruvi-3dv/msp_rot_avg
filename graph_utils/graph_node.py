class Node:
    childrens = []
    parent = None

    def __init__(self, attr, level=0):
        self.attr = attr
        self.childrens = []
        self.parent = None
        self.level = 0

    def add_child(self, node):
        node.parent = self
        node.level = self.level + 1
        self.childrens.append(node)

    def __str__(self):
        return str([s.attr for s in self.childrens])