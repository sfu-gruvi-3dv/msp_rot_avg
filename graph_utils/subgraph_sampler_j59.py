import os
from re import L
import sys
import random
import numpy as np
import math as m
print(sys.path)
from exp.make_yfcc_dbg_dataset import make_dataset,clean_cache
from torch.utils.data import dataloader


class SubgraphGenerator(object):

    def __init__(self, inoutMat=None, nodeMapping=None, nodeReMapping=None, maxNodeNum=50, minNodeNum=10, maxCoverRatio=0.1, verbose=False):
        super(SubgraphGenerator, self).__init__()
        self.seed = None
        self.inoutMat = inoutMat
        self.nodeMapping = nodeMapping
        self.nodeReMapping = nodeReMapping
        self.maxNodeNum = maxNodeNum
        self.minNodeNum = minNodeNum
        self.maxCoverRatio = maxCoverRatio
        self.verbose = verbose
        self.numOfNode = 0
        self.numOfEdge = 0
        if self.inoutMat is not None:
            self.numOfNode = self.inoutMat.shape[0]
            self.numOfEdge = np.sum(self.inoutMat) / 2
        if self.verbose:
            self.printInitSummary()

    def setSeed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def printInitSummary(self):
        print("SubgraphGenerator Summary:")
        print("Num of Nodes: ", self.numOfNode)
        print("Num of Edges: ", self.numOfEdge)
        print("Maximun Subgraph Node: ", self.maxNodeNum)
        print("Minimum Subgraph Node: ", self.minNodeNum)
        print("Maximum Cover Ratio: ", self.maxCoverRatio)

    def printGenerationSummary(self, subgraphs):
        print("Subgraph Generation Summary:")
        print("Num of Subgraph: ", len(subgraphs))
        totalNumOfNode = 0
        for subgraph in subgraphs:
            nowNodes = set()
            for edge in subgraph:
                nowNodes.add(edge[0])
                nowNodes.add(edge[1])
            totalNumOfNode += len(nowNodes)
        print("Total Num of Node: ", totalNumOfNode)

    def generationOnceNode(self, pickedNode, startNode=None, maxCoveredNode=None):
        if startNode is None and self.verbose:
            print("Start Node cannot be None")
            return
        if maxCoveredNode is None:
            maxCoveredNode = self.maxNodeNum
        nowPickedNode = [0 for x in range(self.numOfNode)]
        cnt_pick = 0
        nowCoveredNode = 0
        q = []
        if type(startNode) == int:
            q.append(startNode)
        else:
            nowPickedNode[startNode[0]] = 1
            cnt_pick += 1
            q.append(startNode[1])
            if pickedNode[startNode[0]] == 1:
                nowCoveredNode += 1
        

        while len(q) != 0 and cnt_pick < self.maxNodeNum:
            u = q.pop(0)
            if nowPickedNode[u] == 1:
                continue
            if pickedNode[u] == 1 and nowCoveredNode >= maxCoveredNode:
                continue
            nowPickedNode[u] = 1
            cnt_pick += 1
            if pickedNode[u]:
                nowCoveredNode += 1
            for i in range(self.numOfNode):
                if i == u or self.inoutMat[u, i] == 0:
                    continue
                q.append(i)

        nowPickedEdge = []
        for u in range(self.numOfNode):
            if nowPickedNode[u] != 1:
                continue
            for v in range(u, self.numOfNode):
                if u == v or nowPickedNode[v] != 1 or self.inoutMat[u, v] != 1:
                    continue
                nowPickedEdge.append((u, v))
        nowVisNode = []
        for node, vis in enumerate(nowPickedNode):
            if vis == 0:
                continue
            nowVisNode.append(node)
        return nowVisNode, nowPickedEdge

    def generationCoverNode(self):
        subgraphs = []

        pickedNode = set()
        unpickedNode = set(range(self.numOfNode))
        visedNode = [0 for x in range(self.numOfNode)]
        maxCoveredNode = m.floor(self.maxCoverRatio * self.maxNodeNum)

        while len(pickedNode) != self.numOfNode:
            startNode = random.choice(list(unpickedNode))
            nowPickedNode, nowPickedEdge = self.generationOnceNode(
                visedNode, startNode=startNode, maxCoveredNode=maxCoveredNode)
            if len(nowPickedNode) < self.minNodeNum:
                nowPickedNode, nowPickedEdge = self.generationOnceNode(
                    visedNode, startNode=startNode)
            for node in nowPickedNode:
                visedNode[node] = 1
                pickedNode.add(node)
                if node in unpickedNode:
                    unpickedNode.remove(node)
            subgraphs.append(nowPickedEdge)

        if self.nodeReMapping is not None:
            subgraphs_tmp = []
            for subgraph in subgraphs:
                subgraph_tmp = []
                for edge in subgraph:
                    subgraph_tmp.append(
                        (self.nodeReMapping[edge[0]], self.nodeReMapping[edge[1]]))
                subgraphs_tmp.append(subgraph_tmp)
            subgraphs = subgraphs_tmp

        if self.verbose:
            self.printGenerationSummary(subgraphs)

        return subgraphs

    def generationOnceNodeRotio(self, startNode, pickedNode, minCoverRatio=0.1):

        q = []
        nowPickedNode = []
        nowPickedEdge = []
        foundCover = False
        for i in range(self.numOfNode):
            if i in pickedNode:
                foundCover = True
        minCoverNodeNum = m.floor(self.maxNodeNum * minCoverRatio)
        if foundCover:
            q.append(startNode)
            while len(q) != 0:
                u = q.pop(0)
                if u in pickedNode:
                    nowPickedNode.append(u)
                if len(nowPickedNode) == minCoverNodeNum:
                    break
                for v in range(self.numOfNode):
                    if v!=u and self.inoutMat[u,v] ==1 and \
                        v in pickedNode and v not in nowPickedNode:
                        q.append(v)
        q = []
        q.append(startNode)
        while len(q) != 0:
            u = q.pop(0)
            if u not in pickedNode:
                nowPickedNode.append(u)
            if len(nowPickedNode) == self.maxNodeNum:
                break
            for v in range(self.numOfNode):
                if v!=u and self.inoutMat[u,v] ==1 and \
                    v not in pickedNode and v not in nowPickedNode:
                    q.append(v)
        nowPickedEdge = []
        for u in range(self.numOfNode):
            if u not in nowPickedNode:
                continue
            for v in range(u, self.numOfNode):
                if u == v or v not in nowPickedNode or self.inoutMat[u, v] != 1:
                    continue
                nowPickedEdge.append((u, v))
        return nowPickedNode, nowPickedEdge

    def generationOnceNodeUnpickedFirst(self, startNode, pickedNode, minCoverRatio=0.1):
        nowPickedNode = []
        nowPickedEdge = []
        minCoverNodeNum = self.maxNodeNum- m.floor(self.maxNodeNum * minCoverRatio)
        q = []
        q.append(startNode)
        while len(q) != 0:
            u = q.pop(0)
            if u not in pickedNode:
                nowPickedNode.append(u)
            if len(nowPickedNode) == minCoverNodeNum:
                break
            for v in range(self.numOfNode):
                if v!=u and self.inoutMat[u,v] ==1 and \
                    v not in pickedNode and v not in nowPickedNode:
                    q.append(v)
        
        foundCover = False
        for i in range(self.numOfNode):
            if i in pickedNode:
                foundCover = True
        
        if foundCover:
            q=[]
            q.append(startNode)
            while len(q) != 0:
                u = q.pop(0)
                if u in pickedNode:
                    nowPickedNode.append(u)
                if len(nowPickedNode) == self.maxNodeNum:
                    break
                for v in range(self.numOfNode):
                    if v!=u and self.inoutMat[u,v] ==1 and \
                        v in pickedNode and v not in nowPickedNode:
                        q.append(v)

        nowPickedEdge = []
        for u in range(self.numOfNode):
            if u not in nowPickedNode:
                continue
            for v in range(u, self.numOfNode):
                if u == v or v not in nowPickedNode or self.inoutMat[u, v] != 1:
                    continue
                nowPickedEdge.append((u, v))
        return nowPickedNode, nowPickedEdge

    def getStartNode(self, pickedNode, unpickedNode):
        startNode = -1
        if len(pickedNode) == 0:
            startNode = random.choice(list(unpickedNode))
        else:
            for i in range(self.numOfNode):
                if i in pickedNode:
                    continue
                for j in range(i+1,self.numOfNode):
                    if j in pickedNode:
                        startNode=i
                        break
                if startNode != -1:
                    break
        return startNode

    def generationCoverNodeByRatio(self, minCoverRatio=0.1):
        subgraphs = []
        pickedNode = set()
        unpickedNode = set(range(self.numOfNode))

        while len(pickedNode) != self.numOfNode:
            startNode = self.getStartNode(pickedNode, unpickedNode)
            if len(pickedNode) == 0:
                nowPickedNode, nowPickedEdge = self.generationOnceNodeUnpickedFirst(startNode, pickedNode, minCoverRatio=0)
            else:
                nowPickedNode, nowPickedEdge = self.generationOnceNodeUnpickedFirst(startNode, pickedNode, minCoverRatio)
            for node in nowPickedNode:
                pickedNode.add(node)
                if node in unpickedNode:
                    unpickedNode.remove(node)
            subgraphs.append(nowPickedEdge)

        if self.nodeReMapping is not None:
            subgraphs_tmp = []
            for subgraph in subgraphs:
                subgraph_tmp = []
                for edge in subgraph:
                    subgraph_tmp.append(
                        (self.nodeReMapping[edge[0]], self.nodeReMapping[edge[1]]))
                subgraphs_tmp.append(subgraph_tmp)
            subgraphs = subgraphs_tmp

        if self.verbose:
            self.printGenerationSummary(subgraphs)

        return subgraphs

    def generationCoverEdge(self):
        subgraphs = []

        pickedEdge = set()
        unpickedEdge = set()
        visedNode = [0 for x in range(self.numOfNode)]
        maxCoveredNode = m.floor(self.maxCoverRatio * self.maxNodeNum)
        if maxCoveredNode == 0:
            maxCoveredNode = 1

        for u in range(self.numOfNode):
            for v in range(u+1, self.numOfNode):
                if self.inoutMat[u, v] != 0:
                    unpickedEdge.add((u, v))

        while len(pickedEdge) != self.numOfEdge:
            startEdge = random.choice(list(unpickedEdge))
            nowPickedNode, nowPickedEdge = self.generationOnceNode(
                visedNode, startNode=startEdge,maxCoveredNode=maxCoveredNode)
            if len(nowPickedNode) < self.minNodeNum:
                nowPickedNode, nowPickedEdge = self.generationOnceNode(
                    visedNode, startNode=startEdge)
            for edge in nowPickedEdge:
                if edge[0] > edge[1]:
                    edge = (edge[1], edge[0])
                pickedEdge.add(edge)
                if edge in unpickedEdge:
                    unpickedEdge.remove(edge)
                visedNode[edge[0]] = 1
                visedNode[edge[1]] = 1
            subgraphs.append(nowPickedEdge)

        if self.nodeReMapping is not None:
            subgraphs_tmp = []
            for subgraph in subgraphs:
                subgraph_tmp = []
                for edge in subgraph:
                    subgraph_tmp.append(
                        (self.nodeReMapping[edge[0]], self.nodeReMapping[edge[1]]))
                subgraphs_tmp.append(subgraph_tmp)
            subgraphs = subgraphs_tmp

        if self.verbose:
            self.printGenerationSummary(subgraphs)

        return subgraphs

    def generationOnceDFS(u, vis,):
        pass

    def generationCoverEdgeBasedOnDFS(self):
        subgraphs = []

        self.visMat = np.zeros_like(self.inoutMat, type=np.int32)

        while len(self.unpickedEdge) != 0:
            startEdge = random.choice(list(self.unpickedEdge))
            self.nowPickedNode = []
            self.nowPickedEdge = []
            self.visedEdge = dict()

        if self.nodeReMapping is not None:
            subgraphs_tmp = []
            for subgraph in subgraphs:
                subgraph_tmp = []
                for edge in subgraph:
                    subgraph_tmp.append(
                        (self.nodeReMapping[edge[0]], self.nodeReMapping[edge[1]]))
                subgraphs_tmp.append(subgraph_tmp)
            subgraphs = subgraphs_tmp

        if self.verbose:
            self.printGenerationSummary(subgraphs)

        return subgraphs

    def validationSubgraphs(self, subgraphs):
        countNode = dict()
        countEdge = dict()
        for subgraph in subgraphs:
            for edge in subgraph:
                if self.nodeMapping is not None:
                    edge = (self.nodeMapping[edge[0]],
                            self.nodeMapping[edge[1]])
                if edge[0] > edge[1]:
                    edge = (edge[1], edge[0])
                if edge[0] not in countNode:
                    countNode[edge[0]] = 0
                if edge[1] not in countNode:
                    countNode[edge[1]] = 0
                if edge not in countEdge:
                    countEdge[edge] = 0
                countNode[edge[0]] += 1
                countNode[edge[1]] += 1
                countEdge[edge] += 1
        coveredEdge = 0
        coveredNode = 0
        for u in range(self.numOfNode):
            for v in range(u+1, self.numOfNode):
                if self.inoutMat[u, v] != 0 and (u, v) in countEdge:
                    coveredEdge += 1
        for i in range(self.numOfNode):
            if i in countNode:
                coveredNode += 1
        print("Node Cover Ratio: ", coveredNode/self.numOfNode)
        print("Edge Cover Ratio: ", coveredEdge/self.numOfEdge)

    @classmethod
    def fromInoutMat(cls, inoutMat, maxNodeNum=50, minNodeNum=10, maxCoverRatio=0.1, verbose=False):
        try:
            if len(inoutMat.size()) == 3:
                inoutMat = inoutMat[0]
        except:
            a=1
        inoutMat = np.where(inoutMat == -1, 0, inoutMat).astype(np.int32)
        for i in range(len(inoutMat)):
            for j in range(len(inoutMat)):
                if inoutMat[i,j] == 1:
                    inoutMat[j,i] = 1
        return cls(inoutMat, maxNodeNum=maxNodeNum, minNodeNum=minNodeNum, maxCoverRatio=maxCoverRatio, verbose=verbose)

    @classmethod
    def fromEdgelist(cls, edgelist, maxNodeNum=50, minNodeNum=10, maxCoverRatio=0.1, verbose=False):
        nodemapping = dict()
        noderemapping = dict()
        for edge in edgelist:
            u, v = edge
            if u not in nodemapping:
                nodemapping[u] = len(nodemapping)
                noderemapping[nodemapping[u]] = u
            if v not in nodemapping:
                nodemapping[v] = len(nodemapping)
                noderemapping[nodemapping[v]] = v
        inoutMat = np.zeros(
            [len(nodemapping), len(nodemapping)], dtype=np.int32)
        for edge in edgelist:
            inoutMat[nodemapping[edge[0]], nodemapping[edge[1]]] = 1
            inoutMat[nodemapping[edge[1]], nodemapping[edge[0]]] = 1
        return cls(inoutMat=inoutMat, nodeMapping=nodemapping, nodeReMapping=noderemapping, maxNodeNum=maxNodeNum,
                   minNodeNum=minNodeNum, maxCoverRatio=maxCoverRatio, verbose=verbose)


if __name__ == "__main__":
    edgelist = [(1, 2), (1, 3), (1, 4), (2, 8), (2, 4), (8, 11), (4, 12), (12, 4), (2, 1), (3, 4), (8, 12), (8, 17), (2, 17), (19, 3), (25, 4), (19, 25), (17, 25), (59, 12), (23, 59), (23, 12),
                (1,23), (12, 23),(74,23),(65,23),(88,52),(52,23),(56,23),(87,23),(17,96),(96,8),(23,17),(47,32),(47,4),(32,25),(95,12),(4,32),(78,19),(59,64),(63,21),(21,23)]
    # subgraphgen = SubgraphGenerator.fromEdgelist(
        # edgelist=edgelist, maxNodeNum=10, minNodeNum=3, verbose=True)


    _, valid = make_dataset()
    loader = dataloader.DataLoader(valid,
                                 batch_size=1,
                                 shuffle=True,
                                 pin_memory=True,
                                 drop_last=True)

    # itr = iter(loader)
    for s_itr, sample in enumerate(loader):
        dataset_name, idx, img_names, imgs, img_ori_dim, cam_Es, cam_Ks, out_graph_mat, img_id2sub_id, sub_id2img_id, out_covis_mat, edge_subnode_idx, edge_type, \
            edge_local_matches_n1, edge_local_matches_n2, edge_rel_Rt, edge_rel_err, node_feats, edge_feats = sample
        mat = np.zeros_like(out_graph_mat[0].detach().cpu().numpy())
        if len(edge_subnode_idx) < 1000:
            continue
        for edge in edge_subnode_idx:
            u = edge[0].item()
            v = edge[1].item()
            mat[u,v] = mat[v,u] = 1
        subgraphgen = SubgraphGenerator.fromInoutMat(inoutMat=mat, maxNodeNum=50, minNodeNum=20, verbose=True)
        break


    subgraphs = subgraphgen.generationCoverNodeByRatio(0.5)
    print("su:", len(subgraphs))
    # for subgraph in subgraphs:
    #     print(subgraph)
    subgraphgen.validationSubgraphs(subgraphs)
    # subgraphs = subgraphgen.generationCoverNode()
    # # for subgraph in subgraphs:
    # #     print(subgraph)
    # subgraphgen.validationSubgraphs(subgraphs)