import sys, os, pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import torch, cv2
import numpy as np
from core_dl.torch_vision_ext import UnNormalize

matplotlib.use('Agg')
unorm = UnNormalize()

class GraphVisualizer:

    def __init__(self):
        self.G = nx.Graph()

    def add_node(self, node_id, img, unormlize_tensor=False, img_size_limit=800):
        if isinstance(img, torch.Tensor):
            if unormlize_tensor:
                img = unorm(img)
            img_vis = img.permute(1, 2, 0).cpu().numpy()
            v_h, v_w = img_vis.shape[:2]
            while v_h > img_size_limit or v_w > img_size_limit:
                img_vis = cv2.resize(img_vis, (0, 0), fx=0.5, fy=0.5)
                v_h, v_w = img_vis.shape[:2]
            img_vis = (img_vis * 255).astype(np.uint8)
        elif isinstance(img, np.ndarray):
            img_vis = img
            v_h, v_w = img_vis.shape[:2]
            while v_h > img_size_limit or v_w > img_size_limit:
                img_vis = cv2.resize(img_vis, (0, 0), fx=0.5, fy=0.5)
                v_h, v_w = img_vis.shape[:2]
            img_vis = (img_vis * 255).astype(np.uint8)
        self.G.add_node(node_id, image=img_vis)

    def add_edge(self, n1, n2, pred_dist=1.0, gt_dist=0.0, color=(1.0, 0.0, 0.0), label=None):
        self.G.add_edge(n1, n2, weight=pred_dist, color=color, label=label)

    def draw_spring_layout(self, draw=False, save_fig_path=None, figsize=(200, 200), img_size=0.06):
        edges = self.G.edges()
        colors = [self.G[u][v]['color'] for u, v in edges]
        pos = nx.spring_layout(self.G, k=2.0 / np.sqrt(len(self.G.nodes)), iterations=20, scale=1.8)

        fig = plt.figure(figsize=(figsize[0], figsize[1]))
        ax = plt.subplot(111)
        ax.set_aspect('equal')
        nx.draw_networkx_edges(self.G, pos, ax=ax, edge_color=colors, width=2.0)

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform

        piesize = img_size  # this is the image size
        p2 = piesize / 2.0
        for n in self.G:
            xx, yy = trans(pos[n])  # figure coordinates
            xa, ya = trans2((xx, yy))  # axes coordinates
            a = plt.axes([xa - p2, ya - p2, piesize, piesize])
            a.set_aspect('equal')
            a.imshow(self.G.nodes[n]['image'])
            a.axis('off')
        ax.axis('off')
        print('[GraphVisualizer] Done Rendering Graph (%d nodes, %d edges)' % (len(self.G.nodes), len(edges)))
        if draw:
            plt.show()
        plt.savefig(save_fig_path)

