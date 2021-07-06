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

    def __init__(self, draw_edge_label=True):
        self.G = nx.Graph()
        self.inlier_color = (0, 0.2, 0)
        self.outlier_color = (0.8, 0.8, 0.8)
        self.draw_edge_label = draw_edge_label
        self.pos = None
        self.node_text = {}
        self.mark_edge_list = []

    def add_node(self, node_id, img, name=None, unormlize_tensor=False, img_size_limit=800, text=None):
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
        self.G.add_node(node_id, image=img_vis, node_idx=node_id, name=name)
        if text is not None:
            self.node_text[node_id] = text

    def add_mark_edge(self, n1, n2):
        self.mark_edge_list.append((n1, n2))

    def add_edge(self, n1, n2, pred_prob:float, gt_inlier:bool, pred_prob_thres=0.8):
        gt_inlier = bool(gt_inlier)

        if pred_prob > pred_prob_thres:
            color = self.inlier_color
        else:
            color = self.outlier_color

        if gt_inlier is True and pred_prob > pred_prob_thres:
            flag = 'inlier_correct'
        elif gt_inlier is False and pred_prob < pred_prob_thres:
            flag = 'outlier_correct'
        elif gt_inlier is False and pred_prob > pred_prob_thres:
            flag = 'outlier_incorrect'
        else:
            flag = 'inlier_incorrect'

        self.G.add_edge(n1, n2, weight=pred_prob, color=color, flag=flag)

    def draw_spring_layout_marker(self, draw=False,  save_fig_path=None, figsize=(200, 200), img_size=0.02, update_pos=False):
        if update_pos is True or self.pos is None:
            pos = nx.spring_layout(self.G, iterations=30, scale=1.8)
            self.pos = pos
        else:
            pos = self.pos

        fig = plt.figure(figsize=(figsize[0], figsize[1]))
        ax = plt.subplot(111)
        ax.set_aspect('equal')

        # node label
        label_dict = {}
        label_pos_offset = {}
        y_off = 0.04
        for n in self.G:
            node_idx = str(self.G.nodes[n]['node_idx'])
            # label_dict[n] = '%s:%s' % (node_idx, self.G.nodes[n]['name'])
            label_dict[n] = '%s' % (node_idx)
            if n in self.node_text:
                label_dict[n] += ':%s' % str(self.node_text[n])

        for k, v in pos.items():
            label_pos_offset[k] = (v[0], v[1] + y_off)

        # assign edges
        edges = self.G.edges()
        probs = dict()
        inlier_corr_edges, inlier_incorr_edges = [], []
        outlier_corr_edges, outlier_incorr_edges = [], []
        cmap = matplotlib.cm.get_cmap('Blues')
        for u, v in edges:
            flag = self.G[u][v]['flag']
            # color = self.G[u][v]['color']
            prob = self.G[u][v]['weight']
            color = cmap(prob)
            color = (color[0], color[1], color[2])
            inlier_flag = 'I'
            if flag == 'inlier_correct':
                inlier_corr_edges.append(((u, v), color))
            elif flag == 'outlier_correct':
                outlier_corr_edges.append(((u, v), color))
                inlier_flag = 'O'
            elif flag == 'outlier_incorrect':
                outlier_incorr_edges.append(((u, v), color))
            else:
                inlier_incorr_edges.append(((u, v), color))
                inlier_flag = 'O'
            probs[(u, v)] = '(%d, %d)=%.3f(%s)' % (u, v, prob, inlier_flag)

        nx.draw_networkx_edges(self.G, pos, ax=ax,
                                 edgelist=self.mark_edge_list, width=15.0, alpha=1)

        nx.draw_networkx_labels(self.G, pos=label_pos_offset, ax=ax, labels=label_dict, font_size=30)

        # append images
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


    def draw_spring_layout(self, draw=False, save_fig_path=None, figsize=(200, 200), img_size=0.02, update_pos=False):

        if update_pos is True or self.pos is None:
            pos = nx.spring_layout(self.G, iterations=30, scale=1.8)
            self.pos = pos
        else:
            pos = self.pos

        fig = plt.figure(figsize=(figsize[0], figsize[1]))
        ax = plt.subplot(111)
        ax.set_aspect('equal')

        # node label
        label_dict = {}
        label_pos_offset = {}
        y_off = 0.04
        for n in self.G:
            node_idx = str(self.G.nodes[n]['node_idx'])
            # label_dict[n] = '%s:%s' % (node_idx, self.G.nodes[n]['name'])
            label_dict[n] = '%s' % (node_idx)

            if n in self.node_text:
                label_dict[n] += ':%s' % str(self.node_text[n])

        for k, v in pos.items():
            label_pos_offset[k] = (v[0], v[1] + y_off)

        edges = self.G.edges()
        max_prob, min_prob = -1, 2.0
        for u, v in edges:
            prob = self.G[u][v]['weight']
            if prob < min_prob:
                min_prob = prob
            if prob > max_prob:
                max_prob = prob

        # assign edges
        probs = dict()
        inlier_corr_edges, inlier_incorr_edges = [], []
        outlier_corr_edges, outlier_incorr_edges = [], []
        cmap = matplotlib.cm.get_cmap('Blues')
        for u, v in edges:
            flag = self.G[u][v]['flag']
            # color = self.G[u][v]['color']
            prob = self.G[u][v]['weight']
            color = cmap((prob - min_prob) / (max_prob - min_prob))
            color = (color[0], color[1], color[2])
            inlier_flag = 'I'
            if flag == 'inlier_correct':
                inlier_corr_edges.append(((u, v), color))
            elif flag == 'outlier_correct':
                outlier_corr_edges.append(((u, v), color))
                inlier_flag = 'O'
            elif flag == 'outlier_incorrect':
                outlier_incorr_edges.append(((u, v), color))
            else:
                inlier_incorr_edges.append(((u, v), color))
                inlier_flag = 'O'
            probs[(u, v)] = '(%d, %d)=%.3f(%s)' % (u, v, prob, inlier_flag)

        if self.draw_edge_label:
            nx.draw_networkx_edge_labels(self.G, pos, ax=ax, edge_labels=probs, font_size=10)


        nx.draw_networkx_edges(self.G, pos, ax=ax,
                               edgelist=[e[0] for e in inlier_corr_edges],
                               edge_color=[e[1] for e in inlier_corr_edges], width=8.0, style='solid')

        nx.draw_networkx_edges(self.G, pos, ax=ax,
                               edgelist=[e[0] for e in outlier_corr_edges],
                               edge_color=[e[1] for e in outlier_corr_edges], width=8.0, style='solid')

        nx.draw_networkx_edges(self.G, pos, ax=ax,
                               edgelist=[e[0] for e in outlier_incorr_edges],
                               edge_color=[e[1] for e in outlier_incorr_edges], width=8.0, alpha=1, style='dashdot')
        nx.draw_networkx_edges(self.G, pos, ax=ax,
                               edgelist=[e[0] for e in inlier_incorr_edges],
                               edge_color=[e[1] for e in inlier_incorr_edges], width=8.0, alpha=1, style='dashdot')

        nx.draw_networkx_labels(self.G, pos=label_pos_offset, ax=ax, labels=label_dict, font_size=30)

        # append images
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

