import torch
import torch.nn as nn
from data.ambi.ambi_dataset import AmbiDataset, imgs2batchsamesize
from torch.utils.data import Dataset, DataLoader
from net.gat_net import GATBase, EdgeClassifier
import networkx as nx
import numpy as np
from vlad_encoder import VLADEncoder
from net.adjmat2dgl_graph import adjmat2graph, gather_edge_feat, gather_edge_label, inv_adjmat, gather_edge_feat2
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import datetime

""" Configuration
"""
max_epochs = 100

vlad_pretrain_path = 'cache/netvlad_vgg16.tar'
log_path = "./logs_invmat/logs_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#log_path = "./log"
# Loss function: binary classification
criterion = nn.BCEWithLogitsLoss()

def accuracy(pred, gt_label, threshold=0.5):
    pred = pred.detach().cpu().numpy()
    gt_label = gt_label.detach().cpu().numpy().astype(np.int).ravel()
    label = (pred > threshold).astype(np.int).ravel()

    correct = np.sum(gt_label == label)
    return float(correct) / float(label.shape[0])

# Training Data
dataset_list = [
    {'name': 'cup', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00064.out'},
    {'name': 'books', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00021.out'},
    {'name': 'cereal', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00025.out'},
    {'name': 'desk', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00031.out'},
    {'name': 'oats', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00024.out'},
    {'name': 'street', 'bundle_file': 'Models_0423_MC_GL_withBA_0.1_DV0.05_30/model000/bundle_00019.out'},
]

transform_func = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
train_dataset = AmbiDataset(dataset_base_dir='/mnt/Tango/pg/Ambi/',
                            dataset_list=dataset_list,
                            sampling_num=100,
                            sub_graph_nodes=16,
                            transform_func=transform_func)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Network
vlad = VLADEncoder(checkpoint_path=vlad_pretrain_path, train_feat_extractor=False, train_vlad=True)
reducer = nn.Conv1d(in_channels=512, out_channels=32, kernel_size=1)
with torch.cuda.device(0):
    vlad.cuda()
    reducer.cuda()
    reducer.train()

gat = GATBase(input_feat_channel=32*64*2)
edge_classifier = EdgeClassifier(input_feat_channel=512)
with torch.cuda.device(1):
    gat.cuda()
    gat.train()
    edge_classifier.cuda()
    edge_classifier.train()

# Logger
# log = SummaryWriter(log_dir=)

# Optimizer
optimizer = torch.optim.Adam([{'params': gat.parameters(), 'lr': 1.0e-4},
                              {'params': vlad.parameters(), 'lr': 1.0e-4},
                              {'params': reducer.parameters(), 'lr': 1.0e-4},
                              {'params': edge_classifier.parameters(), 'lr': 1.0e-4}], lr=1.0e-4)
writer = SummaryWriter(log_path)
# Train
counting = 0
for epoch in range(max_epochs):
    epoch_loss = 0
    epoch_acc = 0
    epoch_count = 0
    print("epoch %d" % (epoch+1))
    for itr, data in enumerate(train_loader):
        epoch_count = epoch_count + 1
        counting = counting + 1
        # print("epoch %d" % itr)
        batch_idx, batch_imgs, cam_Es, cam_Cs, out_graph, sub2id, id2sub = data
        N = out_graph.shape[1]
        invmat, num_edges = inv_adjmat(out_graph[0])
        invmat = torch.from_numpy(invmat)

        edge_label = gather_edge_label(out_graph[0], self_edge=False)
        #batch_idx, batch_imgs = imgs2batchsamesize(img_lists)
        optimizer.zero_grad()

        with torch.cuda.device(0):
            cur_dev = torch.cuda.current_device()

            dummy_feat = torch.rand((N, 32, 64)).to(cur_dev)

            for idx, batch in zip(batch_idx, batch_imgs):

                batch_dummy_feat = vlad.forward(batch.to(cur_dev))              # (N, C, n_cluster)
                batch_dummy_feat = reducer.forward(batch_dummy_feat)
                dummy_feat[idx, :, :] = batch_dummy_feat

        with torch.cuda.device(1):
            cur_dev = torch.cuda.current_device()

            dummy_feat = gather_edge_feat2(out_graph[0], dummy_feat)
            dummy_feat = dummy_feat.to(cur_dev).view(num_edges, -1)
            # print(dummy_feat.shape)

            # run gat net
            # TODO: gat might has bug
            g = adjmat2graph(invmat)
            h = gat.forward(g, dummy_feat)

            # run edge classifier
            # edge_res = gather_edge_feat(out_graph[0], edge_classifier, node_feats=h).unsqueeze(0)
            edge_res = edge_classifier.forward(h.view(h.shape[0], -1))
            edge_label = edge_label.view(*edge_res.shape).to(cur_dev)

            loss = criterion.forward(edge_res, edge_label)
            loss.backward()
            acc = accuracy(edge_res, edge_label)
            epoch_loss = epoch_loss + loss.item()
            epoch_acc = epoch_acc + acc
            # print('[Epoch %d, Iteration %d] Loss=%f, Accuracy=%f' % (epoch, itr, loss.item(), acc))
        optimizer.step()
        writer.add_scalar("train/batch_loss", loss.item(),counting)
        writer.add_scalar("train/batch_acc", acc, counting)
    writer.add_scalar("train/loss",epoch_loss/epoch_count, epoch)
    writer.add_scalar("train/acc",epoch_acc/epoch_count, epoch)

