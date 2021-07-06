
from vlad_net import *
import torchvision.models as models
import torch.nn as nn
import torch
from core_dl.module_util import summary_layers

class VLADDimReducer(nn.Module):

    def __init__(self, in_vlad_feat=512, out_feat_dim=32):
        super(VLADDimReducer, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_vlad_feat, out_channels=out_feat_dim)

    def forward(self, x):
        # x is a (N, 512, 64)
        # out is a (N, 32, 64)
        return self.conv(x)

class VLADEncoder(nn.Module):

    def __init__(self, checkpoint_path, num_clusters=64, encoder_dim=512, train_feat_extractor=False, train_vlad=True):
        super(VLADEncoder, self).__init__()
        pretrained = True
        self.num_clusters = num_clusters
        self.encoder_dim = encoder_dim

        # use VGG16 as basic encoder -----------------------------------------------------------------------------------
        encoder = models.vgg16(pretrained=pretrained)
        layers = list(encoder.features.children())[:-1]
        if pretrained:
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False

        encoder = nn.Sequential(*layers)
        encoder.cuda()
        # summary_layers(encoder, input_size=(3, 256, 256))

        model = nn.Module()
        model.add_module('encoder', encoder)

        net_vlad = NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=False)
        model.add_module('pool', net_vlad)

        # load from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

        # split the layers
        layers = list(encoder.children())[:-1]
        encoder_l1tol3 = nn.Sequential(*layers[:22])
        encoder_l3tol4 = nn.Sequential(*layers[22:])
        model.add_module('encoder_l1tol3', encoder_l1tol3)
        model.add_module('encoder_l3tol4', encoder_l3tol4)
        self.model = model

        if train_feat_extractor:
            self.model.encoder.train()
        else:
            self.model.encoder.eval()

        if train_vlad:
            self.model.pool.train()
        else:
            self.model.pool.eval()

    def forward(self, sample: torch.Tensor):
        N = sample.shape[0]
        with torch.no_grad():
            image_encoding = self.model.encoder_l1tol3(sample)
            image_encoding2 = self.model.encoder_l3tol4(image_encoding)


        vlad_encoding = self.model.pool(image_encoding2)
        return vlad_encoding.view(N, self.num_clusters, -1).permute(0, 2, 1).contiguous(), image_encoding

    def forward_img_feats(self, sample: torch.Tensor):
        N = sample.shape[0]
        with torch.no_grad():
            image_encoding = self.model.encoder(sample)
        vlad_encoding = self.model.pool(image_encoding)
        return vlad_encoding.view(N, self.num_clusters, -1).permute(0, 2, 1).contiguous()