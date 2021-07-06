import torch
import torch.nn as nn

class QuadKeypointPooling(nn.Module):
    pass

    def __init__(self):
        super(QuadKeypointPooling, self).__init__()
        self.mp = nn.MaxPool2d(8, stride=None, padding=0)

    def forward(self, feat_key_pos, feat):
        """
        :param feat_key_pos: dim: (N, C, 3)
        :param feat: feat map: (N, C, H, W)
        :return:
        """
        x = self.mp(feat)
        return x