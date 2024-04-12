import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import BaseModule


class RPN_Modulator(BaseModule):

    def __init__(self, roi_out_size=(7, 7), channels=256, featmap_num=5):
        super(RPN_Modulator, self).__init__()
        self.proj_modulator = nn.ModuleList([
            nn.Conv2d(channels, channels, roi_out_size, padding=0)
            for _ in range(featmap_num)])
        self.proj_out = nn.ModuleList([
            nn.Conv2d(channels, channels, (1, 1), padding=0)
            for _ in range(featmap_num)])

    def forward(self, feats_x, bbox_feats_z):
        out = [self.proj_modulator[k](bbox_feats_z) * feats_x[k] for k in range(len(feats_x))]
        out = [p(o) for p, o in zip(self.proj_out, out)]
        return out

    def init_weights(self):
        for m in self.proj_modulator:
            normal_init(m, std=0.01)
        for m in self.proj_out:
            normal_init(m, std=0.01)


class RCNN_Modulator(BaseModule):

    def __init__(self, channels=256):
        super(RCNN_Modulator, self).__init__()
        self.proj_z = nn.Conv2d(channels, channels, (3, 3), padding=1)
        self.proj_x = nn.Conv2d(channels, channels, (3, 3), padding=1)
        self.proj_out = nn.Conv2d(channels, channels, (1, 1))

    def forward(self, z, x):
        modulator = self.proj_z(z)
        x_hat = self.proj_x(x) * modulator
        out = self.proj_out(x_hat)
        return out

    def init_weights(self):
        normal_init(self.proj_z, std=0.01)
        normal_init(self.proj_x, std=0.01)
        normal_init(self.proj_out, std=0.01)
