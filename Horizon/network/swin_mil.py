import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_
from backbone.SwinTransformer import SwinTransformer
from fast_pytorch_kmeans import KMeans, MultiKMeans
from network.GMM import GaussianMixture

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from utils import batch_index_select, get_index, get_index16
import numpy as np
import math

from einops import rearrange, repeat
import sys


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, proj_dim, 1))

    def forward(self, x):
        return self.proj(x)


class SwinMIL(nn.Module):

    def __init__(self,  num_classes=1, ):
        super(SwinMIL, self).__init__()
        self.num_classes = num_classes

        self.backbone = SwinTransformer(depths=(2, 2, 6, 2), out_indices=(0, 1, 2))

        self.decoder = nn.Conv2d(672, 1, kernel_size=1, stride=1, padding=0)

    def pretrain(self, model, device):
        for w in model.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.xavier_uniform_(w.weight)

        model.backbone.init_weights()
        model_dict = model.state_dict()
        pretrained_dict = torch.load('pretrained_models/swin_tiny_patch4_window7_224.pth')['model']
        new_dict = {'backbone.' + k: v for k, v in pretrained_dict.items() if 'backbone.' + k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)

        return model

    def cam_phase(self, x):

        x1, x2, x3 = self.backbone(x)

        _, _, h, w = x2.size()
        x1 = F.interpolate(x1, size=(h, w), mode="bilinear", align_corners=True)
        x3 = F.interpolate(x3, size=(h, w), mode="bilinear", align_corners=True)

        out = torch.cat([x1, x2, x3], dim=1)
        cam = self.decoder(out)

        score = F.adaptive_max_pool2d(cam, (1, 1)).view(-1)  # F.adaptive_max_pool2d(logits, (1, 1)).view(-1)

        return score, cam

    def forward(self, x):

        scores, cam = self.cam_phase(x)

        return scores, cam
