import numpy as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

def global_avgpool2d(x):
    # input : a tensor with size [batch, C, H, W]
    x = torch.mean(torch.mean(x, dim=-1), dim=-1)
    return x  # [batch, C]

def winner_take_all(x, sparsity_ratio):
    # input : a tensor with size [batch, C]
    if sparsity_ratio < 1.0:
        k = ceil((1 - sparsity_ratio) * x.size(-1))
        inactive_idx = (-x).topk(k - 1, 1)[1]
        return x.scatter_(1, inactive_idx, 0)
    else:
        return x


class FBSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, fbs=False, sparsity_ratio=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.fbs = fbs
        self.sparsity_ratio = sparsity_ratio

        if fbs:
            self.channel_saliency_predictor = nn.Linear(in_channels, out_channels)
            nn.init.kaiming_normal_(self.channel_saliency_predictor.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.channel_saliency_predictor.bias, 1.)

            self.bn.weight.requires_grad_(False)

    def forward(self, x, inference=False):
        if self.fbs:
            x, g = self.fbs_forward(x, inference)
            return x, g

        else:
            x = self.original_forward(x)
            return x

    def original_forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

    def fbs_forward(self, x, inference):
        ss = global_avgpool2d(x)  # [batch, C1, H1, W1] -> [batch, C1]
        g = self.channel_saliency_predictor(ss)  # [batch, C1] -> [batch, C2]
        pi = winner_take_all(g, self.sparsity_ratio)  # [batch, C2]

        x = self.conv(x)  # [batch, C1, H1, W1] -> [batch, C2, H2, W2]

        if inference:
            ones, zeros = torch.ones_like(pi), torch.zeros_like(pi)
            pre_mask = torch.where(pi != 0, ones, zeros)
            pre_mask = pre_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.size(2), x.size(3))
            x = x * pre_mask

        x = self.bn(x)
        post_mask = pi.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.size(2), x.size(3))
        x = x * post_mask
        x = F.relu(x)

        return x, torch.mean(torch.sum(g, dim=-1))  # E_x[||g_l(x_l-1)||_1]


