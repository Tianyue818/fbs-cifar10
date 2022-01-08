import numpy as nn
import torch
import torch.nn as nn
from fbs import FBSConv2d

class CifarNet(nn.Module):
    def __init__(self, sparsity_ratio=1.0):
        super().__init__()
        self.layer0 = FBSConv2d(3, 64, 3, stride=1, padding=0, sparsity_ratio=sparsity_ratio)
        self.layer1 = FBSConv2d(64, 64, 3, stride=1, padding=1, sparsity_ratio=sparsity_ratio)
        self.layer2 = FBSConv2d(64, 128, 3, stride=2, padding=1, sparsity_ratio=sparsity_ratio)
        self.layer3 = FBSConv2d(128, 128, 3, stride=1, padding=1, sparsity_ratio=sparsity_ratio)
        self.layer4 = FBSConv2d(128, 128, 3, stride=1, padding=1, sparsity_ratio=sparsity_ratio)
        self.layer5 = FBSConv2d(128, 192, 3, stride=2, padding=1, sparsity_ratio=sparsity_ratio)
        self.layer6 = FBSConv2d(192, 192, 3, stride=1, padding=1, sparsity_ratio=sparsity_ratio)
        self.layer7 = FBSConv2d(192, 192, 3, stride=1, padding=1, sparsity_ratio=sparsity_ratio)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(192, 10)
        self.sparsity_ratio = sparsity_ratio

    # get g for each layer and calculate lasso
    def forward(self, x, inference=False):
        lasso = 0.
        x, g = self.layer0(x, inference)
        lasso += g
        x, g = self.layer1(x, inference)
        lasso += g
        x, g = self.layer2(x, inference)
        lasso += g
        x, g = self.layer3(x, inference)
        lasso += g
        x, g = self.layer4(x, inference)
        lasso += g
        x, g = self.layer5(x, inference)
        lasso += g
        x, g = self.layer6(x, inference)
        lasso += g
        x, g = self.layer7(x, inference)
        lasso += g
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, lasso