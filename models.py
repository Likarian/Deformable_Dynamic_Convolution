import math
import torch
import torch.nn.functional as F
from torch import nn
import copy
import torchvision.ops
import conv_layers
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, conv_type = 'static', dynamic = 2):
        super(ResNet18, self).__init__()
        self.conv_type = conv_type
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 100, bias=True)

        if conv_type == 'tdy':
            self.resnet.conv1 = conv_layers.ElementwiseDynamicConv2d(3, 64, kernel_size=7, padding=3, dynamic = dynamic)
        elif conv_type == 'detdy':
            self.resnet.conv1 = conv_layers.DeformableElementwiseDynamicConv2d(3, 64, kernel_size=7, padding=3, dynamic = dynamic)
        else:
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, bias = False)

    def forward(self, x):
        x = self.resnet(x)
        return x

    def get_attention(self, x):
        attention = self.resnet.conv1.get_attention(x)
        return attention


