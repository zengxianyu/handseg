import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init
import pdb


def nothing(x):
    return x


class Deconv(nn.Module):
    def __init__(self, iii):
        super(Deconv, self).__init__()
        if 'resnet' == iii:
            self.reduce_channel = nn.Conv2d(2048, 512, kernel_size=1)
        elif 'densenet' == iii:
            self.reduce_channel = nn.Conv2d(1024, 512, kernel_size=1)
        else:
            self.reduce_channel = nothing
        self.main = nn.Sequential(
            # fc6
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.ReLU(),
            nn.Dropout(),
            # fc7
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            # fc8
            nn.Conv2d(1024, 2, kernel_size=1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, ceil_mode=True)
        x = self.reduce_channel(x)
        return self.main(x)


class Classifier(nn.Module):
    def __init__(self, iii):
        super(Classifier, self).__init__()
        if 'resnet' == iii:
            self.reduce_channel = nn.Conv2d(2048, 512, kernel_size=1)
        elif 'densenet' == iii:
            self.reduce_channel = nn.Conv2d(1024, 512, kernel_size=1)
        else:
            self.reduce_channel = nothing
        self.main = nn.Sequential(
            # fc6
            nn.Linear(16*16*512, 1024),
            # fc7
            nn.Linear(1024, 1024),
            # fc8
            nn.Linear(1024, 2)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = F.max_pool2d(x, 2, 2, ceil_mode=True)
        x = self.reduce_channel(x)
        bsize = x.size(0)
        x = self.main(x.view(bsize, -1))
        return x