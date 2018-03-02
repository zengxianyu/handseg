import torch
import torch.nn as nn
import torchvision
from torch.nn import init
import pdb


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.main = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(1, stride=1, ceil_mode=True),  # 1/16
            # conv5 features
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),  # 1/16
        )
        vgg16 = torchvision.models.vgg16(pretrained=True)
        L_vgg16 = list(vgg16.features)
        L_self = list(self.main)
        for l1, l2 in zip(L_vgg16, L_self):
            if (isinstance(l1, nn.Conv2d) and
                    isinstance(l2, nn.Conv2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

    def forward(self, x):
        return self.main(x)


class Deconv(nn.Module):
    def __init__(self):
        super(Deconv, self).__init__()
        self.main = nn.Sequential(
            # fc6
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
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
        return self.main(x)