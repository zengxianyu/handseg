import gc
import torch
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from dataset import MyClsTestData
import cv2
from criterion import CrossEntropyLoss2d
from model import Classifier
from vgg import Vgg16
from resnet import resnet50
from densenet import densenet121
from PIL import Image
from tensorboard import SummaryWriter
from datetime import datetime
import os
import pdb
from myfunc import make_image_grid, avg_func, crf_func
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--i', default='vgg')  # dataset
parser.add_argument('--test_dir', default='/home/zeng/data/datasets/clshand/val')  # dataset
parser.add_argument('--feat', default='/home/zeng/handseg/parameters_cls/feature-epoch-19-step-365.pth')
parser.add_argument('--cls', default='/home/zeng/handseg/parameters_cls/classifier-epoch-19-step-365.pth')
parser.add_argument('--b', type=int, default=16)  # batch size
opt = parser.parse_args()
print(opt)

test_dir = opt.test_dir
feature_param_file = opt.feat
class_param_file = opt.cls
bsize = opt.b

# models
if 'vgg' == opt.i:
    feature = Vgg16()
elif 'resnet' == opt.i:
    feature = resnet50()
elif 'densenet' == opt.i:
    feature = densenet121()
feature.cuda()
feature.load_state_dict(torch.load(feature_param_file))

classifier = Classifier(opt.i)
classifier.cuda()
classifier.load_state_dict(torch.load(class_param_file))

loader = torch.utils.data.DataLoader(
    MyClsTestData(test_dir, transform=True),
    batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)

it = 0.0
num_correct = 0
for ib, (data, lbl) in enumerate(loader):
    inputs = Variable(data.float()).cuda()
    lbl = lbl.cuda()
    it+=lbl.size(0)
    feats = feature(inputs)
    output = classifier(feats)
    _, pred_lbl = torch.max(output, 1)
    num_correct += (pred_lbl.data==lbl).sum()
print num_correct / it