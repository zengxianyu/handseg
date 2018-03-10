import gc
import torch
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from dataset import MyTestData
import cv2
from criterion import CrossEntropyLoss2d
from model import Deconv
from vgg import Vgg16
from resnet import resnet50
from densenet import densenet121
from PIL import Image
from tensorboard import SummaryWriter
from datetime import datetime
import os
import pdb
from myfunc import make_image_grid
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--i', default='vgg')  # dataset
parser.add_argument('--test_dir', default='/home/zeng/data/datasets/oxhand/val')  # dataset
parser.add_argument('--output_dir', default='/home/zeng/data/datasets/oxhand/val/seg_alt')
parser.add_argument('--feat', default='/home/zeng/handseg/parameters_alt/feature-epoch-19-step-869.pth')
parser.add_argument('--deconv', default='/home/zeng/handseg/parameters_alt/deconv-epoch-19-step-869.pth')
opt = parser.parse_args()
print(opt)

test_dir = opt.test_dir
output_dir = opt.output_dir
feature_param_file = opt.feat
deconv_param_file = opt.deconv

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# models
# models
if 'vgg' == opt.i:
    feature = Vgg16()
elif 'resnet' == opt.i:
    feature = resnet50()
elif 'densenet' == opt.i:
    feature = densenet121()
feature.cuda()
feature.load_state_dict(torch.load(feature_param_file))

deconv = Deconv(opt.i)
deconv.cuda()
deconv.load_state_dict(torch.load(deconv_param_file))

loader = torch.utils.data.DataLoader(
    MyTestData(test_dir, transform=True),
    batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

it = 1
for ib, (data, img_name, img_size) in enumerate(loader):
    print img_name
    it +=1
    inputs = Variable(data).cuda()

    feats = feature(inputs)

    msk = deconv(feats)
    # _, msk = msk.max(1)
    # msk = msk.data[0].cpu().numpy()
    msk = functional.softmax(msk)

    msk = msk.data[0, 1].cpu().numpy()
    msk = (msk*255).astype(np.uint8)

    msk = Image.fromarray(msk)

    msk = msk.resize((img_size[0][0], img_size[1][0]))
    msk.save('%s/%s.png'%(output_dir, img_name[0]), 'PNG')