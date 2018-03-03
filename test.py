import gc
import torch
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from dataset import MyTestData
import cv2
from criterion import CrossEntropyLoss2d
from model import Feature, Deconv
from PIL import Image
from tensorboard import SummaryWriter
from datetime import datetime
import os
import pdb
from myfunc import make_image_grid
import numpy as np

test_dir = '/home/zeng/data/datasets/oxhand/test'  # training dataset
output_dir = '/home/zeng/data/datasets/oxhand/test/seg1'  # save checkpoint parameters
feature_param_file = '/home/zeng/handseg/parameters/feature-epoch-19-step-100.pth'
deconv_param_file = '/home/zeng/handseg/parameters/deconv-epoch-19-step-100.pth'

os.system('rm -rf ./runs/*')
writer = SummaryWriter('./runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

if not os.path.exists('./runs'):
    os.mkdir('./runs')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# models
feature = Feature()
feature.cuda()
feature.load_state_dict(torch.load(feature_param_file))

deconv = Deconv()
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
    msk = functional.softmax(msk)

    msk = msk.data[0, 1].cpu().numpy()
    msk = (msk*255).astype(np.uint8)

    msk = Image.fromarray(msk)

    msk = msk.resize((img_size[0][0], img_size[1][0]))
    msk.save('%s/%s.png'%(output_dir, img_name[0]), 'PNG')