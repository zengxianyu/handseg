import gc
import torch
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from dataset import MyData
from criterion import CrossEntropyLoss2d
from model import Feature, Deconv
from tensorboard import SummaryWriter
from datetime import datetime
import os
import glob
import pdb
from myfunc import make_image_grid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='/home/crow/data/datasets/oxhand/trainval_pix')  # training dataset
parser.add_argument('--check_dir', default='./parameters')  # save checkpoint parameters
parser.add_argument('--pretrained_feature_file', default=None)
parser.add_argument('--resum_ep', type=int, default=-1)  # latest checkpoint, set to -1 if don't need to load checkpoint
parser.add_argument('--bsize', type=int, default=48)  # baatch size
parser.add_argument('--iter_num', type=int, default=20)  # baatch size
opt = parser.parse_args()
print(opt)

resume_ep = opt.resum_ep
train_dir = opt.train_dir
check_dir = opt.check_dir
pretrained_feature_file = opt.pretrained_feature_file

bsize = opt.bsize
iter_num = opt.iter_num  # training iterations

std = [.229, .224, .225]
mean = [.485, .456, .406]

# os.system('rm -rf ./runs/*')
# writer = SummaryWriter('./runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))
#
# if not os.path.exists('./runs'):
#     os.mkdir('./runs')

if not os.path.exists(check_dir):
    os.mkdir(check_dir)

# models
feature = Feature()
feature.cuda()

if pretrained_feature_file:
    feature.load_state_dict(torch.load(pretrained_feature_file))

deconv = Deconv()
deconv.cuda()

if resume_ep >= 0:
    feature_param_file = glob.glob('%s/feature-epoch-%d*.pth'%(check_dir, resume_ep))
    deconv_param_file = glob.glob('%s/deconv-epoch-%d*.pth'%(check_dir, resume_ep))
    feature.load_state_dict(torch.load(feature_param_file[0]))
    deconv.load_state_dict(torch.load(deconv_param_file[0]))

train_loader = torch.utils.data.DataLoader(
    MyData(train_dir, transform=True, crop=True, hflip=True, vflip=True),
    batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)

criterion = CrossEntropyLoss2d(weight=torch.FloatTensor([1.0, 7.0]))
criterion.cuda()

optimizer_deconv = torch.optim.Adam(deconv.parameters(), lr=1e-3)
optimizer_feature = torch.optim.Adam(feature.parameters(), lr=1e-4)


for it in range(resume_ep+1, iter_num):
    for ib, (data, lbl) in enumerate(train_loader):
        inputs = Variable(data).cuda()
        lbl = Variable(lbl.long()).cuda()

        feats = feature(inputs)

        msk = deconv(feats)
        msk = functional.upsample(msk, scale_factor=8)

        loss = criterion(msk, lbl)

        deconv.zero_grad()
        feature.zero_grad()

        loss.backward()

        optimizer_feature.step()
        optimizer_deconv.step()
        # if ib % 20 ==0:
        #     # visulize
        #     image = make_image_grid(inputs.data[:4, :3], mean, std)
        #     writer.add_image('Image', torchvision.utils.make_grid(image), ib)
        #     msk = functional.softmax(msk)
        #     mask1 = msk.data[:4, 1:2]
        #     mask1 = mask1.repeat(1, 3, 1, 1)
        #     writer.add_image('Image2', torchvision.utils.make_grid(mask1), ib)
        #     writer.add_scalar('M_global', loss.data[0], ib)
        print('loss: %.4f (epoch: %d, step: %d)' % (loss.data[0], it, ib))
        del inputs, msk, lbl, loss, feats
        gc.collect()

    filename = ('%s/deconv-epoch-%d-step-%d.pth' % (check_dir, it, ib))
    torch.save(deconv.state_dict(), filename)
    filename = ('%s/feature-epoch-%d-step-%d.pth' % (check_dir, it, ib))
    torch.save(feature.state_dict(), filename)
    print('save: (epoch: %d, step: %d)' % (it, ib))


