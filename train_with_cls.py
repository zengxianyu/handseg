import gc
import torch
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from dataset import MyClsBoxPixData
from criterion import CrossEntropyLoss2d
from model import Deconv
from vgg import Vgg16
from resnet import resnet50
from densenet import densenet121
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import glob
import pdb
from myfunc import make_image_grid, crf_func
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--i', default='vgg')  # 'vgg' or 'resnet' or 'densenet'
parser.add_argument('--q', default='')  # '' or 'pix' or 'box'
parser.add_argument('--train_dir', default='/home/zeng/data/datasets/oxhand/train')  # training dataset
parser.add_argument('--check_dir', default='./parameters_with_cls')  # save checkpoint parameters
parser.add_argument('--f', default=None)
parser.add_argument('--r', type=int, default=19)  # latest checkpoint, set to -1 if don't need to load checkpoint
parser.add_argument('--b', type=int, default=8)  # batch size
parser.add_argument('--e', type=int, default=40)  # epoches
opt = parser.parse_args()
print(opt)

label_weight = [1, 25]

# label_weights = {'box':[1.01, 89.88], 'pix':[1.01, 80.69]}
#
# if opt.q:
#     opt.check_dir = '%s_%s'%(opt.check_dir, opt.q)
#     label_weight = label_weights[opt.q]

resume_ep = opt.r
train_dir = opt.train_dir
check_dir = opt.check_dir
pretrained_feature_file = opt.f

bsize = opt.b
iter_num = opt.e  # training iterations

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
if 'vgg' == opt.i:
    feature = Vgg16(pretrained=True)
elif 'resnet' == opt.i:
    feature = resnet50(pretrained=True)
elif 'densenet' == opt.i:
    feature = densenet121(pretrained=True)
feature.cuda()

deconv = Deconv(opt.i)
deconv.cuda()

if pretrained_feature_file:
    feature.load_state_dict(torch.load(pretrained_feature_file))


if resume_ep >= 0:
    feature_param_file = glob.glob('%s/feature-epoch-%d*.pth'%(check_dir, resume_ep))
    deconv_param_file = glob.glob('%s/deconv-epoch-%d*.pth'%(check_dir, resume_ep))
    feature.load_state_dict(torch.load(feature_param_file[0]))
    deconv.load_state_dict(torch.load(deconv_param_file[0]))

train_loader = torch.utils.data.DataLoader(
    MyClsBoxPixData(train_dir, transform=True, crop=True, hflip=True, vflip=False, source=opt.q),
    batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)

criterion = CrossEntropyLoss2d(weight=torch.FloatTensor(label_weight))
criterion.cuda()

optimizer_deconv = torch.optim.Adam(deconv.parameters(), lr=1e-3)
optimizer_feature = torch.optim.Adam(feature.parameters(), lr=1e-4)


for it in range(resume_ep+1, iter_num):
    for ib, (data, lbl) in enumerate(train_loader):
        lbl = lbl.long()
        if lbl.max() == 2:
            sb = (lbl[:, 0, 0] == 2).nonzero().view(-1)
            pseudo_inputs = Variable(data[sb, :, :, :]).cuda()
            feats = feature(pseudo_inputs)
            pseudo_lbl = deconv(feats)
            pseudo_lbl = functional.upsample(pseudo_lbl, scale_factor=8)
            pseudo_lbl = functional.softmax(pseudo_lbl).data.cpu().numpy()
            imgs = data[sb, :, :, :].numpy()

            pseudo_lbl = crf_func(imgs.transpose(0, 2, 3, 1), pseudo_lbl)
            pseudo_lbl = torch.from_numpy(pseudo_lbl[:, 1])

            pseudo_lbl[pseudo_lbl>0.2] = 1
            pseudo_lbl[pseudo_lbl<=0.2] = 0
            lbl[sb, :, :] = pseudo_lbl.long()
        del feats
        gc.collect()

        lbl = Variable(lbl).cuda()
        inputs = Variable(data).cuda()

        feats = feature(inputs)

        msk = deconv(feats)
        msk = functional.upsample(msk, scale_factor=8)

        loss = criterion(msk, lbl)

        deconv.zero_grad()
        feature.zero_grad()

        loss.backward()

        optimizer_feature.step()
        optimizer_deconv.step()

        # if ib % 1 ==0:
        #     # visulize
        #     image = make_image_grid(inputs.data[:4, :3], mean, std)
        #     writer.add_image('Image', torchvision.utils.make_grid(image), ib)
        #     msk = functional.softmax(msk)
        #     mask1 = msk.data[:4, 1:2]
        #     mask1 = mask1.repeat(1, 3, 1, 1)
        #     writer.add_image('Image2', torchvision.utils.make_grid(mask1), ib)
        #     mask1 = lbl.data[:4].unsqueeze(1).float()
        #     mask1 = mask1.repeat(1, 3, 1, 1)
        #     writer.add_image('Label', torchvision.utils.make_grid(mask1), ib)
        #     writer.add_scalar('M_global', loss.data[0], ib)
        print('loss: %.4f (epoch: %d, step: %d)' % (loss.data[0], it, ib))
        del inputs, msk, lbl, loss, feats
        gc.collect()

    filename = ('%s/deconv-epoch-%d-step-%d.pth' % (check_dir, it, ib))
    torch.save(deconv.state_dict(), filename)
    filename = ('%s/feature-epoch-%d-step-%d.pth' % (check_dir, it, ib))
    torch.save(feature.state_dict(), filename)
    print('save: (epoch: %d, step: %d)' % (it, ib))