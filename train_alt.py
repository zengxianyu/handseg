import gc
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from dataset import MyData, MyClsData
from criterion import CrossEntropyLoss2d
from model import Feature, Classifier, Deconv
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import glob
import pdb
from myfunc import make_image_grid
import torchvision.datasets as datasets

resume_ep = -1  # set to -1 if don't need to load checkpoint
# resume_ep = 10  # latest checkpoint
cls_train_dir = '/home/zeng/data/datasets/clshand'  # classification data
seg_train_dir = '/home/zeng/data/datasets/oxhand/trainval_pix'  # segmentation data
check_dir = './parameters_alt'  # save checkpoint parameters

bsize = 16  # batch size
iter_num = 20

std = [.229, .224, .225]
mean = [.485, .456, .406]

os.system('rm -rf ./runs2/*')
writer = SummaryWriter('./runs2/'+datetime.now().strftime('%B%d  %H:%M:%S'))

if not os.path.exists('./runs2'):
    os.mkdir('./runs2')

if not os.path.exists(check_dir):
    os.mkdir(check_dir)

# models
feature = Feature()
feature.cuda()

classifier = Classifier()
classifier.cuda()

deconv = Deconv()
deconv.cuda()

if resume_ep >= 0:
    feature_param_file = glob.glob('%s/feature-epoch-%d*.pth'%(check_dir, resume_ep))
    classifier_param_file = glob.glob('%s/classifier-epoch-%d*.pth'%(check_dir, resume_ep))
    deconv_param_file = glob.glob('%s/deconv-epoch-%d*.pth'%(check_dir, resume_ep))
    feature.load_state_dict(torch.load(feature_param_file[0]))
    classifier.load_state_dict(torch.load(classifier_param_file[0]))
    deconv.load_state_dict(torch.load(deconv_param_file[0]))

cls_loader = torch.utils.data.DataLoader(
    MyClsData(cls_train_dir, transform=True, crop=True, hflip=True, vflip=True),
    batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)

seg_loader = torch.utils.data.DataLoader(
    MyData(seg_train_dir, transform=True, crop=True, hflip=True, vflip=True),
    batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)

criterion_cls = nn.BCEWithLogitsLoss()
criterion_cls.cuda()

criterion_seg = CrossEntropyLoss2d(weight=torch.FloatTensor([1.0, 7.0]))
criterion_seg.cuda()

optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)
optimizer_deconv = torch.optim.Adam(deconv.parameters(), lr=1e-3)
optimizer_feature = torch.optim.Adam(feature.parameters(), lr=1e-4)

segIter = iter(seg_loader)
ibs = 0
for it in range(resume_ep+1, iter_num):
    for ib, (data, lbl) in enumerate(cls_loader):
        # train with classification data
        inputs = Variable(data.float()).cuda()
        lbl = Variable(lbl.float()).cuda()
        feats = feature(inputs)
        output = classifier(feats)
        loss_cls = criterion_cls(output, lbl)

        # train with segmentation data
        data, lbl = segIter.next()
        if ibs >= len(segIter):
            segIter = iter(seg_loader)
            ibs = 0
        inputs = Variable(data.float()).cuda()
        lbl = Variable(lbl.long()).cuda()
        msk = deconv(feats)
        msk = functional.upsample(msk, scale_factor=8)

        loss_seg = criterion_seg(msk, lbl)

        classifier.zero_grad()
        deconv.zero_grad()
        feature.zero_grad()
        loss = loss_seg + loss_cls
        loss.backward()

        optimizer_feature.step()
        optimizer_classifier.step()
        optimizer_deconv.step()
        if ib % 20 ==0:
            # visulize
            image = make_image_grid(inputs.data[:4, :3], mean, std)
            writer.add_image('Image', torchvision.utils.make_grid(image), ib)
            msk = functional.softmax(msk)
            mask1 = msk.data[:4, 1:2]
            mask1 = mask1.repeat(1, 3, 1, 1)
            writer.add_image('Image2', torchvision.utils.make_grid(mask1), ib)
            writer.add_scalar('M_global', loss.data[0], ib)
        print('loss: %.4f (epoch: %d, step: %d)' % (loss.data[0], it, ib))
        del inputs, lbl, loss, feats
        gc.collect()

    filename = ('%s/classifier-epoch-%d-step-%d.pth' % (check_dir, it, ib))
    torch.save(classifier.state_dict(), filename)
    filename = ('%s/deconv-epoch-%d-step-%d.pth' % (check_dir, it, ib))
    torch.save(deconv.state_dict(), filename)
    filename = ('%s/feature-epoch-%d-step-%d.pth' % (check_dir, it, ib))
    torch.save(feature.state_dict(), filename)
    print('save: (epoch: %d, step: %d)' % (it, ib))