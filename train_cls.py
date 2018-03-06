import gc
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from dataset import MyData, MyClsData
from criterion import CrossEntropyLoss2d
from model import Feature, Classifier
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
train_dir = '/home/zeng/data/datasets/clshand'  # classification data
check_dir = './parameters_cls'  # save checkpoint parameters

bsize = 36  # batch size
iter_num = 20

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

if resume_ep >= 0:
    feature_param_file = glob.glob('%s/feature-epoch-%d*.pth'%(check_dir, resume_ep))
    classifier_param_file = glob.glob('%s/classifier-epoch-%d*.pth'%(check_dir, resume_ep))
    feature.load_state_dict(torch.load(feature_param_file[0]))
    classifier.load_state_dict(torch.load(classifier_param_file[0]))

train_loader = torch.utils.data.DataLoader(
    MyClsData(train_dir, transform=True, crop=True, hflip=True, vflip=True),
    batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)

criterion = nn.BCEWithLogitsLoss()
criterion.cuda()

optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)
optimizer_feature = torch.optim.Adam(feature.parameters(), lr=1e-4)

for it in range(resume_ep+1, iter_num):
    for ib, (data, lbl) in enumerate(train_loader):
        pdb.set_trace()
        inputs = Variable(data).cuda()
        lbl = Variable(lbl.float()).cuda()
        feats = feature(inputs)

        output = classifier(feats)
        loss = criterion(output, lbl)

        classifier.zero_grad()
        feature.zero_grad()

        loss.backward()

        optimizer_feature.step()
        optimizer_classifier.step()
        if ib % 20 ==0:
            writer.add_scalar('M_global', loss.data[0], ib)
        print('loss: %.4f (epoch: %d, step: %d)' % (loss.data[0], it, ib))
        del inputs, lbl, loss, feats
        gc.collect()

    filename = ('%s/classifier-epoch-%d-step-%d.pth' % (check_dir, it, ib))
    torch.save(classifier.state_dict(), filename)
    filename = ('%s/feature-epoch-%d-step-%d.pth' % (check_dir, it, ib))
    torch.save(feature.state_dict(), filename)
    print('save: (epoch: %d, step: %d)' % (it, ib))