import gc
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from dataset import MyData, MyClsData, MyClsTestData
from criterion import CrossEntropyLoss2d
from vgg import Vgg16
from resnet import resnet50
from densenet import densenet121
from model import Classifier
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import glob
import pdb
from myfunc import make_image_grid
import torchvision.datasets as datasets
from test_cls import eval_acc
import argparse
from os.path import expanduser
home = expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--i', default='vgg')  # 'vgg' or 'resnet' or 'densenet'
parser.add_argument('--train_dir', default='%s/data/datasets/clshand'%home)  # training dataset
parser.add_argument('--val_dir', default='/home/crow/data/datasets/clshand/val')  # dataset
parser.add_argument('--check_dir', default='./parameters_cls')  # save checkpoint parameters
parser.add_argument('--r', type=int, default=-1)  # latest checkpoint, set to -1 if don't need to load checkpoint
parser.add_argument('--b', type=int, default=38)  # batch size
parser.add_argument('--e', type=int, default=20)  # training epoches
opt = parser.parse_args()
print(opt)


def main():
    resume_ep = opt.r
    train_dir = opt.train_dir
    check_dir = opt.check_dir
    val_dir = opt.val_dir

    bsize = opt.b
    iter_num = opt.e

    label_weight = [4.858, 17.57]
    std = [.229, .224, .225]
    mean = [.485, .456, .406]

    os.system('rm -rf ./runs2/*')
    writer = SummaryWriter('./runs2/'+datetime.now().strftime('%B%d  %H:%M:%S'))

    if not os.path.exists('./runs2'):
        os.mkdir('./runs2')

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

    classifier = Classifier(opt.i)
    classifier.cuda()

    if resume_ep >= 0:
        feature_param_file = glob.glob('%s/feature-epoch-%d*.pth'%(check_dir, resume_ep))
        classifier_param_file = glob.glob('%s/classifier-epoch-%d*.pth'%(check_dir, resume_ep))
        feature.load_state_dict(torch.load(feature_param_file[0]))
        classifier.load_state_dict(torch.load(classifier_param_file[0]))

    train_loader = torch.utils.data.DataLoader(
        MyClsData(train_dir, transform=True, crop=True, hflip=True, vflip=False),
        batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            MyClsTestData(val_dir, transform=True),
            batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(label_weight))
    criterion.cuda()

    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    optimizer_feature = torch.optim.Adam(feature.parameters(), lr=1e-4)

    acc = 0.0
    for it in range(resume_ep+1, iter_num):
        for ib, (data, lbl) in enumerate(train_loader):
            inputs = Variable(data.float()).cuda()
            lbl = Variable(lbl.long()).cuda()
            feats = feature(inputs)

            output = classifier(feats)
            loss = criterion(output, lbl)

            classifier.zero_grad()
            feature.zero_grad()

            loss.backward()

            optimizer_feature.step()
            optimizer_classifier.step()
            if ib % 20 ==0:
                # image = make_image_grid(inputs.data[:4, :3], mean, std)
                # writer.add_image('Image', torchvision.utils.make_grid(image), ib)
                writer.add_scalar('M_global', loss.data[0], ib)
            print('loss: %.4f (epoch: %d, step: %d), acc: %.4f' % (loss.data[0], it, ib, acc))
            del inputs, lbl, loss, feats
            gc.collect()
        new_acc = eval_acc(feature, classifier, val_loader)
        if new_acc > acc:
            filename = ('%s/classifier-epoch-%d-step-%d.pth' % (check_dir, it, ib))
            torch.save(classifier.state_dict(), filename)
            filename = ('%s/feature-epoch-%d-step-%d.pth' % (check_dir, it, ib))
            torch.save(feature.state_dict(), filename)
            print('save: (epoch: %d, step: %d)' % (it, ib))
            acc = new_acc


if __name__ == '__main__':
    main()