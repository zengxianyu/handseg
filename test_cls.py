import gc
import torch
from torch.autograd import Variable
from dataset import MyClsTestData
from model import Classifier
from vgg import Vgg16
from resnet import resnet50
from densenet import densenet121
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--i', default='vgg')  # dataset
parser.add_argument('--test_dir', default='/home/crow/data/datasets/clshand/val')  # dataset
parser.add_argument('--feat', default='/home/crow/handseg/parameters_cls/feature-epoch-9-step-716.pth')
parser.add_argument('--cls', default='/home/crow/handseg/parameters_cls/classifier-epoch-9-step-716.pth')
parser.add_argument('--b', type=int, default=10)  # batch size
opt = parser.parse_args()
print(opt)


def main():
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
    # feature.load_state_dict(torch.load(feature_param_file))
    feature.eval()

    classifier = Classifier(opt.i)
    classifier.cuda()
    # classifier.load_state_dict(torch.load(class_param_file))
    classifier.eval()

    loader = torch.utils.data.DataLoader(
        MyClsTestData(test_dir, transform=True),
        batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)
    acc = eval_acc(feature, classifier, loader)
    print acc


def eval_acc(feature, classifier, loader):
    it = 0.0
    num_correct = 0
    for ib, (data, lbl) in enumerate(loader):
        print("evaluated %d"%it)
        inputs = Variable(data.float()).cuda()
        lbl = lbl.cuda()
        it+=lbl.size(0)
        feats = feature(inputs)
        output = classifier(feats)
        _, pred_lbl = torch.max(output, 1)
        # print pred_lbl.data[0]
        # print output.data
        num_correct += (pred_lbl.data==lbl).sum()
        del pred_lbl, feats, output, inputs
        gc.collect()
    feature.train()
    classifier.train()
    return num_correct / it


if __name__ == '__main__':
    main()