import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import pdb
import random
from torch.autograd.variable import Variable

# plt.switch_backend('agg')


def make_image_grid(img, mean, std):
    img = make_grid(img)
    for i in range(3):
        img[i] *= std[i]
        img[i] += mean[i]
    return img


def crf_func(imgs, plabels):
    outputs = []
    for img, probs in zip(imgs, plabels):
        # Example using the DenseCRF class and the util functions
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], 2)

        # get unary potentials (neg log probability)
        U = unary_from_softmax(probs)
        d.setUnaryEnergy(U)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=1,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # Run five inference steps.
        Q = d.inference(5)

        # Find out the most probable class for each pixel.
        # MAP = np.argmax(Q, axis=0)
        # bb = 1-MAP.reshape(img.shape[:2])
        bb = np.array(Q).reshape((2, img.shape[0], img.shape[1]))
        outputs.append(bb)
    outputs = np.array(outputs)
    return outputs


def avg_func(feature, deconv, imgs, num=8):
    imgH = imgs.size(2)
    imgW = imgs.size(3)
    avg_msk = torch.zeros(imgs.size(0), imgH, imgW).cuda()
    H = int(0.9 * imgH)
    H -= H%8
    W = int(0.9 * imgW)
    W -= W%8
    for n in range(num):
        H_offset = random.choice(range(imgH - H))
        W_offset = random.choice(range(imgW - W))
        # H_slice = slice(H_offset, H_offset + H)
        # W_slice = slice(W_offset, W_offset + W)
        _imgs = imgs[:, :, H_offset:H_offset+H, W_offset:W_offset+W]
        feat = feature(_imgs)
        msk = deconv(feat)
        msk = F.upsample(msk, scale_factor=8)
        msk = F.softmax(msk).data[:, 1]
        avg_msk[:, H_offset:H_offset+H, W_offset:W_offset+W] = avg_msk[:, H_offset:H_offset+H, W_offset:W_offset+W] * n / (n+1) + msk / (n+1)
    sb = imgs.data.cpu().numpy()
    sb = sb[:, :, :, ::-1]
    _imgs = Variable(torch.from_numpy(sb.copy()).cuda())
    avg_msk2 = torch.zeros(imgs.size(0), imgH, imgW).cuda()
    for n in range(num):
        H_offset = random.choice(range(imgH - H))
        W_offset = random.choice(range(imgW - W))
        # H_slice = slice(H_offset, H_offset + H)
        # W_slice = slice(W_offset, W_offset + W)
        __imgs = _imgs[:, :, H_offset:H_offset + H, W_offset:W_offset + W]
        feat = feature(__imgs)
        msk = deconv(feat)
        msk = F.upsample(msk, scale_factor=8)
        msk = F.softmax(msk).data[:, 1]
        avg_msk2[:, H_offset:H_offset + H, W_offset:W_offset + W] = avg_msk2[:, H_offset:H_offset + H,
                                                                   W_offset:W_offset + W] * n / (n + 1) + msk / (n + 1)
    sb = avg_msk2.cpu().numpy()
    sb = sb[:, :, ::-1]
    avg_msk2 = torch.from_numpy(sb.copy()).cuda()
    return (avg_msk+avg_msk2)/2
