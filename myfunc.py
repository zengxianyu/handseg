import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import pdb

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
        d.addPairwiseEnergy(feats, compat=3,
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
