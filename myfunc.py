import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

# plt.switch_backend('agg')


def make_image_grid(img, mean, std):
    img = make_grid(img)
    for i in range(3):
        img[i] *= std[i]
        img[i] += mean[i]
    return img

