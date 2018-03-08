from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral


img_root = '/home/zeng/data/datasets/oxhand/val/images'
map_root = '/home/zeng/data/datasets/oxhand/val/seg'
output_root = '/home/zeng/data/datasets/oxhand/val/seg_crf'

files = os.listdir(img_root)
it = 1
for img_name in files:
    if not img_name.endswith('.jpg'):
        continue
    pass
    print it
    it += 1
    img = cv2.imread(os.path.join(img_root, img_name))
    probs = cv2.imread(os.path.join(map_root, img_name[:-4] + '.png'), 0).astype('float')
    probs /= 255
    probs = np.stack((probs, 1-probs), 0)

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
    bb = np.array(Q)[0].reshape(img.shape[:2])
    bb = (bb*255).astype(np.uint8)
    bb = Image.fromarray(bb)
    bb.save('%s/%s.png'%(output_root, img_name[:-4]))
