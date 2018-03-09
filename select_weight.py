import os

# lws = [27, 30, 33]
# lws = [36, 39, 42]
# lws = [45, 48]
# lws = [27, 30, 33, 36, 39, 42]
lws = [45, 48]
for lw in lws:
    # os.system('CUDA_VISIBLE_DEVICES=3 python train.py --q pix --check_dir %s --lw %d'%(str(lw), lw))
    os.system('python test.py --output_dir /home/zeng/data/datasets/oxhand/val/%d \
    --feat /home/zeng/handseg/%d/feature-epoch-19-step-33.pth \
    --deconv /home/zeng/handseg/%d/deconv-epoch-19-step-33.pth'%(lw, lw, lw))