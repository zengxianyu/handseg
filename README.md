# Handseg

## Usage
First, install [PyTorch](https://github.com/pytorch/pytorch) (and [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) if you'd like to visualize training progress in the browser.

To train a new model, run 
```
python train.py --train_dir 'path/to/training/data' --check_dir 'path/to/save/parameters'
```
there should be a folder of .jpg training images, named 'images', a folder of .png pixel-level groundtruth maps, named 'pix', and a folder of .png box groudtruth maps, named 'box' in ```path/to/training/data```.
By default, both box and pixel-level annotations will be used for training. Use ```--q box``` or ```--q pix``` to specify training data to be box annotations or pixel-level annotations.


To resume training from a checkpoint, specify ```--r``` to the epoch to resume. For example, run 
```
python train.py --train_dir 'path/to/training/data' --check_dir 'path/to/save/parameters' --r 5
``` 
if there are feature-epoch-5-step-xx.pth and deconv-epoch-5-step-xx.pth in ```'path/to/save/parameters'```.

To test an existing model, run 
```
python test.py --test_dir 'path/to/test/images' --output_dir 'path/to/save/results' --feat 'path/to/feature/parameters' --deconv 'path/to/segmentation/parameters'
```
there should be a folder of .jpg input images, named 'images', in ```test_dir```.

Train with image-level data: run 
```
python train_cls.py --train_dir 'path/to/training/data' --check_dir 'path/to/save/parameters'
```

Option 1: training with image-level data and pixel-level (box-level) data sequentially:
* run ```train_cls.py``` to train the feature extractor and the classifier. 
* run 
```
python train.py --train_dir 'path/to/training/data' --check_dir 'path/to/save/parameters' --f 'path/to/pretrained/feature/file'
``` 
to train with the feature extractor trained in the previous step.

Option 2: training with image-level data and pixel-level (box-level) data alternately:
* run ```train_alt.py```
