# Handseg

## Usage
* first, install [PyTorch](https://github.com/pytorch/pytorch) (and [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) if you'd like to visualize training progress in the browser.
* if your tesnsorboard-pytorch is recently installed, ```import tensorboard``` in ```train.py``` should be replaced with ```import tensorboardx```.
* to train a new model, firstly modify the file ```train.py``` and then run it. Set ```resume_ep=-1```Set ```train_dir``` to the path to training data , and set ```check_dir``` to the path to save parameters. There should be a folder of .jpg training images, named 'images', and a folder of .png groudtruth maps, named 'masks' in ```train_dir```.
* to test an existing model, firstly modify the file ```test.py``` and then run it. Set ```test_dir``` to the path to test data, set ```feature_param_file``` and ```deconv_param_file``` to the corresponding parameter files. There should be a folder of .jpg input images, named 'images', in ```test_dir```.
* to resume training from checkpoint, specify ```resume_ep``` to the epoch to resume by setting. For example, set ```resume_ep``` to 5 if there are feature-epoch-4-step-xx.pth and deconv-epoch-4-step-xx.pth in ```check_dir```

Train with image-level data: run ```train_cls.py```. I met an error "RuntimeError: DataLoader worker (pid 37825) is killed by signal: Terminated.". After wasting a lot of time trying to find a solution, I finally solved it by running on another computer.

