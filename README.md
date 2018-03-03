# Handseg

## Usage
* first, install [PyTorch](https://github.com/pytorch/pytorch) (and [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) if you'd like to visualize training progress in the browser.
* if your tesnsorboard-pytorch is recently installed, ```import tensorboard``` in ```train.py``` should be replaced with ```import tensorboardx```.
* to train a new model, firstly modify the file ```train.py``` and then run it. Set ```train_dir``` to the path to training data , and set ```check_dir``` to the path to save parameters. There should be a folder of .jpg training images, named 'images', and a folder of .png groudtruth maps, named 'masks' in ```train_dir```.
* to test an existing model, firstly modify the file ```test.py``` and then run it. Set ```test_dir``` to the path to test data, set ```feature_param_file``` and ```deconv_param_file``` to the corresponding parameter files. There should be a folder of .jpg input images, named 'images', in ```test_dir```.
