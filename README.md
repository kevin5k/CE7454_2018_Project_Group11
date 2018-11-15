# A Faster RCNN implementation for Kaggle ships localization


## Introduction

This network is to be trained to detect and localize the ships from the satellite image.
Dataset are obtained from Airbus Ship Detection Challenge with segmentation label. 

For the phase 1 of the project, we train a VGG with normlization and resnet to classify a image to has ship and no ship to classes. This phase help us to evaluate the learning capability of the base network.

In the phase 2, base on the base model we buid our own model on top of a faster-rcnn project from a faster-rcnn project at https://github.com/jwyang/faster-rcnn.pytorch.git. to localize the ships

### Tested environment

Host computer: Ubuntu 18.04
CUDA 10.0 in host

Docker image: nvidia/cuda:9.0-cudnn7-devel 
Docker images: CUDA 9.0
python 3.6

### Data Preparation

A project %ROOT folder should contains at least dataset, models and src folders

Data can be downloaded from Airbus Ship Detection Challenge. The downloaded data then renamed to dataset under $ROOT

run $ROOT/src/data_loader/gt_box_gen.py to convert segmentation mask to bounding box for training.

### Phase 1
Go to %ROOT/src/base, you can either open the base python notebook or run the following on terminal

Training: python base.py -t
Evaluate: python base.py -e
Demo: python base.py -d

For just 1 epoch
The VGG model can achieve 91.5% accuracy 
The Resnet model can achieve 92.2% accuracy

Because VGG comsume lesser memory, and perform almost the same, we choose VGG as our base in the next phase

### Pretrained Model

We used five pretrained models, each of them are saved as 1 to 5 epoch during training

Download models folder from https://drive.google.com/drive/folders/1X-bitFl5E6uOfCE3jnhE3XC5XvIQJbae?usp=sharing and put them into the $ROOT/

### Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

This code has be run with a computer has Nvidia GPU
## Train

To train a faster R-CNN model with vgg16 on kaggle ship dataset, simply run:
python trainval_net.py --cuda

## Test

If you want to evlauate the detection recall and precision of a pre-trained vgg16 model on ships dataset test set, simply run

python test_net.py --cuda

The test data are randomly selected from test set and saved to %ROOT/demo/output after processed

If you want to visualize detected output simply run
python test_net.py --cuda --vis


