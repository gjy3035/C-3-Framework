# **C**rowd **C**ounting **C**ode Framework (C^3-Framework) 

An open-source PyTorch code for crowd counting

---

This repo is under development. We will spare my spare time to develop it. 

We expect to complete the initial development in March, 2019. 

**Tips: Before March, 2019, we will not address and accept any issues and pull requests from outside the team.**

## Goal

The purpose of this code is an efficient, flexible framework for supervised crowd counting. At the same time, we provide the performances of some basic networks and classic algorithms on the mainstream datasets.


## Features
- **Convenient development kit**. It support the convenient dev kit on the six maintream datasets.
- **Solid baselines**. It provides some baselines of classic pre-trained models, such as VGG, ResNet, DenseNet and so on. Base on it, you can compare your proposed models' effects.
- **Powerful log**. It does not only record the loss, visulization in Tensorboard, but also save the current code package (including parameters settings). The saved code package can be directly ran to reproduce the experiments at any time. You won't be bothered by forgetting the parameters.


## Performance
Due to linmited spare time and the number of GPUs, I do not plan to conduct some experiments (named as "TBD"). If you are intested in the project, you are welcome to submit your own experimental parameters and results.

|          Method          |           GCC(rd,cc,cl)             | UCF-QNRF  |   SHT A   |  SHT B  |
|--------------------------|-------------------------------------|-----------|-----------|---------|
| MCNN (RGB Image)         |102.2/238.3,  ing       , 176.1/373.9|243.5/364.7|110.6/171.1|23.9/42.7|
| VGG-16 (conv4_3)         |  36.6/88.9,  57.6/133.9,  91.4/222.0|119.3/207.7|71.4/115.7 |10.3/16.5|
| VGG-16 (conv4_3)+decoder |  37.2/91.2,  56.9/138.3,  88.9/220.9|115.2/189.6|71.5/117.6 |10.5/17.4|
| ResNet-50 (layer3)       |  32.4/76.1,  54.5/129.7,  78.3/201.6|    TBD    |    TBD    |7.7/12.6 |
| CSRNet                   |                                     |    TBD    |69.3/111.9 |10.6/16.6|
| SANet                    |  42.4/85.4,  ing                    |    TBD    |    TBD    |12.1/19.2|
| CMTL                     |                                     |    TBD    |    TBD    |   ing   |
| SFCN                     |                                     |           |           |         |
| SFCN $\dag$              |                                     |           |           |         |


|          Method          | WE |UCF50|
|--------------------------|----|-----|
| MCNN (RGB Image)         |TBD | TBD |
| VGG-16 (conv4_3)         |TBD | TBD |
| VGG-16 (conv4_3)+decoder |TBD | TBD |
| ResNet-50 (layer3)       |TBD | TBD |
| CSRNet                   |TBD | TBD |
| SANet                    |TBD | TBD |
| CMTL                     |TBD | TBD |
| SFCN                     |    |     |
| SFCN $\dag$              |    |     |


### data processing code
- [x] GCC
- [x] UCF-QNRF
- [x] ShanghaiTech Part_A
- [x] ShanghaiTech Part_B
- [x] WorldExpo'10
- [x] UCF_CC_50


## Getting Started

### Preparation
- Prerequisites
  - Python 2.7
  - Pytorch 1.0 (some networks only suport 0.4): http://pytorch.org .
  - other libs in ```requirements.txt```, run ```pip install -r requirements.txt```.


- Installation
  - Clone this repo:
    ```
    git clone https://github.com/gjy3035/C-3-Framework.git
    ```

- Data Preparation
  - In ```./datasets/XXX/readme.md```, download our processed dataset or run the ```prepare_XXX.m/.py``` to generate the desity maps. If you want to directly download all processeed data (including Shanghai Tech, UCF-QNRF, UCF_CC_50 and WorldExpo'10), please visit the [**link**](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/EkxvOVJBVuxPsu75YfYhv9UBKRFNP7WgLdxXFMSeHGhXjQ?e=IdyAzA).
  - Place the processed data to ```../ProcessedData```.

- Pretrained Model
  - Some Counting Networks (such as VGG, CSRNet and so on) adopt the pre-trained models on ImageNet. You can download them from [TorchVision](https://github.com/pytorch/vision/tree/master/torchvision/models)
  - Place the processed data to ```../PyTorch_Pretrained```.

- Folder Tree

    ```
    +-- C-3-Framework
    |   +-- datasets
    |   +-- misc
    |   +-- ......
    +-- ProcessedData
    |   +-- shanghaitech_part_A
    |   +-- ......
    +-- PyTorch_Pretrained
    |   +-- resnet50-19c8e357.pth
    |   +-- ......
    ```

