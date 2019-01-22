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


##  Performance
Due to linmited spare time and the number of GPUs, I do not plan to conduct some experiments (named as "TBD"). If you are intested in the project, you are welcome to submit your own experimental parameters and results.

|          Method          |         GCC(rd,cc,cl)             | UCF-QNRF  |   SHT A   |  SHT B  |
|--------------------------|-----------------------------------|-----------|-----------|---------|
| MCNN (RGB Image)         |102.2/238.3,                       |243.5/364.7|110.6/171.1|23.9/42.7|
| VGG-16 (conv4_3)         |  36.6/88.9, 57.6/133.9, 91.4/222.0|119.3/207.7|71.4/115.7 |10.3/16.5|
| VGG-16 (conv4_3)+decoder |  37.2/91.2, 56.9/138.3, 88.9/220.9|115.2/189.6|71.5/117.6 |10.5/17.4|
| ResNet-50 (layer3)       |  32.4/76.1, 54.5/129.7, 78.3/201.6|    TBD    |    TBD    |7.7/12.6 |
| CSRNet                   |                                   |    TBD    |69.3/111.9 |10.6/16.6|
| SANet                    |                                   |    TBD    |    TBD    |12.1/19.2|
| CMTL                     |                                   |    TBD    |    TBD    |   ing   |



|          Method          | WE |UCF50|
|--------------------------|----|-----|
| MCNN (RGB Image)         |TBD | TBD |
| VGG-16 (conv4_3)         |TBD | TBD |
| VGG-16 (conv4_3)+decoder |TBD | TBD |
| ResNet-50 (layer3)       |TBD | TBD |
| CSRNet                   |TBD | TBD |
| SANet                    |TBD | TBD |
| CMTL                     |TBD | TBD |



### data processing code
- [x] GCC
- [x] UCF-QNRF
- [x] ShanghaiTech Part_A
- [x] ShanghaiTech Part_B
- [x] WorldExpo'10
- [x] UCF_CC_50


