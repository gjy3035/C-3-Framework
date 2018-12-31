# **C**rowd **C**ounting **C**ode Framework (C^3-Framework) 

An open-source PyTorch code for crowd counting

---

This repo is under development. We will spare my spare time to develop it. 

We expect to complete the initial development in March, 2019. 

**Tips: Before March, 2019, we will not address and accept any issues and pull requests from outside the team.**

## Goal

The purpose of this code is an effient, flexible framework for supervised crowd counting. At the same time, we provide the performances of some basic networks and claasic algorithms on the mainstream datasets.


## Features
- **Convenient development kit**. It support the convenient  dev kit on the six maintream datasets.
- **Solid baselines**. It provides some baselines of classic pre-trained models, such as VGG, ResNet, DenseNet and so on. Base on it, you can compare your proposed models' effects.
- **Powerful log**. It does not only record the loss, visulization in Tensorboard, but also save the current code package (including parameters settings). The saved code package can be directly ran to reproduce the experiments at any time. You won't be bothered by forgetting the parameters.


##  Performance

|                          | GCC(rd,cc,cl) | UCF-QNRF | SHT A | SHT B | WorldExpo | UCF_CC_50 |
|--------------------------|-----|----------|-------|-------|-----------|-----------|
| MCNN (RGB Image)         |     | 243.5/364.7 |110.6/171.1|23.9/42.7|           |           |
| VGG-16 (conv4_3)         |36.6/88.9, , |119.3/207.7|71.4/115.7|10.3/16.5|           |           |
| VGG-16 (conv4_3)+decoder |     |115.2/189.6|71.5/117.6|10.5/17.4|           |           |
| ResNet-50 (layer3)       |     |          |       |7.7/12.6 |           |           |
| CSRNet                   |     |          |69.3/111.9|10.6/16.6|           |           |


##  Progress

|                          | GCC | UCF-QNRF | SHT A | SHT B | WorldExpo | UCF_CC_50 |
|--------------------------|-----|----------|-------|-------|-----------|-----------|
| MCNN (RGB Image)         |     |  &radic; |&radic;|&radic;|           |           |
| VGG-16 (conv4_3)         |ing  | &radic;  |&radic;|&radic;|           |           |
| VGG-16 (conv4_3)+decoder |ing  | &radic;  |&radic;|&radic;|           |           |
| ResNet-50 (layer3)       |ing  |          |       |&radic;|           |           |
| CSRNet                   |     |          |&radic;|&radic;|           |           |





### data processing code
- [ ] GCC
- [ ] UCF-QNRF
- [ ] Shanghai Tech A
- [ ] Shanghai Tech B
- [ ] WorldExpo'10
- [ ] UCF_CC_50


