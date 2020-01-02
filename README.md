# **C**rowd **C**ounting **C**ode Framework (C^3-Framework) 

# Python 3 dev version!

An open-source PyTorch code for crowd counting

---

This repo is under development. We will spare our spare time to develop it. 
If you have any question/suggestion or find any bugs, please submit the issue/PR instead of email or other ways. 

## Technical Blog
- [2019.05] [Chinese Blog] C^3 Framework系列之一：一个基于PyTorch的开源人群计数框架 [[Link](https://zhuanlan.zhihu.com/p/65650998)]

## Goal

The purpose of this code is an efficient, flexible framework for supervised crowd counting. At the same time, we provide the performances of some basic networks and classic algorithms on the mainstream datasets.


## Features
- **Convenient development kit**. It is a convenient dev kit on the six maintream datasets.
- **Solid baselines**. It provides some baselines of some classic pre-trained models, such as AlexNet, VGG, ResNet and so on. Base on it, you can easily compare your proposed models' effects with them.
- **Powerful log**. It does not only record the loss, visualization in Tensorboard, but also save the current code package (including parameters settings). The saved code package can be directly ran to reproduce the experiments at any time. You won't be bothered by forgetting the confused parameters.


## Performance
Due to limited spare time and the number of GPUs, I do not plan to conduct some experiments (named as "TBD"). If you are interested in the project, you are welcomed to submit your own experimental parameters and results. GCC(rd,cc,cl) stand for GCC dataset using **r**an**d**om/**c**ross-**c**amera/**c**ross-**l**ocation/ splitting, respectively.


|          Method          |                GCC(rd,cc,cl)              | UCF-QNRF  |   SHT A   |  SHT B  |
|--------------------------|-------------------------------------------|-----------|-----------|---------|
| MCNN (RGB Image)         |102.2/238.3,     140.3/285.7,   176.1/373.9|243.5/364.7|110.6/171.1|21.5/38.1|
| AlexNet (conv5)          | 46.3/110.9,      83.7/180.3,   101.2/233.6|    TBD    |    TBD    |13.6/21.7|
| VGG-16 (conv4_3)         |  36.6/88.9,      57.6/133.9,    91.4/222.0|119.3/207.7|71.4/115.7 |10.3/16.5|
| VGG-16 (conv4_3)+decoder |  37.2/91.2,      56.9/138.3,    88.9/220.9|115.2/189.6|71.5/117.6 |10.5/17.4|
| ResNet-50 (layer3)       |  32.4/76.1,  **54.5/129.7**,**78.3/201.6**|114.7/205.7|    TBD    |7.7/12.6 |
| ResNet-101 (layer3)      |  31.9/81.4,      56.8/139.5,    86.9/214.2|    TBD    |    TBD    |**7.6/12.2**|
| CSRNet                   |  32.6/74.3,      54.6/135.2,    87.3/217.2|    TBD    |69.3/111.9 |10.6/16.6|
| SANet                    |  42.4/85.4,      79.3/179.9,   110.0/246.0|    TBD    |    TBD    |12.1/19.2|
| CMTL                     |                       -                   |    TBD    |    TBD    |14.0/22.3|
| ResSFCN-101 (SFCN+)      |  **26.8/66.1**,  56.5/139.0,    83.5/211.5|112.67/198.27|    TBD    |7.8/12.6 |


|          Method          | WE |UCF50|
|--------------------------|----|-----|
| MCNN (RGB Image)         |TBD | TBD |
| AlexNet (conv5)          |TBD | TBD |
| VGG-16 (conv4_3)         |TBD | TBD |
| VGG-16 (conv4_3)+decoder |TBD | TBD |
| ResNet-50 (layer3)       |TBD | TBD |
| ResNet-101 (layer3)      |TBD | TBD |
| CSRNet                   |TBD | TBD |
| SANet                    |TBD | TBD |
| CMTL                     |TBD | TBD |
| ResSFCN-101 (SFCN+)      |TBD | TBD |


### data processing code
- [x] GCC
- [x] UCF-QNRF
- [x] ShanghaiTech Part_A
- [x] ShanghaiTech Part_B
- [x] WorldExpo'10
- [x] UCF_CC_50
- [x] UCSD
- [x] Mall

## Getting Started

### Preparation
- Prerequisites
  - Python 3.x
  - Pytorch 1.0 (some networks only support 0.4): http://pytorch.org .
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
  - Place the processed model to ```~/.cache/torch/checkpoints/``` (only for linux OS). 

- Folder Tree

    ```
    +-- C-3-Framework
    |   +-- datasets
    |   +-- misc
    |   +-- ......
    +-- ProcessedData
    |   +-- shanghaitech_part_A
    |   +-- ......
    ```
    

### Training

- set the parameters in ```config.py``` and ```./datasets/XXX/setting.py``` (if you want to reproduce our results, you are recommonded to use our parameters in ```./results_reports```).
- run ```python train.py```.
- run ```tensorboard --logdir=exp --port=6006```.

### Testing

We only provide an example to test the model on the test set. You may need to modify it to test your own models.


### Pretrained Models on GCC

Considering the large-scale GCC, we provide the pretrained models on GCC using random splitting to save the researcher's training time. You can download them from this [link](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/EvH1YcdFBbRJoc7G_Vqv0DwBGmuyXPlH899OrQ54-cybWQ?e=t93edQ). Unfortunately, we've lost the MCNN model trained on GCC, and we will re-train and realease it ASAP.

## Citation
If you find this project is useful for your research, please cite:
```
@inproceedings{wang2019learning,
  title={Learning from Synthetic Data for Crowd Counting in the Wild},
  author={Wang, Qi and Gao, Junyu and Lin, Wei and Yuan, Yuan},
  booktitle={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages-{8198--8207},
  year={2019}
}
```
```
@article{gao2019c,
  title={C$^3$ Framework: An Open-source PyTorch Code for Crowd Counting},
  author={Gao, Junyu and Lin, Wei and Zhao, Bin and Wang, Dong and Gao, Chenyu and Wen, Jun},
  journal={arXiv preprint arXiv:1907.02724},
  year={2019}
}
```
