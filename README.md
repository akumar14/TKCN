# Tree-structured Kronecker Convolutional Networks for Semantic Segmentation


## Introduction
Most existing semantic segmentation methods employ atrous convolution to enlarge the receptive field of filters, but neglect important
local contextual information. To tackle this issue, we firstly propose a novel Kronecker convolution which adopts Kronecker product to expand its kernel for taking into account the feature vectors neglected by atrous convolutions. Therefore, it can capture local contextual information and enlarge the field of view of filters simultaneously without introducing extra parameters. Secondly, we propose Tree-structured Feature Aggregation (TFA) module which follows a recursive rule to expand and forms a hierarchical structure. Thus, it can naturally learn representations of multi-scale objects and encode hierarchical contextual information in complex scenes. Finally, we design Tree-structured Kronecker Convolutional Networks (TKCN) that employs Kronecker convolution and TFA module. Extensive experiments on three datasets, PASCAL VOC 2012, PASCAL-Context and Cityscapes, verify the effectiveness of our proposed approach.

## Approach


<div align="left">
  <img src="img/ArchOfNetwork.png" width="700"><br><br>
</div>

## Performance
For VOC 2012, we evaluate the proposed TKCN model on test set without external data such as COCO dataset. 

For Cityscapes, the proposed TKCN only trains with the fine-labeled set.

Method | Conference | Backbone | PASCAL VOC 2012 </br> test set  |Cityscapes </br> test set | PASCAL-Context </br> val set
-------|:-------:|:--------:|:--------:|:--------:|:--------:|
DeepLabv2 |-           | ResNet-101  | 79.7   | 70.4   | 45.7
RefineNet |  CVPR2017  | ResNet-101  | 82.4   | 73.6   | 47.1 
SAC       |  ICCV2017  | ResNet-101  | -      | 78.1   | -
PSPNet    |  CVPR2017  | ResNet-101  | 82.6   | 78.4   | 47.8
DUC-HDC   |  WACV2018  | ResNet-101  | -      | 77.6   | -
AAF       |  ECCV2018  | ResNet-101  | 82.2    |79.1   | -
BiSeNet   |  ECCV2018  | ResNet-101  | -      | 78.9   | - 
PSANet    |  ECCV2018  | ResNet-101  |-       | 80.1   | -
DeepLabv3+|  ECCV2018  | Xception    |89.0    |  -     | -
DFN       |  CVPR2018  | ResNet-101  | 82.7   | 79.3   | -
DSSPN     |  CVPR2018  | ResNet-101  |-       | 77.8   | -
CCL       |  CVPR2018  | ResNet-101  |-       | -      | 51.6
EncNet    |  CVPR2018  | ResNet-101  | 82.9   | -      | 51.7
DenseASPP |  CVPR2018  | DenseNet-201 |-      | **80.6**   | -
**TKCN**  |           -| ResNet-101  | **83.2** | 79.5   | **51.8**

**Note that: DeepLabv3+ is pretrained on MS-COCO and JFT.**
## Installation
1. Install PyTorch
  - The code is developed on python3.6.6 on Ubuntu 16.04. (GPU: Tesla K80; PyTorch: 0.5.0a0+a24163a; Cuda: 8.0)
2. Clone the repository
   ```shell
   git clone https://github.com/wutianyiRosun/TKCN.git 
   cd TKCN
   python setup.py install
   ```
3. Pretrained model
   The pretrained model ImageNet_ResNet-101 can be available at [here](https://pan.baidu.com/s/13hhr4xFpp7ldjVJbpX1m4Q). Put it under the folder "./TKCN/tkcn/pretrained_models".
4. Dataset Configuration

  - Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset and convert the dataset to [19 categories](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py). It should have this basic structure. 
  
  ```
├── cityscapes_test_list.txt
├── cityscapes_train_list.txt
├── cityscapes_trainval_list.txt
├── cityscapes_val_list.txt
├── cityscapes_val.txt
├── gtCoarse
│   ├── train
│   ├── train_extra
│   └── val
├── gtFine
│   ├── test
│   ├── train
│   └── val
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
├── license.txt

  ```
  - These .txt files can be downloaded from [here](https://github.com/wutianyiRosun/Semantic_segmentation_datasets/tree/master/Cityscapes) 
  
## Train your own model
###  For Cityscapes
  
  1. training on train+val set
  ```
  cd tkcn
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py  --model tkcnet --backbone resnet101 
  ```

  2. single-scale testing (on test set)
  ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval.py  --model tkcnet --backbone resnet101  --resume-dir cityscapes/model/tkcnet_model_resnet101_cityscapes_gpu6bs6epochs240/TKCNet101 --resume-file checkpoint_240.pth.tar
  ```
  3. multi-scale testing (on test set)
  ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval.py  --model tkcnet --backbone resnet101  --multi-scales  --resume-dir cityscapes/model/tkcnet_model_resnet101_cityscapes_gpu6bs6epochs240/TKCNet101 --resume-file checkpoint_240.pth.tar
  ```
  - For testing, the pretrained model file can be downloaded here: [tkcn_cityscapes_checkpoint_240_ontrainval.pth](https://pan.baidu.com/s/1xTYK2uZ1Aey-oDShdPuOZw)
  

  ## Citation
If TKCN is useful for your research, please consider citing:
```
 
```
  ## License

This code is released under the MIT License. See [LICENSE](LICENSE) for additional details.

## Thanks to the Third Party Libs
https://github.com/zhanghang1989/PyTorch-Encoding

https://github.com/junfu1115/DANet

## Note
The original code for TKCN is based on CAFFE, which will be released later. This is an implementation of TKCN in PyTorch.
