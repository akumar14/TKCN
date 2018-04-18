# Tree-structured Kronecker Convolutional Networks for Semantic Segmentation
(submitted in ECCV-18)

## Abstract
Most existing semantic segmentation methods employ atrous convolution to enlarge the receptive field of filters, but neglect important
local contextual information. To tackle this issue, we firstly propose a novel Kronecker convolution which adopts Kronecker product to expand its kernel for taking into account the feature vectors neglected by atrous convolutions. Therefore, it can capture local contextual information and enlarge the field of view of filters simultaneously without introducing extra parameters. Secondly, we propose Tree-structured Feature Aggregation (TFA) module which follows a recursive rule to expand and forms a hierarchical structure. Thus, it can naturally learn representations of multi-scale objects and encode hierarchical contextual information in complex scenes. Finally, we design Tree-structured Kronecker Convolutional Networks (TKCN) that employs Kronecker convolution and TFA module. Extensive experiments on three datasets, PASCAL VOC 2012, PASCAL-Context and Cityscapes, verify the effectiveness of our proposed approach.


## Proposed Approach
<div align="left">
  <img src="img/ArchOfNetwork.png" width="700"><br><br>
</div>

### Kronecker Convolution
<div align="left">
  <img src="img/KConv.png" width="700"><br><br>
</div>

### TFA module
<div align="left">
  <img src="img/TFA_module.png" width="700"><br><br>
</div>

## Performance

### Results on PASCAL VOC 2012

<div align="left">
  <img src="img/voc12_result.png" width="700"><br><br>
</div>

### Results on Cityscapes
<div align="left">
  <img src="img/cityscapes_result.png" width="700"><br><br>
</div>

### Results on PASCAL-Context

<div align="left">
  <img src="img/pascalcontext_result.png" width="700"><br><br>
</div>


