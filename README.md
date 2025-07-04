# DenoisingDiffusionProbabilityModel
### Introduction

This is a clone from [DenoisingDiffusionProbabilityModel-ddpm-](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-.git). It is an excellent implement of DDPM trained with CIFAR-10 dataset which let me get a better grasp of DDPM.

This is just the project I do my own learning and update some notes. If you are interested, welcome clone, else you can clone from [the source project](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-.git).

On the basis, I just do some small changes like turning single-gpu training to multi-gpu training using DDP.

### Files added

Diffusion/Train.py : where we add the code of multi-gpu training.

DDP.md : some notes I made about PyTorch multi-gpu training and multi-processing.

Notes/ : some notes I made during learning model DDPM

### How to run

- You can run Main.py to train the UNet on CIFAR-10 dataset. After training, you can set the parameters in the model config to see the amazing process of DDPM.
- You can run MainCondition.py to train UNet on CIFAR-10. This is for DDPM + Classifier free guidence.

multi-gpu train:

```
torchrun --nproc_per_node=8 Main.py
```

single-gpu train / eval:

```
python Main.py
```

### Reference

diffusion paper: https://proceedings.mlr.press/v37/sohl-dickstein15.html

DDPM paper: https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html

DDPM code: https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-.git

An In-Depth Guide to DDPM: https://learnopencv.com/denoising-diffusion-probabilistic-models/

DDPM理论梳理: https://zhuanlan.zhihu.com/p/560603623

扩散模型Diffusion Model原理: https://www.cnblogs.com/weilonghu/articles/16994430.html
