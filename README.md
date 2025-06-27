# DenoisingDiffusionProbabilityModel
### Introduction

This is a clone from [DenoisingDiffusionProbabilityModel-ddpm-](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-.git). It is an excellent implement of DDPM trained with CIFAR-10 dataset which let me get a better grasp of DDPM.

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

### Other

Still updating
