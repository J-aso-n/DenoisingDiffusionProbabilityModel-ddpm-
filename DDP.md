# PyTorch多卡训练

### （1）DataParallel

```主卡（cuda:0）压力大，速度慢，显存利用率低。```

只需在模型处增加一句

```python
net_model = torch.nn.DataParallel(net_model, device_ids=device_ids)
```

eg.

```
device_ids = [0,1,2,3]
device = torch.device(f"cuda:{device_ids[0]}")
net_model = UNet(...).to(device)
net_model = torch.nn.DataParallel(net_model, device_ids=device_ids)
```

DataParallel 的主模型（主副本）会放在第一个卡上（cuda:0），其它卡是辅助副本。DataParallel 会自动把数据和模型复制到其它卡上。

### （2）DistributedDataParallel（DDP）

```每张卡各自进程，速度快，显存利用率高，几乎线性加速。```

import

```
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
```

初始化分布式环境

```
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
device = torch.device("cuda", local_rank)
```

模型封装

```
net_model = DDP(net_model, device_ids=[local_rank])
```

用Sampler封装数据

```
dataset = CIFAR10(...)
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=False, ... , sampler=sampler)
```

训练时用sampler分发数据，在每个epoch下加上

```
sampler.set_epoch(e)
```

训练循环中只在主进程保存模型

```
if dist.get_rank() == 0:
	torch.save(net_model.module.state_dict(), os.path.join(modelConfig["save_weight_dir"], str(e) + "_.pt"))
```

训练结束后销毁进程组

```
dist.destroy_process_group()
```

启动

```
torchrun --nproc_per_node=8 Main.py
```

若要指定gpu

```
CUDA_VISIBLE_DEVICES=0,1,3,5 torchrun --nproc_per_node=4 Main.py
```

# **多线程**

### **（1）multiprocessing**

import

```
import multiprocessing
```

args是function调用所需的实参

```
for e in epoches:
	args.append((e))
with multiprocessing.Pool(processes=min(8, len(epoches))) as pool:
	results = pool.map(function, args)
```