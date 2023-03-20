import torch
import torch.nn as nn
import torch.nn.functional as F
from model import nnUnet
from utils import OneHot

device = torch.device("cpu")
batch_inp = torch.rand((1, 1, 50, 50, 20), dtype=torch.float32, device=device)
batch_gt = torch.randint(0, 106, (1, 1, 50, 50, 20), dtype=torch.float32, device=device)

batch_gt_oh = OneHot(batch_gt, 106)

model_cfgs = {
    "init_channels": 1,
    "next_channels": 16,
    "num_classes": 105,
    "num_blocks": 4,
}

net = nnUnet(**model_cfgs)
print(net.state_dict.keys())
