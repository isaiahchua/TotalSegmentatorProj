import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='1'
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torchio as tio
from model import nnUnet
from dataset import TotalSegmentatorData
from metrics import DiceMax
from utils import OneHot, RandomCrop2

class TestDataset(Dataset):

    def __init__(self, tensor_list, gt_list, device):
        super().__init__()
        self.data = tensor_list
        self.gts = gt_list
        self.device = device
        self.scaling = 0.1
        self.rotation = 10.
        self.gamma =[-0.3, 0.3]
        self.shrink_f = 64
        self.aug_map = defaultdict(self._AugInvalid,
            {
                "crop": tio.Lambda(RandomCrop2),
                "affine": tio.RandomAffine(scales=self.scaling,
                                           degrees=self.rotation),
                "deformation": tio.RandomElasticDeformation(),
                "gamma": tio.RandomGamma(log_gamma=self.gamma),
                "noise": tio.OneOf({tio.RandomNoise(std=(0., 0.1)): 0.75,
                                    tio.RandomBlur(std=(0., 1.)): 0.25}),
                "pad_to_size": tio.EnsureShapeMultiple(self.shrink_f, method="pad"),
            }
        )
        self.aug = ["crop", "affine", "deformation", "gamma", "pad_to_size"]
        self.aug_list = [self.aug_map[key] for key in self.aug]
        self.augment = tio.Compose(self.aug_list)

    def __getitem__(self, i):
        im, gt = self.data[i], self.gts[i]
        pat = tio.Subject({
                "image": tio.ScalarImage(tensor=np.expand_dims(im, 0)),
                "seg": tio.LabelMap(tensor=np.expand_dims(gt, 0)),
                "location": torch.tensor((0, 0, 0, *im.shape), dtype=torch.int64),
                })
        pat_aug = self.augment(pat)
        output =(
            pat_aug.image.data.to(self.device),
            pat_aug.seg.data.to(self.device),
        )
        return output

    def __len__(self):
        return len(self.data)

    def _AugInvalid(self):
        prompt = "Invalid augmentation"
        raise Exception(prompt)

class TestRun:

    def __init__(self, cfgs):
        assert torch.cuda.is_available()
        self.device = torch.device("cuda:0")
        self.world_size = torch.cuda.device_count()
        self.cfgs = cfgs
        self.data_cfgs = cfgs.dataset_params
        self.model_cfgs = cfgs.model_params
        self.optim_cfgs = cfgs.optimizer_params
        self.num_classes = 106
        self.batch_size = 1
        self.epochs = 3

    @staticmethod
    def GenerateRandShapes():
        num_arr = np.arange(60, 300)
        shape_list = np.random.choice(num_arr, (20, 3), replace=True)
        return shape_list.tolist()

    def GenerateData(self):
        shapes = self.GenerateRandShapes()
        self.data = [torch.rand(sh,
                                dtype=torch.float32,
                                device=torch.device("cpu")) for sh in shapes]
        self.truths = [torch.randint(0, 106, sh,
                                     dtype=torch.uint8,
                                     device=torch.device("cpu")) for sh in shapes]

    def GenerateUniformData(self):
        self.data = [torch.rand((128, 128, 128),
                                dtype=torch.float32,
                                device=torch.device("cpu")) for _ in range(20)]
        self.truths = [torch.randint(0, 106, (128, 128, 128),
                                     dtype=torch.uint8,
                                     device=torch.device("cpu")) for _ in range(20)]

    def _SetupDDP(self, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group("nccl", rank=rank, world_size=self.world_size)
        return

    def _ShutdownDDP(self):
        destroy_process_group()
        return

    def _TrainDDP(self, rank):
        self._SetupDDP(rank)
        test_data = TestDataset(self.data, self.truths, rank)
        test_sampler = DistributedSampler(test_data, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, sampler=test_sampler)
        self.test_net = DDP(nnUnet(**self.model_cfgs).to(rank), device_ids=[rank])
        self.test_opt = AdamW(self.test_net.module.parameters(), **self.optim_cfgs)
        for epoch in range(1, self.epochs+1):
            test_sampler.set_epoch(epoch)
            self.test_net.train()
            for batch, (inp, gt) in enumerate(test_loader):
                logits = self.test_net(inp)
                gt_oh = OneHot(gt, self.num_classes - 1)
                test_loss = DiceMax(F.softmax(logits, 1), gt_oh)
                self.test_opt.zero_grad()
                test_loss.backward()
                self.test_opt.step()
                if rank == 0:
                    print(f"batch: {batch + 1}, shape: {inp.detach().shape}, dice: {test_loss.detach().item()}")
        self._ShutdownDDP()

    def RunDDP(self):
        mp.spawn(self._TrainDDP, nprocs=self.world_size)
        return

    def Run(self):
        test_data = TestDataset(self.data, self.truths, self.device)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        self.test_net = nnUnet(**self.model_cfgs).to(self.device)
        self.test_opt = AdamW(self.test_net.parameters(), **self.optim_cfgs)
        for epoch in range(1, self.epochs+1):
            self.test_net.train()
            for batch, (inp, gt) in enumerate(test_loader):
                logits = self.test_net(inp)
                gt_oh = OneHot(gt, self.num_classes - 1)
                test_loss = DiceMax(F.softmax(logits, 1), gt_oh)
                self.test_opt.zero_grad()
                test_loss.backward()
                self.test_opt.step()
                print(f"batch: {batch + 1}, shape: {inp.detach().shape}, dice: {test_loss.detach().item()}")
        return

    def LoopDataset(self):
        test_data = TestDataset(self.data, self.truths, self.device)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        for batch, (inp, gt) in enumerate(test_loader):
            print(f"shapes: {inp.shape}, {gt.shape}")
        return

if __name__ == "__main__":
    pass
