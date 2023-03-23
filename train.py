import os, sys
from os.path import join, isdir, abspath, dirname
from collections import defaultdict
import numpy as np
import json
import csv
from addict import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
import timm
from model import nnUnet
from dataset import TotalSegmentatorData
from metrics import DiceWin, DiceMax
from utils import OneHot

# ViT transfer learning model? Inception net model?

class NullScheduler:

    def __init__(self, lr):
        self.lr = lr

    def get_last_lr(self):
        return [self.lr]

class Train:

    def __init__(self, cfgs):
        assert torch.cuda.is_available()
        self.device = torch.device("cuda:1")
        self.no_gpus = torch.cuda.device_count()

        self.paths = cfgs.paths
        self.model_cfgs = cfgs.model_params
        self.optimizer_cfgs = cfgs.optimizer_params
        self.scheduler_cfgs = cfgs.scheduler_params
        self.data_cfgs = cfgs.dataset_params
        self.train_cfgs = cfgs.run_params

        if self.paths.model_load_src != None:
            self.model_state = torch.load(self.paths.model_load_src)["model"]
            self.optimizer_state = torch.load(self.paths.model_load_src)["optimizer"]
        else:
            self.model_state = None
            self.optimizer_state = None

        self.ckpts_path = abspath(self.paths.model_ckpts_dest)
        self.model_best_path = abspath(self.paths.model_best_dest)
        self.train_report_path = abspath(self.paths.train_report_path)
        self.eval_report_path = abspath(self.paths.eval_report_path)
        self.data_path = abspath(self.paths.data_dest)
        self.train_data_path = join(self.data_path, "train")
        self.val_data_path = join(self.data_path, "val")


        self.train = self.train_cfgs.train
        self.epochs = self.train_cfgs.epochs
        self.batch_size = self.train_cfgs.batch_size
        self.val_size = self.train_cfgs.validation_size
        self.track_freq = self.train_cfgs.tracking_frequency
        self.block_size = self.track_freq * self.batch_size
        self.model_name = self.train_cfgs.model
        self.sel_optim = self.train_cfgs.optimizer
        self.sel_scheduler = self.train_cfgs.scheduler
        self.label_weights = self.train_cfgs.label_weights
        self.loss_weights = self.train_cfgs.loss_weights
        self.num_classes = self.train_cfgs.num_classes

        self.model_dict = defaultdict(self._NoModelError, {
            "nnunet": nnUnet,
        })
        self.optim_dict = defaultdict(AdamW, {
            "adam": Adam,
            "adamw": AdamW,
            "sgd": SGD,
        })
        self.scheduler_dict = defaultdict(None, {
            "exponential": ExponentialLR,
            "cyclic": CyclicLR,
        })

        self.met_name = "PFbeta"
        # self.labels = self.data_cfgs.labels
        # assert len(self.labels) == len(self.label_weights)
        # assert len(self.labels) == self.model_cfgs.num_classes
        # assert len(self.loss_weights) == 2

        self.e = 1e-6

    def _NoModelError(self):
        raise Exception("Not a valid model")


    def _CheckMakeDirs(self, filepath):
        if not isdir(dirname(filepath)): os.makedirs(dirname(filepath))

    def _RemovePath(self, filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass

    def _SaveBestModel(self, state, value):
        self._CheckMakeDirs(self.model_best_path)
        torch.save(state, self.model_best_path)
        print(f"New best model with {self.met_name} = {value} saved to {self.model_best_path}.")

    def _SaveCkptsModel(self, state, value):
        self._CheckMakeDirs(self.ckpts_path)
        torch.save(state, self.ckpts_path)
        print(f"Checkpoint with {self.met_name} = {value} saved to {self.ckpts_path}.")

    def _TrainModelDDP(self, gpu_id):
        self._SetupDDP(gpu_id, self.no_gpus)
        self.train_data = TotalSegmentatorData(gpu_id, self.train_data_path, self.data_cfgs)
        self.val_data = TotalSegmentatorData(gpu_id, self.val_data_path, self.data_cfgs)
        self.train_sampler = DistributedSampler(self.train_data, shuffle=True)
        self.val_sampler = DistributedSampler(self.val_data, shuffle=True)
        self.trainloader = DataLoader(self.train_data, self.batch_size, sampler=self.train_sampler)
        self.validloader = DataLoader(self.val_data, self.val_size, sampler=self.val_sampler)
        self.total_val_size = self.val_sampler.num_samples
        model = self.model_dict[self.model_name](**self.model_cfgs).to(gpu_id)
        if self.model_state != None:
            model.load_state_dict(self.model_state)
            print(f"gpu_id: {gpu_id} - model loaded.")
        self.model = DDP(model, device_ids=[gpu_id])
        self.optimizer = self.optim_dict[self.sel_optim](self.model.parameters(),
                                                         **self.optimizer_cfgs)
        if self.optimizer_state != None:
            self.optimizer.load_state_dict(self.optimizer_state)
        if self.sel_scheduler == None:
            self.scheduler = NullScheduler()
        else:
            self.scheduler = self.scheduler_dict[self.sel_scheduler](self.optimizer,
                                                                      **self.scheduler_cfgs)
        if self.sel_scheduler == "cyclic":
            for _ in range(self.scheduler_cfgs.step_size_up):
                self.scheduler.step()

        # label_weights = torch.tensor(self.label_weights, dtype=torch.float32).to(gpu_id)
        # loss_weights = torch.tensor(self.loss_weights, dtype=torch.float32).to(gpu_id)
        best_score = 0.
        sco = 0.
        block = 1
        dice_scores = []
        losses = []
        if gpu_id == 0:
            self._RemovePath(self.train_report_path)
            self._RemovePath(self.eval_report_path)
            train_log = open(self.train_report_path, "a")
            eval_log = open(self.eval_report_path, "a")
            train_writer = csv.writer(train_log)
            eval_writer = csv.writer(eval_log)
            train_writer.writerow(["epoch", "block", "learning_rate",
                                   "loss", "dice_scores"])
            eval_writer.writerow(["epoch", "samples", "bboxes"
                                  "dice_scores"])
        for epoch in range(1, self.epochs + 1):
            self.train_sampler.set_epoch(epoch)
            self.val_sampler.set_epoch(epoch)
            if self.train:

                # Training loop
                self.model.train()
                for batch, (pat_id, bbox, inp, gt) in enumerate(self.trainloader):
                    last_lr = self.scheduler.get_last_lr()[0]
                    self.optimizer.zero_grad()
                    p = self.model(inp)
                    # remove overlap class
                    gt[gt == self.num_classes - 1] = 0
                    gt_oh = OneHot(gt, self.num_classes - 1)
                    dice = DiceWin(F.softmax(p), gt_oh)
                    ce = F.cross_entropy(p, gt.squeeze(1))
                    loss = ce + dice
                    loss.backward()
                    dice_scores.append(dice.detach().cpu().item())
                    losses.append(loss.detach().cpu().item())
                    self.optimizer.step()
                    if self.sel_scheduler == "cyclic":
                        self.scheduler.step()
                    if (gpu_id == 0) and ((batch + 1) % self.track_freq == 0):
                        block_loss = np.asarray(losses).mean()
                        block_dice = np.asarray(dice_scores).mean()
                        train_writer.writerow([epoch, block, last_lr, block_loss,
                                               block_dice])
                        print((f"epoch {epoch}/{self.epochs} | block: {block}, block_size: {self.block_size} | "
                               f"loss: {block_loss:.5f}, dice: {block_dice:.3f}, "
                               f"{self.met_name}: {sco:.3f}, best: {best_score:.3f}"))
                        block += 1

            # Validation loop;  every epoch
            if self.sel_scheduler != "cyclic":
                self.scheduler.step()
            self.model.eval()
            samples = []
            bboxes = []
            scores = []
            for vbatch, (vpat_id, vbbox, vi, vt) in enumerate(self.validloader):
                samples.append(vpat_id.detach())
                bboxes.append(vbbox.detach())
                scores.append(DiceMax(F.softmax(self.model(vi)), OneHot(vt, self.num_classes)))
            samples = torch.cat(samples)
            bboxes = torch.cat(bboxes)
            scores = torch.cat(scores)
            sam_gather = [torch.zeros((self.total_val_size, 1),
                                       dtype=torch.int64).to(scores.device) for _ in range(self.no_gpus)]
            sam_gather = [torch.zeros((self.total_val_size, 1),
                                       dtype=torch.int64).to(scores.device) for _ in range(self.no_gpus)]
            scores_gather = [torch.zeros((self.total_val_size, self.num_classes - 1),
                                        dtype=torch.float32).to(scores.device) for _ in range(self.no_gpus)]
            dist.all_gather(sam_gather, samples)
            dist.all_gather(bbox_gather, bboxes)
            dist.all_gather(scores_gather, scores)
            all_samples = torch.cat(sam_gather)
            all_bboxes = torch.cat(bbox_gather)
            all_scores = torch.cat(scores_gather)
            if gpu_id == 0:
                sco = all_scores.mean().cpu()
                eval_writer.writerow([epoch, all_samples.cpu().tolist(),
                                      all_bboxes.cpu().tolist(),
                                      all_scores.cpu().tolist()])
                state = {
                    "model": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }
                self._SaveCkptsModel(state, sco)
                if sco > best_score:
                    best_score = sco
                    self._SaveBestModel(state, best_score)
        if gpu_id == 0:
            train_log.close()
            eval_log.close()
        self._ShutdownDDP()
        return

    def _ConfusionMatrix(self, p, gt):
        tp = np.sum((p == 1) & (gt == 1)).astype(int)
        fp = np.sum(p).astype(int) - tp
        tn = np.sum((p == 0) & (gt == 0)).astype(int)
        fn = np.sum(gt).astype(int) - tp
        return tp, fp, tn ,fn

    def _SetupDDP(self, rank, world_size):
        """
        Args:
            rank: Unique identifier of each process
            world_size: Total number of processes
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        return

    def _ShutdownDDP(self):
        destroy_process_group()
        return

    def RunDDP(self):
        mp.spawn(self._TrainModelDDP, nprocs=self.no_gpus)

    def LoopDataset(self):
        temp_data = TotalSegmentatorData(self.device, self.train_data_path, self.data_cfgs)
        temp_loader = DataLoader(temp_data, batch_size=self.batch_size, shuffle=False)
        for name, loc, inp, gt in temp_loader:
            print(f"pat: {name} | im: {inp.shape}, {inp.dtype} | gt: {gt.shape}, {gt.dtype}")
        return

if __name__ == "__main__":
    pass
