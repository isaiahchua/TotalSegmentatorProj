import os, sys
from os.path import join, isdir, abspath, dirname
from collections import defaultdict
import shutil
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
from utils import OneHot, TimeFuncDecorator

class NullScheduler:

    def __init__(self, lr):
        self.lr = lr

    def get_last_lr(self):
        return [self.lr]

class Train:

    def __init__(self, cfgs):
        assert torch.cuda.is_available()
        self.no_gpus = torch.cuda.device_count()

        self.paths = cfgs.paths
        self.model_cfgs = cfgs.model_params
        self.optimizer_cfgs = cfgs.optimizer_params
        self.scheduler_cfgs = cfgs.scheduler_params
        self.train_data_cfgs = cfgs.train_dataset_params
        self.eval_data_cfgs = cfgs.validation_dataset_params
        self.train_cfgs = cfgs.run_params
        self.prev_model = self.paths.model_load_src

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
        self.met_name = self.train_cfgs.metric
        self.sel_optim = self.train_cfgs.optimizer
        self.sel_scheduler = self.train_cfgs.scheduler
        self.label_weights = self.train_cfgs.label_weights
        self.lw = self.train_cfgs.loss_weights
        self.lw_decay = self.train_cfgs.loss_weights_decay
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

        # self.labels = self.data_cfgs.labels
        # assert len(self.labels) == len(self.label_weights)
        # assert len(self.labels) == self.model_cfgs.num_classes
        assert len(self.lw) == 2
        assert len(self.lw_decay) == 2

        self.e = 1e-6

    def _NoModelError(self):
        raise Exception("Not a valid model")


    def _CheckMakeDirs(self, filepath):
        if not isdir(dirname(filepath)): os.makedirs(dirname(filepath))

    def _RemoveDir(self, dirpath):
        try:
            shutil.rmtree(dirpath)
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

    def _SetupLogs(self, gpu_id):
        self._RemoveDir(self.train_report_path)
        self._RemoveDir(self.eval_report_path)
        os.makedirs(self.train_report_path)
        os.makedirs(self.eval_report_path)
        train_log = open(join(self.train_report_path, f"rank{gpu_id}.csv"), "a")
        eval_log = open(join(self.eval_report_path, f"rank{gpu_id}.csv"), "a")
        train_writer = csv.writer(train_log)
        eval_writer = csv.writer(eval_log)
        train_writer.writerow(["epoch", "block", "learning_rate",
                               "loss", "cross_entropy_scores", "dice_scores"])
        eval_writer.writerow(["epoch", "samples", "bboxes",
                              "cross_entropy_scores", "dice_scores"])
        return train_log, eval_log, train_writer, eval_writer

    def _SetupData(self, mode, device):
        self.train_data = TotalSegmentatorData(device, self.train_data_path, self.train_data_cfgs)
        self.val_data = TotalSegmentatorData(device, self.val_data_path, self.eval_data_cfgs)
        if mode == "DDP":
            self.train_sampler = DistributedSampler(self.train_data, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_data, shuffle=True)
            self.trainloader = DataLoader(self.train_data, self.batch_size, sampler=self.train_sampler)
            self.validloader = DataLoader(self.val_data, self.val_size, sampler=self.val_sampler)
            self.total_val_size = len(self.val_sampler)
        else:
            self.trainloader = DataLoader(self.train_data, self.batch_size, shuffle=True)
            self.validloader = DataLoader(self.val_data, self.val_size, shuffle=True)
            self.total_val_size = len(self.validloader)

    def _SetupModelAndOptim(self, mode, gpu_id, device):
        model = self.model_dict[self.model_name](**self.model_cfgs).to(device)
        if self.prev_model != None:
            self.prev_model = torch.load(self.prev_model, map_location=device)
            model_state = self.prev_model["model"]
            optimizer_state = self.prev_model["optimizer"]
            model.load_state_dict(model_state)
            print(f"gpu_id: {gpu_id} - model loaded.")
        else:
            model_state = None
            optimizer_state = None
        if mode == "DDP":
            self.model = DDP(model, device_ids=[gpu_id])
        else:
            self.model = model
        self.optimizer = self.optim_dict[self.sel_optim](self.model.parameters(),
                                                         **self.optimizer_cfgs)
        if optimizer_state != None:
            self.optimizer.load_state_dict(optimizer_state)
        del model_state
        del optimizer_state
        del self.prev_model
        torch.cuda.empty_cache()
        return

    def _SetupScheduler(self):
        if self.sel_scheduler == None:
            self.scheduler = NullScheduler(self.optimizer_cfgs.lr)
        else:
            self.scheduler = self.scheduler_dict[self.sel_scheduler](self.optimizer,
                                                                      **self.scheduler_cfgs)
        if self.sel_scheduler == "cyclic":
            for _ in range(self.scheduler_cfgs.step_size_up):
                self.scheduler.step()

    def _TrainLoop(self, mode, gpu_id, train_log, eval_log, train_writer, eval_writer):
        best_score = 0.
        sco = 0.
        block = 1
        dice_scores = []
        ce_scores = []
        losses = []
        for epoch in range(1, self.epochs + 1):
            if mode == "DDP":
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
                    mask=~torch.eq(gt, 105)
                    gt[gt == 105] = 0
                    gt_oh = OneHot(gt, self.num_classes - 1)
                    ce = F.cross_entropy(p, gt.squeeze(1))
                    dice = DiceMax(F.softmax(p, 1), gt_oh, mask)
                    loss = self.lw[0]*self.lw_decay[0]**(epoch-1)*ce + self.lw[1]*self.lw_decay[1]**(epoch-1)*dice
                    loss.backward()
                    dice_scores.append(1. - dice.detach().cpu().item())
                    ce_scores.append(ce.detach().cpu().item())
                    losses.append(loss.detach().cpu().item())
                    self.optimizer.step()
                    if self.sel_scheduler == "cyclic":
                        self.scheduler.step()
                    if (batch + 1) % self.track_freq == 0:
                        block_loss = np.asarray(losses).mean()
                        block_dice = np.asarray(dice_scores).mean()
                        block_ce = np.asarray(ce_scores).mean()
                        train_writer.writerow([epoch, block, last_lr, block_loss,
                                               block_ce, block_dice])
                        if mode != "DDP":
                            print((f"epoch {epoch}/{self.epochs} | block: {block}, block_size: {self.block_size} | "
                                   f"loss: {block_loss:.5f}, cross_entropy: {block_ce}, block_dice: {block_dice:.3f}, "
                                   f"{self.met_name}: {sco:.3f}, best: {best_score:.3f}"))
                        elif gpu_id == 0:
                            print((f"epoch {epoch}/{self.epochs} | block: {block}, block_size: {self.block_size} | "
                                   f"loss: {block_loss:.5f}, cross_entropy: {block_ce}, block_dice: {block_dice:.3f}, "
                                   f"{self.met_name}: {sco:.3f}, best: {best_score:.3f}"))
                        dice_scores = []
                        ce_scores = []
                        losses = []
                        block += 1

            # Validation loop;  every epoch
            if (self.sel_scheduler != "cyclic") and (self.sel_scheduler != None):
                self.scheduler.step()
            samples = []
            bboxes = []
            dice_scores = []
            ce_scores = []
            self.model.eval()
            with torch.no_grad():
                for vbatch, (vpat_id, vbbox, vi, vt) in enumerate(self.validloader):
                    samples.append(vpat_id.detach().item())
                    bboxes.append(vbbox.detach().tolist())
                    pv = self.model(vi)
                    mask=~torch.eq(gt, 105)
                    gt[gt == 105] = 0
                    gt_oh = OneHot(gt, self.num_classes - 1)
                    ce = F.cross_entropy(p, gt.squeeze(1))
                    dice = DiceMax(F.softmax(p, 1), gt_oh, mask)
                    vt[vt == self.num_classes - 1] = 0
                    dice_scores.append(1. - dice.detach().item())
                    ce_scores.append(ce.detach().item())
            eval_writer.writerow([epoch, samples, bboxes, ce_scores, dice_scores])
            if mode != "DDP":
                sco = np.asarray(dice_scores).mean()
                state = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }
                self._SaveCkptsModel(state, sco)
                if sco > best_score:
                    best_score = sco
                    self._SaveBestModel(state, best_score)
            elif gpu_id == 0:
                sco = np.asarray(dice_scores).mean()
                state = {
                    "model": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }
                self._SaveCkptsModel(state, sco)
                if sco > best_score:
                    best_score = sco
                    self._SaveBestModel(state, best_score)
        train_log.close()
        eval_log.close()

    def _TrainModel(self, gpu_id, mode=""):
        if mode == "DDP":
            self._SetupDDP(gpu_id, self.no_gpus)
        train_log, eval_log, train_writer, eval_writer = self._SetupLogs(gpu_id)
        device = torch.device("cuda", gpu_id)
        torch.cuda.set_device(device)
        self._SetupData(mode, device)
        self._SetupModelAndOptim(mode, gpu_id, device)
        self._SetupScheduler()
        self._TrainLoop(mode, gpu_id, train_log, eval_log, train_writer, eval_writer)
        if mode == "DDP":
            self._ShutdownDDP()
        return

    @TimeFuncDecorator(True)
    def Run(self, gpu_id=0):
        self._TrainModel(gpu_id)

    @TimeFuncDecorator(True)
    def RunDDP(self):
        mp.spawn(self._TrainModel, args=("DDP"), nprocs=self.no_gpus)

    @TimeFuncDecorator(True)
    def LoopDataset(self):
        temp_data = TotalSegmentatorData(torch.device("cpu"), self.train_data_path, self.train_data_cfgs)
        temp_loader = DataLoader(temp_data, batch_size=self.batch_size, shuffle=False)
        for name, loc, inp, gt in temp_loader:
            print(f"pat: {name} | im: {inp.shape}, {inp.dtype} | gt: {gt.shape}, {gt.dtype}")
        return

if __name__ == "__main__":
    pass
