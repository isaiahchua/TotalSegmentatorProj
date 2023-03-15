import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import yaml
from addict import Dict
import argparse
from utils import RandXYZSlices
from dataset import TotalSegmentatorData

class DataViewer:

    def __init__(self, device, dpath, cfgs, batch_size=3):
        self.data = TotalSegmentatorData(device, dpath, cfgs)
        self.dloader = DataLoader(self.data, batch_size=batch_size)

    def __getitem__(self, i):
        pat_name, im, gt = iter(self.dloader)[i]
        return pat_name, im, gt

    def __len__(self):
        return len(self.dloader)

    def NextBatch(self):
        pat_name, im, gt = next(iter(self.dloader))
        return pat_name, im, gt

    def ViewCT(self, i, view_indices=None):
        pat_name, im, _ = iter(self.dloader)[i]
        RandXYZSlices(im, pat_name, view_indices)

    def ViewSeg(self, i, j, view_indices=None):
        pat_name, _, gt = iter(self.dloader)[i]
        RandXYZSlices(gt[j], pat_name, view_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, metavar="PATH",
                        dest="cfgs", nargs="?", default=None,
                        help=("Configuration yaml file."))
    parser.add_argument("index", type=int, metavar="[INT]",
                        help=("Batch index."))
    args = parser.parse_args()
    cfgs = Dict(yaml.load(open(args.cfgs, "r"), Loader=yaml.Loader))
    idx = args.index
    paths = cfgs.paths
    data_cfgs = cfgs.dataset_params
    torch.manual_seed(42)
    if  torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')
    viewer = DataViewer(device, paths.data_dest.train, data_cfgs)
    viewer.ViewCT(idx)



