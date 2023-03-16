import sys, os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import yaml
import json
from addict import Dict
import argparse
from utils import RandXYZIndices, RandXYZSlices, PlotXYZSlices
from dataset import TotalSegmentatorData

class DataViewer:

    def __init__(self, device, dpath, cfgs, batch_size=1):
        self.data = TotalSegmentatorData(device, dpath, cfgs)
        self.dloader = DataLoader(self.data, shuffle=True, batch_size=batch_size)

    def __len__(self):
        return len(self.dloader)

    def NextBatch(self):
        pat_name, im, gt = next(iter(self.dloader))
        return pat_name, im, gt

    def ViewNextCT(self, view_indices=None):
        pat_name, im, _ = next(iter(self.dloader))
        cpu_im = im.detach().squeeze().cpu().numpy()
        PlotXYZSlices(cpu_im, pat_name, view_indices)

    def ViewNextSeg(self, view_indices=None):
        pat_name, _, gt = next(iter(self.dloader))
        cpu_gt = gt.detach().squeeze().cpu().numpy()
        PlotXYZSlices(cpu_gt, pat_name, view_indices)

    def ViewNextDataset(self, view_indices=None):
        pat_name, im, gt = next(iter(self.dloader))
        cpu_im = im.detach().squeeze().cpu().numpy()
        cpu_gt = gt.detach().squeeze().cpu().numpy()
        PlotXYZSlices(cpu_im, pat_name, cpu_gt, view_indices)


def IndicesToLabels(indices, label_dict):
    return [label_dict[str(i)] for i in indices]

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
    viewer = DataViewer(device, join(paths.data_dest, "train"), data_cfgs, batch_size=1)

    label_file = "/home/isaiah/TotalSegmentatorProj/metadata/classes.json"
    name, inp, gt = viewer.NextBatch()
    present_lbls = np.unique(gt)
    with open(label_file, "r") as f:
        labels = json.load(f)
    print(f"patient: {name}")
    print(f"CT Image shape, type, dtype: {inp.shape}, {type(inp), {inp.dtype}}")
    print(f"Segmentation shape, type, dtype: {gt.shape}, {type(inp)}, {gt.dtype}")
    print(f"Labels Present: {IndicesToLabels(present_lbls, labels)}")
    # viewer.ViewNextCT()



