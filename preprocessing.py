import sys, os
from os.path import join, dirname, abspath
import glob
import numpy as np
import pandas as pd
import torch
import torchio as tio
import nibabel as nib
import time
import yaml
import argparse
from addict import Dict
from utils import printProgressBarRatio, PrintTimeStats

class ProcessImg:

    def __init__(self, file):
        ds = nib.load(file)
        self.im = ds.get_fdata().astype(np.float32)

    def MinMaxNorm(self, lower, upper):
        self.im[self.im < lower] = lower
        self.im[self.im > upper] = upper
        return (self.im.astype(np.float32) - lower) / (upper - lower)

class ProcessSeg:

    def __init__(self, dirpath, pack_bits=False):
        self.seg_files = sorted(glob.glob(join(dirpath, "*.nii.gz"), recursive=True))
        self.pack_bits = pack_bits

    def OrdinalEncode(self):
        seg_map = tio.LabelMap(self.seg_files).data
        seg_sum = torch.sum(seg_map, 0)
        bmask = (seg_sum == 0).unsqueeze(0)
        omask = (seg_sum > 1).unsqueeze(0)
        seg_all = torch.concat([bmask, seg_map, omask])
        res_seg = torch.argmax(seg_all, dim=0)
        return res_seg.numpy()

    def OneHotEncode(self):
        seg_list = []
        for file in self.seg_files:
            seg_ds = nib.load(file)
            seg_im = seg_ds.get_fdata().astype(np.uint8)
            seg_im = np.expand_dims(seg_im, 0)
            seg_list.append(seg_im)
        all_seg = np.concatenate(seg_list)
        if self.pack_bits:
            all_seg = np.packbits(all_seg)
        return all_seg

class Metadata:

    def __init__(self, metadata_file):
        self.md = pd.read_csv(metadata_file, sep=";")

    def __len__(self):
        return len(self.md.index)

    def Split(self):
        splits = {}
        for label in list(self.md["split"].unique()):
            splits[label] = self.md.loc[self.md["split"].isin([label]),
                                        "image_id"].tolist()
        return splits

def SaveFile(dest, im, gt):
    np.savez(dest, image=im, gt=gt)
    return

def run(src, dest, md_file, cfgs):
    # Configs
    lower_lim = cfgs.intensity_range[0]
    upper_lim = cfgs.intensity_range[1]
    pack_bits = cfgs.pack_bits

    # create splits dictionary
    meta = Metadata(md_file)
    splits = meta.Split()
    total_no_pats = len(meta)

    # create image and label files
    i = 0
    for lbl, pats in splits.items():
        os.makedirs(join(dirname(dest), lbl), exist_ok=True)
        for pat in pats:
            if pat == "s0864":
                continue
            i += 1
            ct_file = abspath(join(src, pat, "ct.nii.gz"))
            seg_dir = abspath(join(src, pat, "segmentations"))
            process_ct = ProcessImg(ct_file)
            im = process_ct.MinMaxNorm(lower_lim, upper_lim)
            process_seg = ProcessSeg(seg_dir, pack_bits)
            seg = process_seg.OrdinalEncode()
            savepath = join(dest, lbl, pat + ".npz")
            SaveFile(savepath, im, seg)
            printProgressBarRatio(i, total_no_pats, "Patient")
            break
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, metavar="PATH",
                        dest="cfgs", nargs="?", default=None,
                        help=("Configuration yaml file."))
    args = parser.parse_args()
    cfgs = Dict(yaml.load(open(args.cfgs, "r"), Loader=yaml.Loader))
    paths = cfgs.paths
    prep_cfgs = cfgs.preprocess_params
    start = time.time()
    run(paths.data_src, paths.data_dest, paths.metadata_src, prep_cfgs)
    end = time.time()
    PrintTimeStats(start, end)
