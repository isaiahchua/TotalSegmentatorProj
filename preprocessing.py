import sys, os
from os.path import isdir, dirname
from collections import defaultdict
from os.path import join, abspath, dirname, basename
import glob
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import label
from skimage.measure import regionprops
import nibabel
import SimpleITK as sitk
import matplotlib.pyplot as plt
import ast
import argparse
from addict import Dict
import json
import h5py
import yaml
from utils import printProgressBarRatio, PrintTimeStats

# to time script
import time

class CTPreprocess:

    def __init__(self, intensity_range=[None, None], offsets=[0, 0],
                 resolution=None, init_downsample_ratio=3., normalization=False):
        self.irange = intensity_range
        self.ioffset = offsets
        self.res = resolution
        self.init_res = [int(n * init_downsample_ratio) for n in self.res]
        self.normit = normalization

    def MinMaxNormalization(self, im):
        im = (im - self.irange[0])/(self.irange[1] - self.irange[0])
        return im.astype(np.float32)

    def ProportionInvert(self, im, alpha=0.7):
        p = np.sum(im[im == np.max(im)])/np.prod(im.shape)
        if p > alpha:
            im = im.max() - im
        return im

    def PhotometricLabelInvert(self, im, label):
        if label == "MONOCHROME1":
            im = im.max() - im
        return im

    def Compress(self, im, resolution):
        if np.max(resolution) > np.max(im.shape):
            print(f"WARNING: {self.img_id} input image size is smaller than output image size.")
        else:
            end_shape = (np.asarray(resolution) * (im.shape/np.max(im.shape))).astype(np.int16)[::-1]
            im = cv2.resize(im, dsize=end_shape, interpolation=cv2.INTER_NEAREST)
        return im

    def Pad(self, im):
        h, w = im.shape
        diff = np.abs(h - w)
        if h > w:
            top, bot, left, right = [0, 0, diff // 2, diff - (diff // 2)]
        else:
            top, bot, left, right = [diff // 2, diff - (diff // 2), 0, 0]
        im_pad = cv2.copyMakeBorder(im, top, bot, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=0.)
        return im_pad

    def MinMaxIntensityClip(self, im):
        if self.irange[0] == None:
            minimum = np.min(im)
        else:
            minimum = self.irange[0]
        if self.irange[1] == None:
            maximum = np.max(im)
        else:
            maximum = self.irange[1]
        im[im < minimum + self.ioffset[0]] = minimum + self.ioffset[0]
        im[im > maximum + self.ioffset[1]] = maximum + self.ioffset[1]
        return im

    def LargestObjCrop(self, im, mask):
        mask, _ = label(mask)
        # # ignore the first index of stats because it is the background
        _, stats = np.unique(mask, return_counts=True)
        if len(stats) > 1:
            obj_idx = np.argmax(stats[1:]) + 1
            x1,y1,x2,y2 = regionprops(1*(mask == obj_idx))[0].bbox
            res = im[x1:x2, y1:y2]
        else:
            res = im
        return res

    def LoadCT(self, file):
        try:
            # using nibabel for totalsegmentator errors in the dataset
            ds = nibabel.load(file)
            im = ds.get_fdata().astype(np.int32)
        except:
            # except for s0864
            # ds = sitk.ReadImage(file)
            # im = sitk.GetArrayFromImage(ds).T
            pass
        return im

    def LoadSegLabel(self, file):
        try:
            ds = nibabel.load(file)
            im = ds.get_fdata()
        except:
            ds = sitk.ReadImage(file)
            im = sitk.GetArrayFromImage(ds).T
        return im.astype(bool)

def run(src, dest, md_file, cfgs):
    # create splits dictionary
    md = pd.read_csv(md_file, sep=";")
    total_no_pats = len(md.index)
    splits = {}
    for label in list(md["split"].unique()):
        splits[label] = md.loc[md["split"].isin([label]), "image_id"].tolist()
    # make h5 file
    if not isdir(dirname(dest)):
        os.makedirs(dirname(dest))
    hdf = h5py.File(dest, "w")
    # create datasets and labels
    i = 0
    for lbl, pats in splits.items():
        for pat in pats:
            i += 1
            if pat == "s0864":
                continue
            ct_file = abspath(join(src, pat, "ct.nii.gz"))
            seg_dir = abspath(join(src, pat, "segmentations"))
            preprocessor = CTPreprocess(**cfgs)
            ct_im = preprocessor.LoadCT(ct_file)
            ct_im = preprocessor.MinMaxIntensityClip(ct_im)
            ct_im = preprocessor.MinMaxNormalization(ct_im)
            seg_list = []
            for seg_file in glob.glob(join(seg_dir, "*.nii.gz")):
                seg_im = preprocessor.LoadSegLabel(seg_file)
                seg_list.append(np.expand_dims(seg_im, 0))
            seg_all = np.concatenate(seg_list, axis=0)
            hdf.create_dataset(f"{lbl}/{pat}/image", data=ct_im, compression="gzip",
                               compression_opts=9)
            hdf.create_dataset(f"{lbl}/{pat}/gt", data=seg_all, compression="gzip",
                               compression_opts=9)
            printProgressBarRatio(i, total_no_pats, "Patient")
    hdf.close()


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
    pass
