import sys, os
from os.path import join
import glob
import numpy as np
import nibabel as nib

def LoadSeg(seg_dir):
    seg_list = []
    seg_files = glob.glob(join(seg_dir, "*.nii.gz"), recursive=True)
    for file in seg_files:
        seg_ds = nib.load(file)
        seg_im = seg_ds.get_fdata().astype(np.uint8)
        seg_im = np.expand_dims(seg_im, 0)
        seg_list.append(seg_im)
    all_seg = np.concatenate(seg_list)
    return all_seg

def LoadNpz(file):
    ds = np.load(file)
    im = ds["image"]
    seg = np.unpackbits(ds["gt"]).reshape((104, *im.shape))
    return im, seg

if __name__ == "__main__":
    seg_dir = "/Users/isaiah/datasets/Totalsegmentator_dataset/s0000/segmentations/"
    file = "/Users/isaiah/datasets/Totalsegmentator_preprocessed/val/s0000.npz"
    pass
