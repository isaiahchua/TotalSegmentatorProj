import os
from os.path import join, basename
import shutil
import glob
import numpy as np

srcpath = "/home/isaiah/TotalSegmentator/preprocessed2/"
destpath = "/home/isaiah/TotalSegmentator/demo/"
extra_names = ["s0287", "s0296"]
extra_files = [join(srcpath, "train", name + ".npz") for name in extra_names]

datafiles = glob.glob(join(srcpath, "train", "*.npz"))
for i, file in enumerate(datafiles):
    name = basename(file).split(".")[0]
    if name in extra_names:
        datafiles.remove(file)
selected_files = np.random.choice(datafiles, 28).tolist()
print(selected_files)
train_files = selected_files[:8] + extra_files
val_files = selected_files[8:18]
test_files = selected_files[18:]
for tf in train_files:
    shutil.copy2(tf, join(destpath, "train"))
for vf in val_files:
    shutil.copy2(vf, join(destpath, "val"))
for sf in test_files:
    shutil.copy2(sf, join(destpath, "test"))
