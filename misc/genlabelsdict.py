from os.path import join, basename
import glob
import json

datapath = "/Users/isaiah/datasets/Totalsegmentator_dataset/s0000/segmentations/"
savepath = "/Users/isaiah/GitHub/TotalSegmentatorProj/metadata/classes.json"
filelist = sorted(glob.glob(join(datapath, "*.nii.gz")))
labeldict = {}
labeldict["0"] = "background"
for i, file in enumerate(filelist):
    name = basename(file).split(".")[0]
    labeldict[str(i + 1)] = name
labeldict["105"] = "overlap"
with open(savepath, "w") as f:
    json.dump(labeldict, f)
