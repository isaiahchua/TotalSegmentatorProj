import numpy as np
import json

def LoadNpz(file):
    ds = np.load(file)
    im = ds["image"]
    seg = ds["gt"]
    return im, seg

def IndexToLabel(i, label_dict):
    return label_dict[str(i)]

def IndicesToLabels(indices, label_dict):
    return [label_dict[str(i)] for i in indices]

def Run(file, label_file):
    with open(label_file, "r") as f:
        labels = json.load(f)
    im, seg = LoadNpz(file)
    present_lbls = np.unique(seg)
    print(f"CT Image shape, type, dtype: {im.shape}, {type(im), {im.dtype}}")
    print(f"Segmentation shape, type, dtype: {seg.shape}, {type(im)}, {seg.dtype}")
    print(f"Labels Present: {IndicesToLabels(present_lbls, labels)}")

if __name__ == "__main__":
    file = "/Users/isaiah/datasets/Totalsegmentator_preprocessed/val/s0000.npz"
    label_file = "/Users/isaiah/GitHub/TotalSegmentatorProj/metadata/classes.json"
    Run(file, label_file)

