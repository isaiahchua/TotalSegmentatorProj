import sys, os
import numpy as numpy
import nibabel as nib
import SimpleITK as sitk

file = "/Users/isaiah/datasets/Totalsegmentator_dataset/s0864/ct.nii.gz"
im = sitk.ReadImage(file)
print(im)

file2 = "/Users/isaiah/datasets/Totalsegmentator_dataset/s0600/ct.nii.gz"
im2 = nib.load(file2)
print(im2.header)
