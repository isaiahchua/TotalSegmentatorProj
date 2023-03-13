import sys, os
from os.path import dirname
import random
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import ast

def RandXYZSlices(ct_file, view_indices=None):
    patient = dirname(ct_file).split("/")[-1]
    try:
        ct_ds = nib.load(ct_file)
        # transpose image to flip it correct order
        ct_im = ct_ds.get_fdata().T
    except:
        ct_ds = sitk.ReadImage(ct_file)
        ct_im = sitk.GetArrayFromImage(ct_ds)
    if view_indices == None:
        # when read as numpy arrays the images are ordered as (z, y, x) instead
        # of (x, y, z)
        z_idx = random.randint(0, ct_im.shape[0])
        y_idx = random.randint(0, ct_im.shape[1])
        x_idx = random.randint(0, ct_im.shape[2])
    else:
        assert isinstance(view_indices, list)
        assert len(view_indices) == 3
        # input from user should still be in (x, y, z)
        x_idx, y_idx, z_idx = view_indices
    # slice images with z, y, x ordering
    z_slice = ct_im[z_idx,:,:]
    y_slice = ct_im[:,y_idx,:]
    x_slice = ct_im[:,:,x_idx]
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(f"{patient} {ct_im.shape[::-1]}")
    axs[0].set_title(f"Sagital view (x={str(x_idx)})")
    axs[0].set_xlabel("y")
    axs[0].set_ylabel("z")
    # plot images with origin at bottom left instead of top left
    axs[0].imshow(x_slice, cmap="bone", origin="lower")
    axs[1].set_title(f"Frontal/Coronal view (y={str(y_idx)})")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("z")
    axs[1].imshow(y_slice, cmap="bone", origin="lower")
    axs[2].set_title(f"Axial/Transverse view (z={str(z_idx)})")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].imshow(z_slice, cmap="bone", origin="lower")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Input 1: path to ct image
    # Input 2: slice indices to view (x, y, z)
    ct_path = sys.argv[1]
    try:
        vidx = ast.literal_eval(sys.argv[2])
    except:
        vidx = None
    RandXYZSlices(ct_path, vidx)
