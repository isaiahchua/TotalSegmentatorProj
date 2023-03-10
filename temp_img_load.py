import sys, os
from os.path import dirname
import random
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt

def RandXYZSlices(ct_file):
    patient = dirname(ct_file).split("/")[-1]
    try:
        ct_ds = nib.load(ct_file)
        ct_im = ct_ds.get_fdata()
    except:
        ct_ds = sitk.ReadImage(ct_file)
        ct_im = sitk.GetArrayFromImage(ct_ds)
    x_idx = random.randint(0, ct_im.shape[0])
    y_idx = random.randint(0, ct_im.shape[1])
    z_idx = random.randint(0, ct_im.shape[2])
    x_slice = ct_im[x_idx,:,:]
    y_slice = ct_im[:,y_idx,:]
    z_slice = ct_im[:,:,z_idx]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{patient} {ct_im.shape}")
    axs[0, 0].set_title(f"x slice {str(x_idx)}")
    axs[0, 0].imshow(x_slice, cmap="bone")
    axs[0, 1].set_title(f"y slice {str(y_idx)}")
    axs[0, 1].imshow(y_slice, cmap="bone")
    axs[1, 0].set_title(f"z slice {str(z_idx)}")
    axs[1, 0].imshow(z_slice, cmap="bone")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ct_path = sys.argv[1]
    RandXYZSlices(ct_path)
