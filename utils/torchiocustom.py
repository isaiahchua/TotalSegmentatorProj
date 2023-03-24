"""Module for custom transforms."""
from typing import Tuple

import random
import numpy as np
import torch
import torchio as tio

class RandomCrop:
    """Random cropping on subject."""

    def __init__(self, roi_size: Tuple):
        """Init.

        Args:
            roi_size: cropping size.
        """
        self.sampler = tio.data.UniformSampler(patch_size=roi_size)

    def __call__(self, subject: tio.Subject) -> tio.Subject:
        """Use patch sampler to crop.

        Args:
            subject: subject having image and label.

        Returns:
            cropped subject
        """
        try:
            for patch in self.sampler(subject=subject, num_patches=1):
                return patch
        except:
            return subject

def RandomCrop2(im: torch.Tensor):
    """Random cropping on 4D tensor which does not fail on smaller images"""
    size = (128, 128, 128)
    bbox = np.asarray(size)
    sh = np.array(im.shape[1:])
    mask = sh < bbox
    bbox[mask] = sh[mask]
    a = [random.randint(0, n) for n in sh - bbox]
    b = a + bbox
    im = im[:, a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    return im

def CenterCrop(im: torch.Tensor):
    """Random cropping on 4D tensor which does not fail on smaller images"""
    size = (128, 128, 128)
    bbox = np.asarray(size)
    sh = np.array(im.shape[1:])
    mask = sh < bbox
    bbox[mask] = sh[mask]
    a = (sh - bbox) // 2
    b = a + bbox
    im = im[:, a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    return im

