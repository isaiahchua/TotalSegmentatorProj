"""Module for custom transforms."""
from typing import Tuple

import torchio as tio


class RandomCrop:
    """Random cropping on subject."""

    def __init__(self, roi_size: Tuple):
        """Init.

        Args:
            roi_size: cropping size.
        """
        self.sampler = tio.data.LabelSampler(patch_size=roi_size, label_name="label")

    def __call__(self, subject: tio.Subject) -> tio.Subject:
        """Use patch sampler to crop.

        Args:
            subject: subject having image and label.

        Returns:
            cropped subject
        """
        for patch in self.sampler(subject=subject, num_patches=1):
            return patch