import sys, os
from os.path import join, isdir, abspath, basename
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torchio as tio
import glob
from torch.utils.data import Dataset, DataLoader
import time
from utils import CenterCrop, RandomCrop2, PrintTimeDecorator


class TotalSegmentatorData(Dataset):

    def __init__(self, device, datapath, cfgs):
        super().__init__()
        self.device = device
        # self.labels = cfgs.labels
        self.fext = cfgs.file_extension
        self.datafiles = sorted(glob.glob(join(abspath(datapath),
                                               f"*.{self.fext}")))

        self.aug = cfgs.augmentations
        self.scaling = cfgs.scaling_factors
        self.rotation = cfgs.rotation_angles
        self.gamma = cfgs.gamma_range
        self.shrink_f = cfgs.model_shrinking_factor
        self.no_ctrl_pts = cfgs.num_control_points
        self.max_disp = cfgs.max_displacement
        self.aug_map = defaultdict(self._AugInvalid,
            {
                "rand_crop": tio.Lambda(RandomCrop2),
                "center_crop": tio.Lambda(CenterCrop),
                "affine": tio.RandomAffine(scales=self.scaling,
                                           degrees=self.rotation),
                "deformation": tio.RandomElasticDeformation(max_displacement=self.max_disp,
                                                            num_control_points=self.no_ctrl_pts),
                "gamma": tio.RandomGamma(log_gamma=self.gamma),
                "noise": tio.OneOf({tio.RandomNoise(std=(0., 0.1)): 0.75,
                                    tio.RandomBlur(std=(0., 1.)): 0.25}),
                "pad_to_size": tio.EnsureShapeMultiple(self.shrink_f, method="pad"),
            }
        )
        if self.aug != None:
            self.aug_list = [self.aug_map[key] for key in self.aug]
            self.augment = tio.Compose(self.aug_list)
        else:
            self.augment = None

        # self.metadata_path = abspath(metadata_path)
        # self.metadata = None
        # self.md_fext_map = {
            # "json": self._ReadJson,
            # "csv": self._ReadCSV
        # }
        # self.ReadMetadata()
        # self.GetDataIds()

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, i):
        """outputs a list [tuple, torch.tensor, tuple, torch.tensor]"""
        file = self.datafiles[i]
        pat_name = int(basename(file).split(".")[0][1:])
        im, gt = self._LoadNpz(self.datafiles[i])
        pat = tio.Subject({
                "image": tio.ScalarImage(tensor=np.expand_dims(im, 0)),
                "seg": tio.LabelMap(tensor=np.expand_dims(gt, 0)),
                "location": torch.tensor((0, 0, 0, *im.shape), dtype=torch.int64),
                })
        if self.augment != None:
            pat = self.augment(pat)
        output =(
            pat_name,
            pat.location.data,
            pat.image.data,
            pat.seg.data,
        )
        return output

    def _LoadNpz(self, file):
        ds = np.load(file)
        return ds["image"], ds["gt"]

    def _ReadCSV(self):
        self.metadata = pd.read_csv(self.metadata_path)
        return

    def _ReadJson(self):
        self.metadata = pd.read_json(self.metadata_path, orient="index",
                                      convert_axes=False, convert_dates=False)
        return

    def ReadMetadata(self):
        fext = self.metadata_path.split(".", 1)[-1]
        self.md_fext_map.get(fext, lambda: "Invalid file extension for metadata")()
        return

    def GetDataIds(self):
        self.img_ids = list(self.metadata.index.astype(str))
        return

    def _AugInvalid(self):
        prompt = "Invalid augmentation"
        raise Exception(prompt)


if __name__ == "__main__":
    # torch.manual_seed(42)
    # if  torch.cuda.is_available():
        # device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
        # device = torch.device("mps")
    # else:
        # device = torch.device('cpu')
    pass
