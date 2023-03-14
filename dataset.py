import sys, os
from os.path import join, isdir, abspath
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torchio as tio
import h5py
import glob
from torch.utils.data import Dataset, DataLoader

class MammoH5Data(Dataset):

    def __init__(self, device, datapath, metadata_path, cfgs):
        super().__init__()
        self.device = device
        self.labels = cfgs.labels
        self.fext = cfgs.file_extension
        self.datafiles = glob.glob(join(abspath(datapath), f"*.{self.fext}"))
        self.metadata_path = abspath(metadata_path)
        self.metadata = None
        self.md_fext_map = {
            "json": self._ReadJson,
            "csv": self._ReadCSV
        }
        self.aug_map = defaultdict(self._AugInvalid,
            {
                "contrast_brightness": tio.RescaleIntensity(out_min_max=(0, 1),
                                                            percentiles=(0, 99.5)),
                "rotate": tio.RandomAffine(),
                "noise": tio.OneOf({tio.RandomNoise(std=(0., 0.1)): 0.75,
                                    tio.RandomBlur(std=(0., 1.)): 0.25}),
            }
        )
        self.aug = cfgs.augmentations
        self.labels = cfgs.labels
        self.ReadMetadata() # reads data into self.metadata
        self.GetDataIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, i):
        """outputs a list [tuple, torch.tensor, tuple, torch.tensor]"""
        im, gt = self._LoadNpz(self.datafiles[i])
        # figure out how to list the classes
        md = self.metadata.loc[key, self.labels].copy(deep=True)
        self.flip_axis = np.random.choice(self.flip_list, 1).tolist()
        if (self.flip_axis[0] == 2) & ("laterality" in self.labels):
            md.loc["laterality"] = self.flip_lat[str(md.loc["laterality"])]
        md = md.to_numpy(np.float32)
        ds_aug = self.Augment(np.expand_dims(ds_arr, axis=(0, 3))).squeeze(-1)
        keyT = torch.tensor(int(key), dtype=torch.int64).to(self.device)
        mdT = torch.from_numpy(md).to(self.device)
        dsT = torch.from_numpy(ds_aug).to(self.device)
        return keyT, dsT, mdT

    def _LoadNpz(self, file):
        ds = np.load(file)
        im = ds["image"]
        seg = np.unpackbits(ds["gt"]).reshape((104, *im.shape))
        return im, seg

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

    def Augment(self, im):
        aug_list = []
        for key in self.aug:
            aug_list.append(self.aug_map[key])
        transform = tio.Compose(aug_list)
        im = transform(im)
        return im

if __name__ == "__main__":
    torch.manual_seed(42)
    if  torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')
    pass
