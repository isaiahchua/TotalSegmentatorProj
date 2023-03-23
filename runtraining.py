import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='1'
from os.path import abspath, dirname, join, basename, isdir
from addict import Dict
import argparse
import yaml
import torch
import shutil
from train import Train

def SaveConfigFile(src, paths):
    results_path = dirname(abspath(paths.model_ckpts_dest))
    model_id = paths.model_ckpts_dest.split(".", 1)[0][-2:]
    filename = basename(src).split(".", 1)[0] + "_" + model_id + ".yaml"
    cp_path = join(results_path, filename)

    if not isdir(results_path):
        os.makedirs(results_path)
    shutil.copy(src, cp_path)

def TestDataLoader(cfile):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cfgs = Dict(yaml.load(open(abspath(cfile), "r"), Loader=yaml.Loader))
    train = Train(cfgs)
    train.LoopDataset()

def main(cfile):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(42)
    cfgs = Dict(yaml.load(open(abspath(cfile), "r"), Loader=yaml.Loader))
    SaveConfigFile(cfile, cfgs.paths)
    train = Train(cfgs)
    train.RunDDP()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, metavar="PATH",
                        dest="cfile",
                        help=("Configuration file path"))
    args = parser.parse_args()
    main(args.cfile)
