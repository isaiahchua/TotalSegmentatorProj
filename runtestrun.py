import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='1'
from addict import Dict
import yaml
from testrun import TestRun

cfile = "/home/isaiah/TotalSegmentatorProj/config/demo_config.yaml"
cfgs = Dict(yaml.load(open(cfile, "r"), Loader=yaml.Loader))

ddp_test = TestRun(cfgs)
ddp_test.GenerateData()

for i, (inp, gt) in enumerate(zip(ddp_test.data, ddp_test.truths)):
    print(f"Pair {i+1}: {inp.shape}, {gt.shape}")

ddp_test.RunDDP()
