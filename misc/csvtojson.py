import numpy as np
import pandas as pd

file = "/home/datasets/Total_Segmentor/meta.csv"
data = pd.read_csv(file, sep=";")
splits = data.groupby("split")["image_id"]
print(splits.head())
