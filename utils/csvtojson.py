import numpy as np
import pandas as pd
import json

file = "/home/dataset/TotalSegmentor/meta.csv"
data = pd.read_csv(file, sep=";")
final_dict = {}
for label in list(data["split"].unique()):
    final_dict[label] = data.loc[data["split"].isin([label]), "image_id"].tolist()
for key, val in final_dict.items():
    print(f"{key}: {len(val)}")
with open("/home/isaiah/TotalSegmentatorProj/splits.json", "w") as f:
    json.dump(final_dict, f)
