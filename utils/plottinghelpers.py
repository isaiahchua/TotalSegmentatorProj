import numpy as np
import pandas as pd
import torch
import ast

def MovingAvg(v, size):
    avg = np.empty_like(v)
    avg[:size] = np.mean(v[:2*size])
    avg[len(v)-2*size+1:len(v)] = np.mean(v[len(v)-2*size+1:len(v)])
    for i in range(size, len(v) - size):
        avg[i] = np.mean(v[i-size:i+size])
    return avg

def ConvertTensors(df, labels):
    for label in labels:
        for i in range(len(df.index)):
            val = eval(df.loc[i, label]).cpu()
            if len(val) > 1:
                df.loc[i,label] = val.tolist()
            else:
                df.loc[i,label] = val.item()

def StrListColumnToArray(df: pd.DataFrame, src_column: str):
    return np.array(df.loc[:, src_column].apply(ast.literal_eval).to_list())

def ArrayToColumns(df: pd.DataFrame, arr: np.ndarray, new_columns: list):
    assert arr.shape[1] == len(new_columns)
    return pd.concat((df, pd.DataFrame(arr, columns=new_columns)), axis=1)

def ExpandStrArrayColumns(data: pd.DataFrame, groupby: str, ignored_columns: list=[]):
    if not groupby in ignored_columns:
        ignored_columns.append(groupby)
    expanded_subsets = []
    for subset in iter(data.groupby("epoch")):
        res = dict()
        for j, col in enumerate(subset[1].columns):
            cat = []
            for i in range(len(subset[1].index)):
                if col in ignored_columns:
                    cat = subset[1].iloc[i, j]
                else:
                    cat.extend(ast.literal_eval(subset[1].iloc[i, j]))
            res[col] = cat
        expanded_subsets.append(pd.DataFrame(res))
    return pd.concat(expanded_subsets, axis=0, ignore_index=True)
