import os
import pandas as pd
import numpy as np
from pathlib import Path

output_dir = "./dataset/processed/"
data_dir = Path("./dataset/raw")

def outlier(row):
    if row > 200000 or row < 0:
        result = 0
    else:
        result = row
    return result
def remove_outlier(dataset):
    columns = ["power_consumption", "power_surplus"]
    for col in columns:
        dataset[col] = dataset[col].apply(lambda row: outlier(row))
    
    return dataset

def single_preprocess(dataset):
    funcs = [
        remove_outlier
    ]
    for fun in funcs:
        dataset = fun(dataset)

    return dataset

def preprocess(path):
    for data_dir in os.listdir(path):
        if (data_dir.endswith(".csv")):

            preprocessed_data = single_preprocess(pd.read_csv(path / data_dir, index_col = 0))
            
            preprocessed_data.to_csv(output_dir + data_dir)

preprocess(data_dir)
    