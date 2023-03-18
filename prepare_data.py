import os
import pandas as pd
import numpy as np
from pathlib import Path
from imputation import mean_most_impute, VAE_impute, IDW_impute

from clean_data import clean_dataset

from utils import fcall

def outlier(row):
    if row > 200000 or row < 0:
        result = 0
    else:
        result = row
    return result
def remove_outlier(args,dataset):
    columns = ["power_consumption", "power_surplus"]
    for col in columns:
        dataset[col] = dataset[col].apply(lambda row: outlier(row))
    
    return dataset

def impute_identify(imputation):
    funcs = {
        "mean_most_impute" : mean_most_impute,
        "VAE_impute" : VAE_impute,
        "IDW_impute" : IDW_impute
    }
    return funcs[imputation]

def impute_missing_data(args,dataset):
    
    impute_method = impute_identify(args.imputation)
    return dataset

def impute_na(args,dataset):
    if args.impute_na == "remove":
        return dataset.dropna().reset_index(drop = True)

def preprocess(args,dataset):
    funcs = [
        remove_outlier,
        impute_na,
        impute_missing_data,
    ]
    for fun in funcs:
        dataset = fun(args,dataset)

    return dataset

@fcall
def prepare_dataset(args):
    dataset = clean_dataset(args)

    prepared_data = preprocess(args,dataset)

    prepared_data.to_csv(args.data_output_dir)
    
    split_dataset(args, dataset)