import os
import pandas as pd
import numpy as np
from pathlib import Path
from imputation import mean_most_impute, VAE_impute, IDW_impute

from clean_data import clean_dataset

from utils import fcall

class SolarProcess():
    def __init__(self, args):
        self.args = args

    def remove_outlier(self,dataset):
        def outlier(row):
            if row > 200000 or row < 0:
                result = 0
            else:
                result = row
            return result
        columns = ["power_demand", "power_surplus", "power_generation"]
        for col in columns:
            dataset[col] = dataset[col].apply(lambda row: outlier(row))
        
        return dataset

    def impute_missing_data(self,dataset):
        def impute_identify(imputation):
            funcs = {
                "mean_most_impute" : mean_most_impute,
                "VAE_impute" : VAE_impute,
                "IDW_impute" : IDW_impute
            }
            return funcs[imputation]
        
        impute_method = impute_identify(self.args.imputation)
        return dataset

    def fill_na(self,dataset):
        if self.args.fill_na == "remove":
            return dataset.dropna().reset_index(drop = True)
        elif self.args.fill_na == "fill_zero":
            return dataset.fillna(value = 0)

    #    return dataset
    def parse(self,dataset):
        funcs = {
            self.remove_outlier,
            self.fill_na,
            self.impute_missing_data,
        }
        
        for fun in funcs:
            dataset = fun(dataset)
        
        return dataset

@fcall
def prepare_dataset(args):
    dataset = clean_dataset(args)
    Process= SolarProcess(args)

    prepared_data = Process.parse(dataset)
    prepared_data = prepared_data.sort_values(by = "date")

    prepared_data.to_csv(args.data_output_dir)

    