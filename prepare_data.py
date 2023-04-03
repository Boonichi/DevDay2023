import os
import pandas as pd
import numpy as np
from pathlib import Path

from clean_data import clean_dataset

from utils import fcall

from tsmoothie import smoother as sm
from tsmoothie import bootstrap as bs

class SolarProcess():
    def __init__(self, args):
        self.args = args

    def remove_error(self,dataset):
        features = ["power_demand", "power_surplus", "power_generation"]
        max_value = 500000
        min_value = 0

        result = []

        for index, row in dataset.iterrows():
            check = True
            for fea in features:
                if row[fea] > max_value or row[fea] < min_value:
                    check = False
            if check == True:
                result.append(row)
        
        return pd.DataFrame(result, columns = list(dataset.columns))

    def fill_na(self,dataset):
        if self.args.fill_na == "remove":
            dataset = dataset.dropna().reset_index(drop = True)
        elif self.args.fill_na == "fill_zero":
            dataset = dataset.fillna(value = 0)    

        return dataset
    def smooth(self, dataset):
        features = ["power_generation", "power_demand"]
        if self.args.smooth == "Exponent":
            smoother = sm.ExponentialSmoother(window_len= self.args.window_size, alpha = 0.3)
        
        fea_data = []
        for fea in features:
            sample = dataset[fea].tolist()
            
            fea_data.append(sample)
        
        smoother.smooth(fea_data)
        low_interval, up_interval = smoother.get_intervals(self.args.interval + "_interval")
        
        result = []
        
        for index_row, row in dataset.iterrows():
            check = True
            for index_series, fea in enumerate(features):
                if index_row >= self.args.window_size:
                    if row[fea] > up_interval[index_series][index_row - 48]:
                        check = False
            if check == True:
                result.append(row)

        return pd.DataFrame(result, columns = list(dataset.columns))

    def parse(self,dataset):
        funcs = [
            self.remove_error,
            self.fill_na,
            self.smooth,
        ]
        
        for fun in funcs:
            dataset = fun(dataset)
        
        return dataset

@fcall
def prepare_dataset(args):
    dataset = clean_dataset(args)
    Process= SolarProcess(args)
    prepared_data = Process.parse(dataset)
    prepared_data = prepared_data.sort_values(by = "date")

    prepared_data.reset_index().to_csv(args.data_output_dir + "/{}.csv".format(args.station))

    