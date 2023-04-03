import os
import pandas as pd
import numpy as np
from pathlib import Path
import math

from clean_data import clean_dataset

from utils import fcall

from tsmoothie import smoother as sm
from tsmoothie import bootstrap as bs

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

class SolarProcess():
    def __init__(self, args):
        self.args = args

    def remove_error(self,dataset):
        features = ["power_demand", "power_surplus", "power_generation"]
        max_value = 500000
        min_value = 0

        result = []

        for index, row in dataset.iterrows():
            for fea in features:
                if row[fea] > max_value or row[fea] < min_value:
                    row[fea] = math.nan
            result.append(row)
        
        return pd.DataFrame(result, columns = list(dataset.columns))

    def fill_na(self,dataset):
        Numerical_features = ["power_generation", "power_surplus", "power_demand", "cloud", "solar"]
        Categorical_features = ["telop_name"]
        if self.args.fill_na == "fill":
            dataset[Numerical_features] = dataset[Numerical_features].fillna(0)
            dataset[Categorical_features] = dataset[Categorical_features].apply(lambda x:x.fillna(x.value_counts().index[0]))

        return dataset
    
    def impute(self, dataset):
        if self.args.impute == "MICE":
            imputation = IterativeImputer(max_iter = 1000)
        elif self.args.impute == "KNN":
            imputation = KNNImputer(n_neighbors=5)
        
        # Full half hours feature
        full_half_hours = list(range(min(dataset["half_hours_from_start"]), max(dataset["half_hours_from_start"])))
        full_half_hours = pd.DataFrame(full_half_hours, columns = ["half_hours_from_start"])

        dataset = pd.merge(full_half_hours, dataset, how = "outer", on = ["half_hours_from_start"]).reset_index(drop = True)

        
        TempData = dataset[["date", "time", "telop_name"]]

        dataset = dataset.drop(columns = ["date", "telop_name", "telop_code", "time"])
        dataset = pd.DataFrame(imputation.fit_transform(dataset), columns = dataset.columns)

        return dataset.join(TempData)
    
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
                    if row[fea] > up_interval[index_series][index_row - self.args.window_size]:
                        check = False
            if check == True:
                result.append(row)

        return pd.DataFrame(result, columns = list(dataset.columns))

    def customize(self, dataset):
        # Time index should be type integer
        dataset = dataset.astype({'half_hours_from_start': 'int32'})
        return dataset
    def parse(self,dataset):
        funcs = [
            self.remove_error,
            #self.impute,
            self.fill_na,
            self.smooth,
            #self.customize,
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

    prepared_data.to_csv(args.data_output_dir + "/{}.csv".format(args.station))

    