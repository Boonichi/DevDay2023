import logging
import time
import os
from pathlib import Path

from datetime import timedelta
import pandas as pd
from datetime import datetime as dt
from pandas import Timestamp
import torch
#from clean_data import csv_process, xlsx_process

def fcall(fun):
    """
    Convenience decorator used to measure the time spent while executing
    the decorated function.
    :param fun:
    :return:
    """
    def wrapper(*args, **kwargs):

        logging.info("[{}] ...".format(fun.__name__))

        start_time = time.perf_counter()
        res = fun(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time

        logging.info("[{}] Done! {}s\n".format(fun.__name__, timedelta(seconds=runtime)))
        return res

    return wrapper

def time_idx(dates, earliest_time = Timestamp("2022-03-01 00:00:00")):
    """
        Compute half hours from start.
        
        Args:
            dates (list[pandas.Timestamp]) : list of dates
            earliest_time (pd.Timestamp) : earliest time 

        Return:
            list: List of half hours from start with each date
    """
    result = []
    for date in dates:
        half_hours = (date - earliest_time).total_seconds() / 60 / 30 
        result.append(int(half_hours))

    return result

def rmr_score(actual, predict):
    """データフレームから残差平均割合を計算する。

    Args:
        actual (np.array): 実績
        predict (np.array): 予測

    Returns:
        float: 残差平均割合
    """
    eps = 1e-9
    actual = actual + eps
    diff = actual - predict
    mx = sum(abs(diff)) / sum(actual)
    return mx * 100

def load_additional_data(dataset, path):
    '''
        Load more data with given path of additional dataset

        Args:
            dataset (pandas.DataFrame): Original dataset
            path (str): Path of additional dataset

        Returns:
            pd.DataFrame: Dataset with additional datas
    '''
    add_dataset = pd.read_csv(path)
    add_dataset = pd.concat()
    return add_dataset
    
def read_external_dataset(args, target_path):
    path = Path(path)
    ex_dataset = list()
    files = list(set([file_date.split("_")[-1] for file_date in os.listdir(path)]))
    ex_dates = [date.split(".")[0] for date in files]
    
    '''cloud_dir = list(target_path.glob(f"cloud*{date}.csv"))[0]
    solar_dir = list(target_path.glob(f"solar*{ex_date}.csv"))[0]
    weather_dir = list(path.glob(f"weather*{ex_date}.csv"))[0]
    e_generator_dir = list(path.glob)'''

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)