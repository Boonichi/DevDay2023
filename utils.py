import logging
import time
import os
from pathlib import Path

from datetime import timedelta
import pandas as pd
from datetime import datetime as dt
from pandas import Timestamp
import numpy as np
import torch
#from clean_data import create_csv, create_xlsx

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
    mx = sum(abs(diff) / sum(actual))
    return mx * 100

