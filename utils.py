import logging
import time
import os
from pathlib import Path

from datetime import timedelta
import pandas as pd
from datetime import datetime as dt
from pandas import Timestamp
import torch
from clean_data import create_csv, create_xlsx

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


def modify_feature(dataset, is_datetime = False):
    if is_datetime:
        dataset["year"] = dataset["date"].dt.year
        dataset["month"] = dataset["date"].dt.month
        dataset["day"] = dataset["date"].dt.day
        dataset["time"] = dataset["date"].dt.time
        dataset["time"] = dataset["time"].apply(lambda x: str(x)[:-3])
    else:
        dataset[["year", "month", "day"]] = dataset["date"].str.split("-", expand = True)
        dataset[["year", "month"]] = dataset[["year", "month"]].astype(int)
        
        dataset["day"] = dataset["day"].apply(lambda x: int(x[:2]))

        dataset["date"] = pd.to_datetime(dataset["date"])

    dataset["half_hours_from_start"] = pd.DataFrame(time_idx(dataset["date"]))

    dataset["weekday"] = dataset["date"].dt.dayofweek
    
    dataset["group"] = int(0)

    return dataset

def read_external_dataset(args, ex_input, target_date, target_half_hours):
    ex_input = Path(ex_input)
    features = [
        "power_solar",
        "power_surplus",
        "cloud",
        "weather",
        "solar"
    ]
    ex_dir = {}
    for key in features:
        if key.startswith("power"):
            feature = key.split("_")[1]
            ex_dir[key] = ex_input.glob(f"*{feature}*{target_date}.xlsx")
        else:
            ex_dir[key] = ex_input.glob(f"*{key}*{target_date}.csv")
    try:
        csv_data = create_csv(ex_dir["cloud"], ex_dir["solar"], ex_dir["weather"])
    except:
        return None

    xlsx_data = create_xlsx(ex_dir["power_solar"], ex_dir["power_surplus"])

    ex_input = pd.merge(xlsx_data, csv_data, how = "outer", on = ["date", "time"])

    # Extend ex input to target date
    
    ex_input = modify_feature(ex_input, is_datetime=False)
    
    ex_input = prepare_dataset(args, ex_input)
    
    ex_input = ex_input[lambda x: x["half_hours_from_start"] > last_half_hours]
    ex_input = ex_input[lambda x: x["half_hours_from_start"] <= target_half_hours]

    return ex_input

def create_target_dataset(input, is_datetime = True):
    if is_datetime == False:
        input["date"] = pd.to_datetime(input["date"])
    target_input = input[lambda x: x["half_hours_from_start"] == x["half_hours_from_start"].max()]

    target_input = pd.concat(
        [target_input.assign(date=lambda x: x.date + pd.offsets.DateOffset(minutes=30 * i)) for i in range(1, 48 * 2 + 1)],
        ignore_index=True, 
    )

    target_input = modify_feature(target_input, is_datetime = True)
    
    target_input.to_excel("test.xlsx")

    input = pd.concat([input, target_input], ignore_index=True)

    return target_input