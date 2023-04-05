import pandas as pd
import numpy as np
import os
import math

def postprocess(result, target,preds, pred_date):
    preds = preds.tolist()

    for pred in preds[0]:
        result[target + "_pred"].append(round(pred[0]))

    pred_date = set(pred_date)
    if target == "power_generation":
        target_dir = "solar"
    else: target_dir = "demand"

    for date in pred_date:
        actual_dir = "./data/2023_devday_data/v1/eval_y/{}_{}.csv".format(target_dir, str(date))
        actual_value = pd.read_csv(actual_dir, header = None)

        for actual in actual_value.values:
            result[target + "_actual"].append(actual[0])
    
    return result
    
    