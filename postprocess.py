import pandas as pd
import numpy as np
import os
import math
import pickle

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from utils import rmr_score
import matplotlib.pyplot as plt

def postprocess(result, target,preds, pred_date, station):
    pred_date = str(pred_date.date())
    result[target + "_pred"][pred_date] = []
    result[target + "_actual"][pred_date] = []     

    preds = preds.tolist()

    for pred in preds[0]:
        result[target + "_pred"][pred_date].append(pred[1])
    
    result[target + "_pred"][pred_date] = result[target + "_pred"][pred_date][-48:]

    if target == "power_generation":
        target_dir = "solar"
    else: target_dir = "demand"

    
    actual_dir = "./data/2023_devday_data/{}/eval_y/{}_{}.csv".format(station,target_dir, pred_date)
    actual_value = pd.read_csv(actual_dir, header = None)

    for actual in actual_value.values:
        result[target + "_actual"][pred_date].append(actual[0])
    
    return result

def compute_metric(result):
    #with open("result.pickle", "rb") as f:
    #    result = pickle.load(f)
    for fea in ["demand", "generation"]:
        preds = []
        actuals = []
        for date in result["power_demand_pred"]:
            for value in result["power_{}_pred".format(fea)][date]:
                preds.append(value)
            for value in result["power_{}_actual".format(fea)][date]:
                actuals.append(value)
                
        preds = np.asarray(preds)
        actuals = np.asarray(actuals)
        print(fea)
        print(mean_absolute_error(preds, actuals))
        print(mean_absolute_percentage_error(preds, actuals))
        print(rmr_score(preds, actuals))

        res = dict()
        res["pred"] = preds
        res["actual"] = actuals
        res = pd.DataFrame(res)
        res.to_excel("./result/{}_result.xlsx".format(fea))

        plt.figure(figsize=(12,5))
        
        ax1 = res["pred"][:200].plot(color='blue', grid=True, label='pred')
        ax2 = res["actual"][:200].plot(color='red', grid=True, secondary_y=True, label='actual')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()


        plt.legend(h1+h2, l1+l2, loc=2)
        plt.show()
