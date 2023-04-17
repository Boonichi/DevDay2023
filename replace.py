import os
import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px

from configs import get_args_parser
from clean_data import time_idx, create_csv, create_xlsx
from prepare_data import prepare_dataset
from dataset import create_dataloader, create_dataset

from model import SolarModel

import torch

from postprocess import postprocess

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from utils import rmr_score
import utils

lastdate = pd.Timestamp("2023-01-31 23:30:00")
last_half_hours = time_idx(pd.Series(lastdate))[0]

def read_external_dataset(args, ex_input, target_date):
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

    csv_data = create_csv(ex_dir["cloud"], ex_dir["solar"], ex_dir["weather"])
    xlsx_data = create_xlsx(ex_dir["power_solar"], ex_dir["power_surplus"])

    ex_input = pd.merge(xlsx_data, csv_data, how = "outer", on = ["date", "time"])

    ex_input[["year", "month", "day"]] = ex_input.date.str.split("-", expand = True)
    
    ex_input["day"] = ex_input["day"].apply(lambda x: int(x[:2]))
    
    ex_input["date"] = pd.to_datetime(ex_input["date"])

    ex_input["half_hours_from_start"] = pd.DataFrame(time_idx(ex_input["date"]))

    ex_input["weekday"] = ex_input["date"].dt.dayofweek
    
    ex_input["group"] = str(0)
    
    ex_input = prepare_dataset(args, ex_input)
    
    ex_input = ex_input[lambda x: x["half_hours_from_start"] > last_half_hours]
    return ex_input

# Predict
def test(args):
    # Result
    result = {
        'power_generation_pred': [],
        'power_generation_actual' : [],
        'power_demand_pred' : [],
        'power_demand_actual' : []
    }
    pred_dir = f"./data/2023_devday_data/{args.station}/eval_y/"
    ex_input = f"./data/2023_devday_data/{args.station}/eval_input_ex/"

    # Intialize device
    device = torch.device(args.device)

    #Fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    power_generation_model = 0
    power_demand_model = 0

    dataset = pd.read_csv(args.data_output_dir + "/{}.csv".format(args.station), index_col = 0, dtype={args.target: np.float64})

    temp_data = dataset[lambda x: x["half_hours_from_start"] > (last_half_hours - (args.max_encoder_day + 3) * 48 * 2 )]

    for fea in ["demand", "solar"]:
        if fea == "demand":
            args.target = "power_demand"
        else: args.target = "power_generation"

        for target_date in tqdm(os.listdir(pred_dir)):
            if target_date.startswith(fea):
                target_date = target_date.split("_")[1]
                target_date = target_date.split(".")[0]
                target_date = pd.Timestamp(target_date + " 23:30:00")
                print(target_date)

                target_half_hours = time_idx(pd.Series(target_date))[0]

                ex_dataset = read_external_dataset(args, ex_input, target_date.date())

                encoder_data = temp_data[lambda x: x["half_hours_from_start"] <= target_half_hours - 48 * args.max_pred_day]
                encoder_data = encoder_data[lambda x: x["half_hours_from_start"] > (target_half_hours - (args.max_encoder_day + args.max_pred_day) * 48)]
                print(encoder_data)
                print(ex_dataset)
                pred_dataset = pd.concat([encoder_data, ex_dataset], ignore_index = True)

                pred_dataset.to_excel("test.xlsx")

                # DataLoader (Train / Val) and Model

                train_dataset, val_dataset = create_dataset(args, pred_dataset)
                model = SolarModel(args, train_dataset)

                pred = postprocess()
                result.append(pred)

                raise NotImplementedError()
            
    # Save result file as Excel format
    result.to_excel("./result/result.xlsx")
    
    # Plot 2 time series power_pred and power_actual if arg verbose == true
    target = "power_demand"
    if args.verbose:
        plt.figure(figsize=(12,5))
        
        ax1 = result[target + "_pred"].plot(color='blue', grid=True, label='pred')
        ax2 = result[target + "_actual"].plot(color='red', grid=True, secondary_y=True, label='actual')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()


        plt.legend(h1+h2, l1+l2, loc=2)
        plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Solar Model for forecasting task', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    test(args)