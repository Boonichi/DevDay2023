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
from utils import rmr_score
import utils

from pytorch_forecasting import TemporalFusionTransformer

import torch

from postprocess import postprocess

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error


lastdate = pd.Timestamp("2023-01-31 23:30:00")
last_half_hours = time_idx(pd.Series(lastdate))[0]

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

def create_target_dataset(ex_input):
    target_input = ex_input[lambda x: x["half_hours_from_start"] == x["half_hours_from_start"].max()]
    
    target_input = pd.concat(
        [target_input.assign(date=lambda x: x.date + pd.offsets.DateOffset(minutes=30 * i)) for i in range(1, 48 * 2 + 1)],
        ignore_index=True, 
    )

    target_input = modify_feature(target_input, is_datetime = True)
    
    target_input.to_excel("test.xlsx")

    ex_input = pd.concat([ex_input, target_input], ignore_index=True)

    return target_input

# Predict
def test(args):
    # Result
    result = {
        'power_generation_pred': dict(),
        'power_generation_actual' : dict(),
        'power_demand_pred' : dict(),
        'power_demand_actual' : dict()
    }
    pred_dir = f"./data/2023_devday_data/{args.station}/eval_y/"
    ex_input = f"./data/2023_devday_data/{args.station}/eval_input_ex/"

    # Intialize device
    device = torch.device(args.device)

    #Fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = pd.read_csv(args.data_output_dir + "/{}.csv".format(args.station), index_col = 0, dtype={args.target: np.float64})

    temp_data = dataset[lambda x: x["half_hours_from_start"] > (last_half_hours - (args.max_encoder_day + 3) * 48 * 2 )]

    for fea in ["demand", "solar"]:
        if fea == "demand":
            args.target = "power_demand"
        else: args.target = "power_generation"
        
        # Model
        model = TemporalFusionTransformer.load_from_checkpoint("model_logs/{}_{}_{}/lightning_logs/version_0/checkpoints/last.ckpt".format(args.station, args.model, args.target))
        for target_date in tqdm(os.listdir(pred_dir)):
            if target_date.startswith(fea):
                target_date = target_date.split("_")[1]
                target_date = target_date.split(".")[0]
                target_date = pd.Timestamp(target_date + " 23:30:00")
                print(target_date)

                target_half_hours = time_idx(pd.Series(target_date))[0]
                # Encoder Dataset
                encoder_data = temp_data[lambda x: x["half_hours_from_start"] <= target_half_hours - 48 * args.max_pred_day]
                encoder_data = encoder_data[lambda x: x["half_hours_from_start"] > (target_half_hours - (args.max_encoder_day + args.max_pred_day - 1) * 48)]
                print(encoder_data)
                # External Dataset (External + Target)
                ex_dataset = read_external_dataset(args, ex_input, target_date.date(), target_half_hours)
                if type(ex_dataset) == "NoneType":
                    target_dataset = create_target_dataset(encoder_data)
                    ex_dataset = target_dataset
                else:  
                    target_dataset = create_target_dataset(ex_dataset) 
                    ex_dataset = pd.concat([ex_dataset, target_dataset], ignore_index=True)

                pred_dataset = pd.concat([encoder_data, ex_dataset], ignore_index = True)

                pred_dataset.to_excel("test.xlsx")

                pred, x = model.predict(pred_dataset, mode = "raw", return_x = True)
                pred = postprocess(result, args.target, pred["prediction"], target_date)

    print(result)
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