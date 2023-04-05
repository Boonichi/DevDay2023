import os
import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs import get_args_parser
from clean_data import time_idx
from dataset import create_dataloader

from model import SolarModel

import torch

from postprocess import postprocess

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

# Predict
def test(args):
    # Result
    result = {
        'power_generation_pred': [],
        'power_generation_actual' : [],
        'power_demand_pred' : [],
        'power_demand_actual' : []
    }
    # Intialize device
    device = torch.device(args.device)

    #Fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)

    # DataLoader (Train / Val)
    trainset, valset, train_dataloader, val_dataloader  = create_dataloader(args)

    # Encoder - Decoder data (Predict Data)
    dataset = pd.read_csv(args.data_output_dir + "/{}.csv".format(args.station), index_col = 0, dtype={args.target: np.float64})
    
    dataset["date"] = pd.to_datetime(dataset["date"])

    encoder_data = dataset[lambda x: x["half_hours_from_start"] > x["half_hours_from_start"].max() - args.max_encoder_day * 48]
    
    last_data = dataset[lambda x: x["half_hours_from_start"] == x["half_hours_from_start"].max()]
    last_half_hours = list(last_data["half_hours_from_start"])[0]
    decoder_data = pd.concat(
        [last_data.assign(date=lambda x: x.date + pd.offsets.Minute(30 * i)) for i in range(1, (args.max_pred_day * 48) + 1)],
        ignore_index=True,
    )
    
    decoder_data["day"] = decoder_data["date"].dt.day
    decoder_data["month"] = decoder_data["date"].dt.month
    decoder_data["weekday"] = decoder_data["date"].dt.dayofweek
    decoder_data["time"] = decoder_data["date"].dt.time

    for index in range(len(decoder_data["time"])):
        time = str(decoder_data["time"][index]).split(":")
        decoder_data["time"][index] = time[0] + ":" + time[1]

    decoder_data["half_hours_from_start"] = time_idx(decoder_data["date"])
    
    decoder_data.to_excel("test.xlsx")

    pred_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
    
    # Model
    model = SolarModel(args).create(trainset)
    #model.to(device)
    for target in ["power_generation", "power_demand"]:
        if args.model == "base":
            preds = model.predict(pred_data)

        else:
            
            path = args.output_dir + "model_logs/{}_{}_{}".format(args.station, args.model, target) + "/lightning_logs/"
            newest_version = max([os.path.join(path,d) for d in os.listdir(path) if d.startswith("version")], key=os.path.getmtime) + "/checkpoints/"
            print(newest_version)
            checkpoint = newest_version + os.listdir(newest_version)[0]

            model = model.load_from_checkpoint(checkpoint)
            

            preds, actual = model.predict(pred_data, mode = "raw", return_x = True)

            fig = model.plot_prediction(actual, preds, idx=0, add_loss_to_title = True)
            fig.savefig("./result/{}.jpg".format(target))

            result = postprocess(result, target, preds["prediction"],pred_data[lambda x: x["half_hours_from_start"] > last_half_hours]["date"].dt.date)

            # Print metric that model get
            print(target)
            print(mean_absolute_percentage_error(result[target + "_pred"], result[target + "_actual"]))
            print(mean_absolute_error(result[target + "_pred"], result[target + "_actual"]))
    
    # Save result file as Excel format
    result = pd.DataFrame(result).to_excel("./result/result.xlsx")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Solar Model for forecasting task', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    test(args)