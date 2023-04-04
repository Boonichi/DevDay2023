import os
import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from configs import get_args_parser
from clean_data import time_idx
from dataset import create_dataloader

from model import SolarModel

import torch

# Predict
def test(args):
    # Intialize device
    device = torch.device(args.device)

    #Fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)

    # DataLoader (Train / Val)
    trainset, valset, train_dataloader, val_dataloader,  = create_dataloader(args)

    # Encoder - Decoder data (Predict Data)
    dataset = pd.read_csv(args.data_output_dir + "/{}.csv".format(args.station), index_col = 0)
    
    dataset["date"] = pd.to_datetime(dataset["date"])

    encoder_data = dataset[lambda x: x["half_hours_from_start"] > x["half_hours_from_start"].max() - args.max_encoder_day * 48]
    
    last_data = dataset[lambda x: x["half_hours_from_start"] == x["half_hours_from_start"].max()]

    decoder_data = pd.concat(
        [last_data.assign(date=lambda x: x.date + pd.offsets.Minute(30 * i)) for i in range(1, (args.max_pred_day * 48) + 1)],
        ignore_index=True,
    )
    
    decoder_data["day"] = decoder_data["date"].dt.day
    decoder_data["month"] = decoder_data["date"].dt.month
    decoder_data["weekday"] = decoder_data["date"].dt.dayofweek
    decoder_data["time"] = decoder_data["date"].dt.time[:-2]
    decoder_data["half_hours_from_start"] = time_idx(decoder_data["date"])

    print(decoder_data.head())

    pred_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
    
    # Model
    model = SolarModel(args).create(trainset)
    model.to(device)
    
    if args.model == "base":
        preds = model.predict(pred_data)

    else:
        if args.target_mode == "multiple":
            path = args.output_dir + "model_logs/{}_{}".format(args.station, args.model) + "/lightning_logs/"
        else:
            path = args.output_dir + "model_logs/{}_{}_{}".format(args.station, args.model, args.target) + "/lightning_logs/"
        newest_version = max([os.path.join(path,d) for d in os.listdir(path) if d.startswith("version")], key=os.path.getmtime) + "/checkpoints/"
        print(newest_version)
        checkpoint = newest_version + os.listdir(newest_version)[0]

        model = model.load_from_checkpoint(checkpoint)
        

        preds, actual = model.predict(pred_data, mode = "raw", return_x = True)


    with open(args.output_dir + "result/pred_{}.pickle".format(args.target),"wb") as f:
        pickle.dump(preds["prediction"],f)
    f.close()


    with open(args.output_dir + "result/actual_{}.pickle".format(args.target),"wb") as f:
        pickle.dump(actual,f)
        f.close()


        #preds, actuals = postprocess(preds)
        #compute_metric(preds,actuals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Solar Model for forecasting task', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    test(args)