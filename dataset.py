import os

import numpy as np
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, MultiNormalizer, GroupNormalizer, TorchNormalizer
from sklearn.preprocessing import StandardScaler


def create_dataloader(args, train_dataset, val_dataset):
    # DataLoader
    batch_size = args.batch_size
    num_workers = args.num_workers

    train_dataloader = train_dataset.to_dataloader(train = True, batch_size = batch_size, num_workers = num_workers)
    
    val_dataloader = val_dataset.to_dataloader(train = False, batch_size=batch_size, num_workers = num_workers)
    return train_dataloader, val_dataloader

def create_dataset(args):
    
    # Read Dataset
    data = pd.read_csv(args.data_output_dir + "/{}.csv".format(args.station), index_col = 0, dtype = {'power_generation' : np.float64, 'power_demand': np.float64})
    # Time interval in validation set
    max_pred_len = args.max_pred_day * 48
    
    # Encoder Len
    max_encoder_len = args.max_encoder_day * 48

    # Time Interval in train set
    training_cutoff = data["half_hours_from_start"].max() - args.horizon * 48

    # Intialize features and params for DataLoader based on target mode
    time_idx = "half_hours_from_start"
    group_ids = ["group"]
    center = False if args.model == "DeepAR" else True
    min_pred_len = max_pred_len if args.model == "NHIST" or args.model == "NBEAT" else 1
    relative_time = False if args.model == "NHIST" or args.model == "NBEAT" else True
        
    if args.target_mode == "multiple":
        target = ["power_generation", "power_demand"]
        norms = MultiNormalizer([
                                    GroupNormalizer(groups=group_ids, transformation="softplus", center = center),
                                    GroupNormalizer(groups=group_ids, transformation="softplus", center = center)
                                ])
    else:
        target = args.target
        norms = GroupNormalizer(groups=group_ids, transformation="softplus", center = center)
    
    if args.model == "TFT":
        unknown_reals = ["solar", "cloud", "power_surplus", args.target]
        unknown_categoricals = ["telop_name"]
    
    else:
        unknown_reals = [args.target]
        unknown_categoricals = []

        
    # Train TimeSeriesDataset
    train_dataset= TimeSeriesDataSet(data = data[lambda x: x["half_hours_from_start"] <= training_cutoff], 
                                time_idx = time_idx, 
                                target = target, 
                                group_ids = group_ids,
                                max_encoder_length=max_encoder_len,
                                min_prediction_length=min_pred_len,
                                max_prediction_length=max_pred_len,
                                time_varying_known_reals= ["month", "weekday", "day"],
                                time_varying_known_categoricals= ["time"],
                                time_varying_unknown_reals= unknown_reals,
                                time_varying_unknown_categoricals=unknown_categoricals,
                                target_normalizer= norms,
                                add_relative_time_idx= relative_time,
                                add_target_scales=True,
                                add_encoder_length=True,
                                allow_missing_timesteps=True,
                                scalers={
                                    "solar" : StandardScaler(),
                                    "cloud" : StandardScaler(),
                                    "power_surplus" : StandardScaler(),
                                },
                                categorical_encoders = {
                                    "telop_name": NaNLabelEncoder(add_nan=True)
                                }
                                )
    
    # Val TimeSeriesDataset
    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, data, predict= True, stop_randomization= True, min_prediction_idx = training_cutoff + 1)

    return train_dataset, val_dataset
