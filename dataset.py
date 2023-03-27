import os

import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, MultiNormalizer, GroupNormalizer, TorchNormalizer
from sklearn.preprocessing import StandardScaler

def create_dataloader(args):
    
    # Read Dataset
    data = pd.read_csv(args.data_output_dir, index_col = 0)
    
    # Time interval in validation set
    max_prediction_length = args.max_pred_day * 48
    
    # Encoder Len
    max_encoder_len = args.max_encoder_day * 48

    # Time Interval in train set
    training_cutoff = data["half_hours_from_start"].max() - max_prediction_length

    # Intialize features and params for DataLoader
    #target = solar_panels + ["power_usage","power_surplus"]
    target = ["power_generation", "power_demand"]
    time_idx = "half_hours_from_start"
    group_ids = ["group"]
        
    # Train TimeSeriesDataset
    training = TimeSeriesDataSet(data = data[lambda x: x["half_hours_from_start"] <= training_cutoff], 
                                time_idx = time_idx, 
                                target = target, 
                                group_ids = group_ids,
                                min_encoder_length=max_encoder_len//2,
                                max_encoder_length=max_encoder_len,
                                min_prediction_length=1,
                                max_prediction_length=max_prediction_length,
                                time_varying_known_reals=["month", "weekday", "day"],
                                time_varying_known_categoricals=["time"],
                                time_varying_unknown_reals=["solar", "cloud", "telop_code"],
                                target_normalizer=  MultiNormalizer([
                                    GroupNormalizer(),
                                    GroupNormalizer()
                                ]),
                                add_relative_time_idx=True,
                                add_target_scales=True,
                                add_encoder_length=True,
                                allow_missing_timesteps=True,
                                #categorical_encoders={
                                #    "telop_code" : NaNLabelEncoder(add_nan=True)
                                #},
                                #scalers={
                                #    "solar" : StandardScaler(),
                                #    "cloud" : StandardScaler()
                                #}
                                )
    
    # Val TimeSeriesDataset
    val = TimeSeriesDataSet.from_dataset(training, data, predict= True, stop_randomization= True)
    
    # DataLoader
    batch_size = args.batch_size
    num_workers = args.num_workers

    train_dataloader = training.to_dataloader(train = True, batch_size = batch_size, num_workers = num_workers)
    
    val_dataloader = val.to_dataloader(train = False, batch_size=batch_size, num_workers = num_workers)

    return training, val, train_dataloader, val_dataloader
