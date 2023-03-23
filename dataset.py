import os

import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, MultiNormalizer, GroupNormalizer

def create_dataloader(args):
    
    data = pd.read_csv(args.data_output_dir, index_col = 0)
    
    max_prediction_length = args.max_pred_len

    features_col = data.columns
    solar_panels = []
    for col in features_col:
        if (col.startswith("solar_panel")):
            solar_panels.append(col)
        
    training_cutoff = data["half_hours_from_start"].max() - max_prediction_length

    #target = solar_panels + ["power_usage","power_surplus"]
    target = "power_usage"
    time_idx = "half_hours_from_start"
    group_ids = ["time"]
    
    # Target Normalizer
    Normalizer_List = []
    #for _ in target:
    #    Normalizer_List.append(GroupNormalizer(groups = ["time"], transformation="softplus"))

    #target_normalizer = MultiNormalizer(Normalizer_List)
    target_normalizer = GroupNormalizer(groups = ["time"], transformation="softplus")
    

    training = TimeSeriesDataSet(data = data[lambda x: x["half_hours_from_start"] <= training_cutoff], 
                                 time_idx = time_idx, 
                                 target = target, 
                                 group_ids = group_ids,
                                 min_encoder_length=0,
                                 max_encoder_length=args.max_encoder_len,
                                 min_prediction_length=1,
                                 max_prediction_length=args.max_pred_len,
                                 time_varying_known_reals=["year","month", "weekday"],
                                 time_varying_known_categoricals=["time"],
                                 target_normalizer= target_normalizer,
                                 categorical_encoders= {
                                    "telop_name" : NaNLabelEncoder(add_nan = True),
                                 },
                                 add_relative_time_idx=True,
                                 add_target_scales=True,
                                 add_encoder_length=True,
                                 allow_missing_timesteps=True
                                 )

    val = TimeSeriesDataSet.from_dataset(training, data, predict= True, stop_randomization= True)
    
    batch_size = args.batch_size
    num_workers = args.num_workers

    train_dataloader = training.to_dataloader(train = True, batch_size = batch_size, num_workers = num_workers)
    
    val_dataloader = val.to_dataloader(train = False, batch_size=batch_size, num_workers = num_workers)

    return train_dataloader, val_dataloader
