import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")

# Own Package
from pytorch_forecasting import TemporalFusionTransformer, RecurrentNetwork
from clean_data import create_csv, create_xlsx
from utils import time_idx
from prepare_data import prepare_dataset
from configs import get_args_parser

# for real evaluation
start_date = datetime.date(2023, 3, 1)
end_date = datetime.date(2023, 3, 31)

# for test evaluation
#start_date = datetime.date(2023, 3, 1)
#end_date = datetime.date(2023, 3, 4)

dir_path = Path('../data/2023_devday_data')
input_dir_name = 'eval_input'
input_dir_ex_name = 'eval_input_ex'
eval_dir_y_name = 'eval_y'

# Own Intialization
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

def read_external_dataset(args, cloud_dir, solar_dir, weather_dir, power_solar_dir, power_surplus_dir, target_date, target_half_hours):
    try:
        csv_data= create_csv(cloud_dir, solar_dir, weather_dir)
    except:
        return None
    xlsx_data = create_xlsx(power_solar_dir, power_surplus_dir)
    ex_input = pd.merge(xlsx_data, csv_data, how = "outer", on = ["date", "time"])
    # Extend ex input to target date

    ex_input = modify_feature(ex_input, is_datetime=False)

    ex_input = prepare_dataset(args, ex_input)
    return ex_input

def create_target_dataset(input, is_datetime = True):
    if is_datetime == False:
        input["date"] = pd.to_datetime(input["date"])
    target_input = input[lambda x: x["half_hours_from_start"] == x["half_hours_from_start"].max()]
    
    target_input = pd.concat(
        [target_input.assign(date=lambda x: x.date + pd.offsets.DateOffset(minutes=30 * i)) for i in range(1, 48 * args.max_pred_day + 1)],
        ignore_index=True, 
    )

    target_input = modify_feature(target_input, is_datetime = True)

    input = pd.concat([input, target_input], ignore_index=True)

    return target_input

def get_excel_input(input_dir_path, input_dir_ex_path, category, date_str):
    # input
    input_files = list(input_dir_path.glob(f"*{category}*.xlsx"))
    #print(input_files)
    if category == 'battery':
        assert 1 >= len(input_files) >= 0
    else:
        assert 2 >= len(input_files) >= 1
    # extra input
    ex_input_files = list(input_dir_ex_path.glob(f"*{category}*_{date_str}.xlsx"))
    assert 1 >= len(ex_input_files) >= 0
    return input_files, ex_input_files

def get_csv_input(input_dir_path, input_dir_ex_path, category, date_str):
    # input
    input_files = list(input_dir_path.glob(f"*{category}*.csv"))
    assert len(input_files) == 1
    # extra input
    ex_input_files = list(input_dir_ex_path.glob(f"*{category}*_{date_str}.csv"))
    assert 1 >= len(ex_input_files) >= 0
    return input_files, ex_input_files


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


def predict(
    args, dataset, temp_data,
    location_path, target_date,
    solar_input_files, solar_ex_input_files, surplus_input_files, surplus_ex_input_files,
    battery_input_files, battery_ex_input_files, solar_csv_input_files, solar_csv_ex_input_files,
    cloud_csv_input_files, cloud_csv_ex_input_files, weather_csv_input_files, weather_csv_ex_input_files
):
    """
    plaese implement your prediction method here
    :return: 48 * power_generation and power_demand
    """
    station = str(location_path)[-2:]
    model_name = args.model
    if target_date[-2:] == "01": reduce_half_hours = True
    target_date = pd.Timestamp(target_date + " 23:30:00")
    ckpt_name = "final"
    
    result = dict()
    for target in ["power_demand", "power_generation"]:
        checkpoint_dir = "model_ckpt/{}_{}_{}/{}.ckpt".format(station, model_name, target, ckpt_name)
        if model_name == "TFT":
            model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_dir)
        else:
            model = RecurrentNetwork.load_from_checkpoint(checkpoint_dir)
                
        target_half_hours = time_idx(pd.Series(target_date))[0]
        # Encoder Dataset
        encoder_data = temp_data[lambda x: x["half_hours_from_start"] <= target_half_hours - args.max_pred_day * 48]
        encoder_data = encoder_data[lambda x: x["half_hours_from_start"] > (target_half_hours - (args.max_encoder_day + args.max_pred_day + 3) * 48)]

        # External Dataset (External + Target)
        ex_dataset = read_external_dataset(args, 
                                           cloud_dir = cloud_csv_ex_input_files,
                                           solar_dir = solar_csv_ex_input_files,
                                           weather_dir = weather_csv_ex_input_files,
                                           power_solar_dir = solar_ex_input_files,
                                           power_surplus_dir = surplus_ex_input_files,
                                           target_date = target_date.date(), 
                                           target_half_hours = target_half_hours)
        if type(ex_dataset).__name__ == "NoneType":
            target_dataset = create_target_dataset(encoder_data, is_datetime=False)
            ex_dataset = target_dataset
        else:  
            target_dataset = create_target_dataset(ex_dataset) 
            ex_dataset = pd.concat([ex_dataset, target_dataset], ignore_index=True)
        
        if encoder_data.empty:
            pred_dataset = ex_dataset
        else:  
            pred_dataset = pd.concat([encoder_data, ex_dataset], ignore_index = True)

    
        pred_dataset = pred_dataset[lambda x: x["half_hours_from_start"] > (target_half_hours - (args.max_encoder_day + args.max_pred_day) * 48)]

        preds, x = model.predict(pred_dataset, mode = "raw", return_x = True)

        preds = preds["prediction"]
        preds = preds.tolist()

        result[target] = []
        if model_name == "TFT": result_index = 1
        else: result_index = 0

        for pred in preds[0]:
            result[target].append(pred[result_index])
    
        result[target]= result[target][-48:]

    power_generation = result["power_generation"]
    power_demand = result["power_demand"]

    return power_generation, power_demand


def evaluate(args):
    # evaluate for each location
    metric_generation_list = []
    metric_demand_list = []

    for location_path in tqdm(list(dir_path.glob('*'))):

        input_dir_path = location_path / input_dir_name
        input_dir_ex_path = location_path / input_dir_ex_name
        eval_dir_y_path = location_path / eval_dir_y_name

        station = str(location_path)[-2:]

        dataset = pd.read_csv("dataset/{}.csv".format(station), index_col = 0)

        temp_data = dataset[lambda x: x["half_hours_from_start"] > (last_half_hours - (args.max_encoder_day + 3) * 48 * 2 )]

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            print(f"predicting for location: {location_path.name}, date: {date_str}")

            # prepare the input data (EXCEL)
            # solar input
            solar_input_files, solar_ex_input_files = get_excel_input(
                input_dir_path, input_dir_ex_path, 'solar', date_str
            )
            # surplus input
            surplus_input_files, surplus_ex_input_files = get_excel_input(
                input_dir_path, input_dir_ex_path, 'surplus', date_str
            )
            # battery input
            battery_input_files, battery_ex_input_files = get_excel_input(
                input_dir_path, input_dir_ex_path, 'battery', date_str
            )
            # prepare input data (CSV)
            # solar
            solar_csv_input_files, solar_csv_ex_input_files = get_csv_input(
                input_dir_path, input_dir_ex_path, 'solar', date_str
            )
            # cloud
            cloud_csv_input_files, cloud_csv_ex_input_files = get_csv_input(
                input_dir_path, input_dir_ex_path, 'cloud', date_str
            )
            # weather
            weather_csv_input_files, weather_csv_ex_input_files = get_csv_input(
                input_dir_path, input_dir_ex_path, 'weather', date_str
            )

            # your prediction method
            power_generation_pred, power_demand_pred = predict(
                args, dataset, temp_data,
                location_path, date_str,
                solar_input_files, solar_ex_input_files, surplus_input_files, surplus_ex_input_files,
                battery_input_files, battery_ex_input_files, solar_csv_input_files, solar_csv_ex_input_files,
                cloud_csv_input_files, cloud_csv_ex_input_files, weather_csv_input_files, weather_csv_ex_input_files
            )

            # get truth data
            power_generation_truth_path = eval_dir_y_path / f"solar_{date_str}.csv"
            power_generation_truth_df = pd.read_csv(power_generation_truth_path, header=None).fillna(0)
            assert not power_generation_truth_df.iloc[:, 0].hasnans
            power_generation_truth = power_generation_truth_df.iloc[:, 0].values.tolist()

            power_demand_truth_path = eval_dir_y_path / f"demand_{date_str}.csv"
            power_demand_truth_df = pd.read_csv(power_demand_truth_path, header=None).fillna(0)
            assert not power_demand_truth_df.iloc[:, 0].hasnans
            power_demand_truth = power_demand_truth_df.iloc[:, 0].values.tolist()

            # get metric for 1 day
            metric_generation = rmr_score(actual=np.array(power_generation_truth), predict=np.array(power_generation_pred))
            metric_demand = rmr_score(actual=np.array(power_demand_truth), predict=np.array(power_demand_pred))

            metric_generation_list.append(metric_generation)
            metric_demand_list.append(metric_demand)

            # increment current_date
            current_date += datetime.timedelta(days=1)

    print(metric_generation_list)
    print(metric_demand_list)
    # get final metric (average for each day)
    power_generation_metric = np.average(np.array(metric_generation_list))
    power_demand_truth_metric = np.average(np.array(metric_demand_list))
    return (power_generation_metric + power_demand_truth_metric) / 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Solar Model for forecasting task', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metric = evaluate(args)

    print(f"your metrics is {metric}!")
