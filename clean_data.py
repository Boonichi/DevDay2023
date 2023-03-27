from pathlib import Path
import pandas as pd
import os
import re
from datetime import datetime as dt
from pandas import Timestamp

raw_weather_type = ['薄曇','晴れ','くもり','少雨','弱い雨','強い雨','激しい雨','猛烈な雨','みぞれ(弱い)','雪 (強い)','雪 (弱い)','みぞれ(強い)']
weather_type = ['light cloud', 'sunny', 'cloudy', 'light rain', 'light rain', 'strong rain', 'heavy rain', 'heavy rain', 'sleet (weak)', 'snow (heavy)' ,'snow (weak)','sleet (strong)']

def save_xls(dict_df, path):
        writer = pd.ExcelWriter(path)
        for key in dict_df:
            dict_df[key].to_excel(writer, key)
        writer.save()

def init_dir(path):
    cloud_dir = path.glob(f"cloud*.csv")
    solar_dir = path.glob(f"solar*.csv")
    weather_dir = path.glob(f"weather*.csv")
    e_generator_dir = path.glob(f"*solar*.xlsx")
    e_demand_dir = path.glob(f"*surplus*.xlsx")
    
    return cloud_dir, solar_dir, weather_dir, e_generator_dir, e_demand_dir

def csv_process(dataset):
    result = list()
    # Process execution date
    def target_date(sample):
        date, time = sample.split("T")
        time = time.split(":")
        time = time[0] + ":" + time[1]
        return date, time
    
    for index, data in dataset.iterrows():
        sample = dict()
        
        #Target Date
        date, time = target_date(data["target_date"])
        sample["date"] = date + " " + time + ":00" 
        sample["time"] = time
        
        #Features
        
        sample["cloud"] = data["cloud(%)"]
        sample["solar"] = data["solar(W/m2)"]
        sample["telop_code"] = data["telop_code"]
        try:
            sample["telop_name"] = weather_type[raw_weather_type.index(data["telop_name"])]
        except:
            sample["telop_name"] = data["telop_name"]
        result.append(sample)
    return result

def create_csv(cloud_dir, solar_dir, weather_dir):
    # Reconstruct Path Glob
    cloud_dir = list(cloud_dir)[0]
    solar_dir = list(solar_dir)[0]
    weather_dir = list(weather_dir)[0]

    cloud_data = pd.read_csv(cloud_dir)
    solar_data = pd.read_csv(solar_dir)
    weather_data = pd.read_csv(weather_dir)
    csv_data = pd.merge(weather_data, solar_data, how = 'outer', on = ["target_date", "execution_date"])
    csv_data = pd.merge(csv_data, cloud_data, how = 'outer', on = ["target_date", "execution_date"])
    
    # CSV Process
    csv_data = pd.DataFrame(csv_process(csv_data))

    return csv_data

def xlsx_process(path):
    path = list(path)[0]
    f = pd.ExcelFile(path)
    result = list()
    for sheet_name in f.sheet_names:
        # Storage temp xlsx file
        df = f.parse(sheet_name = sheet_name)
        df = df.drop(labels = [0,1,2,3], axis = 0)
        df = df.to_excel("temp.xlsx", header = None)
        # Read xlsx with right format
        df_new = pd.read_excel("temp.xlsx", index_col = None)        

        for index, data in df_new.iterrows():
            solar_state = 0

            sample = dict()
            sample["date"] = sheet_name + " " + data["時刻"] + ":00"
            sample["time"] = data["時刻"]
            for solar_panel in data.index:

                if (solar_panel == "30101:電力量（Wh）"):
                    sample["power_demand"] = int(data[solar_panel])
                elif (solar_panel == "30101:回生電力量（Wh）"):
                    sample["power_surplus"] = int(data[solar_panel])
                elif (len(str(solar_panel)) > 2):
                    if solar_state == 0:
                        solar_state = 1
                        sample["power_generation"] = int(data[solar_panel])
                    else:
                        sample["power_generation"]+=data[solar_panel]
        
            result.append(sample)
    
    return result

def create_xlsx(e_generator_dir, e_demand_dir):
    e_generator_data = pd.DataFrame(xlsx_process(e_generator_dir))

    e_demand_data = pd.DataFrame(xlsx_process(e_demand_dir))
    
    xlsx_data = pd.merge(e_generator_data, e_demand_data, how = "outer", on = ["date", "time"])

    return xlsx_data

def time_idx(dates):
    earliest_time = dates.min()
    result = []
    
    for date in dates:
        half_hours = (date - earliest_time).seconds / 60 / 30 + (date - earliest_time).days * 24
        result.append(int(half_hours))

    return pd.Series(result)

def clean_dataset(args):
    path = args.data_dir

    path = Path(path)

    # Read directory of input files
    cloud_dir, solar_dir, weather_dir, e_generator_dir, e_demand_dir = init_dir(path)

    # Generate csv/xlsx after preprocessing
    csv_data = create_csv(cloud_dir, solar_dir, weather_dir)
    xlsx_data = create_xlsx(e_generator_dir, e_demand_dir)

    #Combine all files into only one DataFrame
    station_data = pd.merge(xlsx_data, csv_data, how = "outer", on = ["date", "time"])

    station_data[["year", "month", "day"]] = station_data.date.str.split("-", expand = True)
    
    station_data["day"] = station_data["day"].apply(lambda x: int(x[:2]))
    
    station_data["date"] = pd.to_datetime(station_data["date"])

    station_data["half_hours_from_start"] = time_idx(station_data["date"])

    station_data["weekday"] = station_data["date"].dt.dayofweek
    
    station_data["group"] = str(0)
    
    return station_data
        


