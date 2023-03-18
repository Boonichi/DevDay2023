from pathlib import Path
import pandas as pd
import os
raw_weather_type = ['薄曇','晴れ','くもり','少雨','弱い雨','強い雨','激しい雨','猛烈な雨','みぞれ(弱い)','雪 (強い)','雪 (弱い)','みぞれ(強い)']
weather_type = ['light cloud', 'sunny', 'cloudy', 'light rain', 'light rain', 'strong rain', 'heavy rain', 'heavy rain', 'sleet (weak)', 'snow (heavy)' ,'snow (weak)','sleet (strong)']
start_date = ["20220301","20230101"]
end_date = ["20221231","20230228"]

def save_xls(dict_df, path):
        writer = pd.ExcelWriter(path)
        for key in dict_df:
            dict_df[key].to_excel(writer, key)
        writer.save()

def init_dir(path, station):
    cloud_dir = path / "cloud_{}0000{}.csv".format(station[0], station[1])
    sunny_dir = path / "solar_{}0000{}.csv".format(station[0], station[1])
    weather_dir = path /  "weather_forecast_{}0000{}.csv".format(station[0], station[1])
    e_generator_dir = path /  "realne_report_solar_30_{}0000{}_{}_{}.xlsx".format(station[0], station[1], start_date[0], end_date[1])
    e_demand_dir = list()
    if station[0] == "v":
        e_demand_dir.append(path / "realne_report_surplus30p_{}0000{}_{}_{}.xlsx".format(station[0], station[1], start_date[0], end_date[0]))
        e_demand_dir.append(path / "realne_report_surplus30p_{}0000{}_{}_{}.xlsx".format(station[0], station[1], start_date[1], end_date[1]))
    else:
        e_demand_dir.append(path / "realne_report_surplus30p_{}0000{}_{}_{}.xlsx".format(station[0], station[1], start_date[0], end_date[1]))
    
    return cloud_dir, sunny_dir, weather_dir, e_generator_dir, e_demand_dir

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
        sample["date"] = date
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

def create_csv(cloud_dir, sunny_dir, weather_dir):
    cloud_data = pd.read_csv(cloud_dir)
    sunny_data = pd.read_csv(sunny_dir)
    weather_data = pd.read_csv(weather_dir)
    csv_data = pd.merge(weather_data, sunny_data, how = 'outer', on = ["target_date", "execution_date"])
    csv_data = pd.merge(csv_data, cloud_data, how = 'outer', on = ["target_date", "execution_date"])
    
    # CSV Process
    csv_data = pd.DataFrame(csv_process(csv_data))

    return csv_data

def xlsx_process(path):
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
            sample = dict()
            sample["date"] = sheet_name
            sample["time"] = data["時刻"]
            
            for solar_panel in data.index:
                
                if (solar_panel == "30101:電力量（Wh）"):
                    sample["power_consumption"] = data[solar_panel]
                elif (solar_panel == "30101:回生電力量（Wh）"):
                    sample["power_surplus"] = data[solar_panel]
                elif (len(str(solar_panel)) > 2):
                    sample[solar_panel] = data[solar_panel]
        
            result.append(sample)
    
    return result

def create_xlsx(e_generator_dir, e_demand_dirs):
    e_generator_data = pd.DataFrame(xlsx_process(e_generator_dir))

    e_demand_data = pd.DataFrame()
    for e_demand_dir in e_demand_dirs:
        e_demand_data = pd.concat([e_demand_data, pd.DataFrame(xlsx_process(e_demand_dir))])
        
    xlsx_data = pd.merge(e_generator_data, e_demand_data, how = "outer", on = ["date", "time"])

    return xlsx_data

def clean_dataset(args):
    path = args.data_dir
    station = path.split("/")[-1]

    path = Path(path)
    cloud_dir, sunny_dir, weather_dir, e_generator_dir, e_demand_dirs = init_dir(path, station)

    csv_data = create_csv(cloud_dir, sunny_dir, weather_dir)
    xlsx_data = create_xlsx(e_generator_dir, e_demand_dirs)

    station_data = pd.merge(xlsx_data, csv_data, how = "outer", on = ["date", "time"])
    station_data[["year", "month", "day"]] = station_data.date.str.split("-", expand = True)
    station_data = station_data.drop(columns = ["date"])
    
    return station_data
        


