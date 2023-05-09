import os
import openpyxl
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import shutil

# for test
# input_start_date = pd.to_datetime('2022-02-01').date()
# input_end_date = pd.to_datetime('2023-01-31').date()
# eval_start_date = pd.to_datetime('2023-02-01').date()
# eval_end_date = pd.to_datetime('2023-02-26').date() # 2days before from end of month

# for final
input_start_date = pd.to_datetime('2022-02-01')
input_end_date = pd.to_datetime('2023-02-28')
eval_start_date = pd.to_datetime('2023-03-01')
eval_end_date = pd.to_datetime('2023-03-29') # 2days before from end of month

if __name__ == '__main__':
    # ディレクトリのパスを指定
    dir_path = Path('../data/2023_devday_data')
    result_dir_name = 'eval_input'
    result_dir_ex_name = 'eval_input_ex'

    # 各拠点を処理
    for location_path in tqdm(list(dir_path.glob('*'))):
        result_dir_path = location_path / result_dir_name
        result_dir_path.mkdir(exist_ok=True)

        result_dir_ex_path = location_path / result_dir_ex_name
        result_dir_ex_path.mkdir(exist_ok=True)
        for content_type in ['solar', 'surplus30p', 'battery30']:
            target_files = list(location_path.glob(f"*{content_type}*.xlsx"))
            sheets_list = [pd.read_excel(x, sheet_name=None) for x in target_files]
            if len(target_files) > 1:
                sheets = {}
                for sheets_item in sheets_list:
                    sheets.update(sheets_item)

            elif len(target_files) == 1:
                sheets = pd.read_excel(target_files[0], sheet_name=None)

            else:
                continue

            # Excelファイルを開く
            file_path = location_path / target_files[0]
            file_name = file_path.name
            no_ext_file_name, ext = file_path.stem, file_path.suffix
            save_file_path = result_dir_path / file_name

            # 日付がstart_date以降のシートだけを選択して、新しいExcelファイルに保存する
            filtered_sheets = {sheet_name: sheet for sheet_name, sheet in sheets.items() if
                               input_start_date <= datetime.strptime(sheet_name, '%Y-%m-%d') <= input_end_date}
            output_file = save_file_path
            with pd.ExcelWriter(output_file) as writer:
                for sheet_name, sheet in filtered_sheets.items():
                    sheet.to_excel(writer, sheet_name=sheet_name, index=False)

            # 3日目以降の追加教師データ
            delta = timedelta(days=1)

            current_date = eval_start_date
            while current_date <= eval_end_date:
                # 対象の日付だけにする
                filtered_sheets = {sheet_name: sheet for sheet_name, sheet in sheets.items() if
                                   eval_start_date <= datetime.strptime(sheet_name, '%Y-%m-%d') <= current_date}

                two_days_later = current_date + timedelta(days=2)
                output_file = result_dir_ex_path.joinpath(
                    f"{no_ext_file_name}_{two_days_later.strftime('%Y-%m-%d')}.xlsx"
                )
                print(output_file)

                with pd.ExcelWriter(output_file) as writer:
                    for sheet_name, sheet in filtered_sheets.items():
                        sheet.to_excel(writer, sheet_name=sheet_name, index=False)

                current_date += delta