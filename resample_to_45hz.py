
import pandas as pd

def resample_to_45hz_and_save(input_path: str, output_path: str):
    df = pd.read_excel(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # 目標頻率為 45Hz → 約 22.222ms
    target_freq = '22.222ms'

    # 重採樣並插值
    df_interpolated = df.resample(target_freq).interpolate(method='linear')

    # 補齊邊界 NaN 並轉整數
    df_filled = df_interpolated.ffill().bfill().round().astype(int)

    # 儲存為 CSV
    df_filled.to_csv(output_path)

# 範例用法（可根據實際檔案修改）
if __name__ == "__main__":
    input_file = "data/正常.xlsx"
    output_file = "data/正常_重採樣45Hz.csv"
    resample_to_45hz_and_save(input_file, output_file)
