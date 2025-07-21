import os
import re
import pandas as pd
import numpy as np

# 中文數字轉阿拉伯數字對照表
chinese_num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6}

# 洩漏圈數對應百分比
leak_percent_map = {
    '1圈': 2.78,
    '10圈': 27.78,
    '20圈': 55.56,
    '全開': 100.0
}

# 資料夾路徑
root_folder = "data"

# 切片參數
window_sec = 11
overlap = 0.5
sampling_rate = 45  # 假設為固定 45Hz
window_size = int(window_sec * sampling_rate)
step_size = int(window_size * (1 - overlap))

sensors = ['pressure1', 'pressure2', 'pressure3', 'pressure4', 'pressure5', 'pressure6']

# 結果列表
all_records = []
X_data = []
y_data = []

def extract_sensor_indices(filename: str):
    filename = filename.replace('.xlsx', '')
    matches = re.findall(r'[一二三四五六]', filename)
    indices = [chinese_num_map[char] for char in matches]
    return indices

# 遍歷所有檔案
for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".xlsx"):
            file_path = os.path.join(root, file)
            print("🔍 處理中:", file_path)

            try:
                df = pd.read_excel(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                
                median_interval = df['timestamp'].diff().dt.total_seconds().median()
                actual_sampling_rate = 1 / median_interval
                print(f"⚙️ 檔案: {file} 實際頻率：{actual_sampling_rate:.2f} Hz")


                # === 處理 label ===
                if file == "正常.xlsx" or "正常" in file_path:
                    label = [0.0] * 6  # 全正常無洩漏
                else:
                    leak_dir = os.path.basename(os.path.dirname(file_path))
                    leak_percent = leak_percent_map.get(leak_dir, 0.0)
                    leak_indices = extract_sensor_indices(file)
                    label = [0.0] * 6
                    for idx in leak_indices:
                        label[idx - 1] = leak_percent

                # === 切時間視窗 ===
                for start in range(0, len(df) - window_size + 1, step_size):
                    window = df.iloc[start:start + window_size]
                    sequence = []
                    for sensor in sensors:
                        values = window[sensor].replace(-1, np.nan).values
                        sequence.append(values)

                    sequence = np.array(sequence).T  # shape: [timesteps, 6]

                    # === 插值補 NaN → 整數（僅限時間序列部分）===
                    for i in range(sequence.shape[1]):
                        col_series = pd.Series(sequence[:, i])
                        interpolated = col_series.interpolate().bfill().ffill().round()
                        sequence[:, i] = interpolated.values
                    
                    sequence_flat = sequence.flatten()
                    all_records.append(np.concatenate([sequence_flat, label]))

                    X_data.append(sequence.astype(int))
                    y_data.append(label)  # 保留浮點數 label

            except Exception as e:
                print(f"❌ 錯誤讀取 {file_path}: {e}")

# # 儲存為 CSV
output_columns = [f't{i}_sensor{j+1}' for i in range(window_size) for j in range(6)] + \
                 [f'label_sensor{i+1}' for i in range(6)]

# output_df = pd.DataFrame(all_records, columns=output_columns)
# output_df.to_csv("all_leakage_LSTM_dataset.csv", index=False)
# print("✅ 所有資料已完成處理，儲存為 all_leakage_LSTM_dataset.csv")

X_data = np.array(X_data)  # shape: [samples, timesteps, 6]
y_data = np.array(y_data)  # shape: [samples, 6]

output_npz_path = "train_dataset.npz"
np.savez_compressed(output_npz_path, X=X_data, y=y_data)