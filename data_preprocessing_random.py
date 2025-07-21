import os
import re
import numpy as np
import pandas as pd

# 中文數字轉阿拉伯數字對照表
chinese_num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6}
leak_percent_map = {'1圈': 2.78, '10圈': 27.78, '20圈': 55.56, '全開': 100.0}
root_folder = "data"

window_sec = 11
overlap = 0.5
sampling_rate = 45
window_size = int(window_sec * sampling_rate)
step_size = int(window_size * (1 - overlap))

sensors = ['pressure1', 'pressure2', 'pressure3', 'pressure4', 'pressure5', 'pressure6']

def extract_sensor_indices(filename: str):
    filename = filename.replace('.xlsx', '')
    matches = re.findall(r'[一二三四五六]', filename)
    indices = [chinese_num_map[char] for char in matches]
    return indices

def extract_features(window: pd.DataFrame, sensors: list):
    features = {}
    for sensor in sensors:
        series = window[sensor].replace(-1, np.nan).interpolate().bfill().ffill()
        values = series.values
        # 基本統計
        features[f'{sensor}_max'] = np.max(values)
        features[f'{sensor}_min'] = np.min(values)
        features[f'{sensor}_mean'] = np.mean(values)
        features[f'{sensor}_std'] = np.std(values)
        # 上升斜率與下降斜率（以中間點為分割）
        midpoint = len(values) // 2
        slope_up = (values[midpoint] - values[0]) / midpoint
        slope_down = (values[-1] - values[midpoint]) / midpoint
        features[f'{sensor}_slope_up'] = slope_up
        features[f'{sensor}_slope_down'] = slope_down
        # 保壓時間（穩定在某一水平）
        stable_threshold = 3  # 壓力變化的容許範圍
        stable_mask = np.abs(np.diff(values)) < stable_threshold
        features[f'{sensor}_hold_time'] = np.sum(stable_mask)
        # 壓力穩定程度（標準差反映）
        features[f'{sensor}_stability'] = 1 / (np.std(values) + 1e-6)
    return features

# 結果列表
all_features = []

for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".xlsx"):
            file_path = os.path.join(root, file)
            try:
                df = pd.read_excel(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)

                # === 處理 label ===
                if file == "正常.xlsx" or "正常" in file_path:
                    label = [0.0] * 6
                else:
                    leak_dir = os.path.basename(os.path.dirname(file_path))
                    leak_percent = leak_percent_map.get(leak_dir, 0.0)
                    leak_indices = extract_sensor_indices(file)
                    label = [0.0] * 6
                    for idx in leak_indices:
                        label[idx - 1] = leak_percent

                for start in range(0, len(df) - window_size + 1, step_size):
                    window = df.iloc[start:start + window_size]
                    feats = extract_features(window, sensors)
                    for i in range(6):
                        feats[f'label_sensor{i+1}'] = label[i]
                    all_features.append(feats)

            except Exception as e:
                print(f"錯誤處理 {file_path}: {e}")

# 儲存為 CSV
features_df = pd.DataFrame(all_features)
csv_path = "extracted_sensor_features.csv"
features_df.to_csv(csv_path, index=False)
