import json
import time
import random
import pandas as pd
import numpy as np
import joblib
import ctypes
import atexit
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
# from model import MySQLConnector, error_log, mysql_log, mqtt_log

#TODO 加入錯誤處理跟log

# === 參數設定 ===
# 資料庫設定
MYSQL_HOST = "localhost" # 資料庫主機
MYSQL_USER = "aict702" # 資料庫使用者名稱
MYSQL_PASSWORD = "aict702@Lab702" # 資料庫使用者密碼
DBNAME = "sensorTest" # 資料庫名稱
TABLE_NAME = "leakage_predictions"

LIB_PATH = "model/arduino_reader_multi.so"  # 請改成你的 .so 或 .dll 檔案路徑
sampling_rate = 45
window_sec = 11
window_size = sampling_rate * window_sec
sensors = ['pressure1', 'pressure2', 'pressure3', 'pressure4', 'pressure5', 'pressure6']

# === 載入模型與策略對應 ===
models = [joblib.load(f"model/model_sensor_{i+1}.joblib") for i in range(6)]
leak_levels = [0.0, 2.78, 27.78, 55.56, 100.0]
policy_map = {
    0: "無需維修",
    1: "觀察壓力變化，定期巡檢",
    2: "安排停機檢查與氣密測試",
    3: "儘快更換管件或電磁閥元件",
    4: "緊急停機，立即維修並追蹤"
}

# === 感測器讀取函式 ===
def read_sensor() -> Dict[str, Union[str, float]]:
    # arduino_lib = ctypes.CDLL(LIB_PATH)
    # arduino_lib.get_sensor_data.argtypes = [
    #     ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    #     ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    #     ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    # ]
    # arduino_lib.init_serial.argtypes = []
    # arduino_lib.close_sensor.argtypes = []

    # arduino_lib.init_serial()
    # atexit.register(arduino_lib.close_sensor)

    # p1 = ctypes.c_float()
    # p2 = ctypes.c_float()
    # p3 = ctypes.c_float()
    # p4 = ctypes.c_float()
    # p5 = ctypes.c_float()
    # p6 = ctypes.c_float()

    # arduino_lib.get_sensor_data(
    #     ctypes.byref(p1), ctypes.byref(p2), ctypes.byref(p3),
    #     ctypes.byref(p4), ctypes.byref(p5), ctypes.byref(p6)
    # )

    # return {
    #     "pressure1": p1.value,
    #     "pressure2": p2.value,
    #     "pressure3": p3.value,
    #     "pressure4": p4.value,
    #     "pressure5": p5.value,
    #     "pressure6": p6.value
    # }
    
    pressures = [round(random.uniform(215, 700.0), 0) for _ in range(6)]

    return {
        **{f"pressure{i+1}": pressures[i] for i in range(6)}
    }

# === 特徵抽取 ===
def extract_features(df: pd.DataFrame):
    feats = {}
    for sensor in sensors:
        series = df[sensor].replace(-1, np.nan).interpolate().bfill().ffill()
        values = series.values
        feats[f'{sensor}_max'] = np.max(values)
        feats[f'{sensor}_min'] = np.min(values)
        feats[f'{sensor}_mean'] = np.mean(values)
        feats[f'{sensor}_std'] = np.std(values)
        midpoint = len(values) // 2
        feats[f'{sensor}_slope_up'] = (values[midpoint] - values[0]) / midpoint
        feats[f'{sensor}_slope_down'] = (values[-1] - values[midpoint]) / midpoint
        stable_mask = np.abs(np.diff(values)) < 3
        feats[f'{sensor}_hold_time'] = np.sum(stable_mask)
        feats[f'{sensor}_stability'] = 1 / (np.std(values) + 1e-6)
    return pd.DataFrame([feats])

def failure_probability_prediction(df: pd.DataFrame) -> float:
    """
    計算故障率（回傳一個百分比 float 值）
    基於 pressure1/4/6 平均壓力 Px 進行推論
    """
    k = 0.015
    Pn = 600
    Px_normal_threshold = 579

    # 清理不合法資料
    df = df.dropna(subset=['pressure1', 'pressure4', 'pressure6'])
    df = df[(df['pressure1'] >= 400) & (df['pressure4'] >= 400) & (df['pressure6'] >= 400)]

    if df.empty:
        return 0.0  # 沒有有效資料

    # 計算 Px
    avg_p1 = df['pressure1'].mean()
    avg_p4 = df['pressure4'].mean()
    avg_p6 = df['pressure6'].mean()
    Px = (avg_p1 + avg_p4 + avg_p6) / 3

    # 計算故障率
    if Px >= Px_normal_threshold:
        F = 0.0
    else:
        F = 1 - np.exp(-k * (Pn - Px))
        F = max(F, 0.0)

    return round(F * 100.0, 2)    # ← 回傳與 random.uniform() 同樣格式：浮點數 %

# def sql_connection(result: Dict) -> Optional[MySQLConnector]:
#     try:
#         sql = MySQLConnector(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, DBNAME)
#         sql.insert_data(TABLE_NAME, result)

#         return sql # 回傳 sql 連線物件 給mqtt_connection使用
#     except Exception as e:
#         mysql_log.logger.error(f"Unexpected error with doing sql: {e}")
#         return None

# === 主程式 ===
print("開始讀取感測器資料中，每 11 秒分析一次...\n")
buffer = []

while True:
    sensor = read_sensor()
    sensor["timestamp"] = pd.Timestamp.now()
    buffer.append(sensor)

    if len(buffer) >= 30:  # 至少等 30 筆資料後再檢查時間範圍
        time_span = (buffer[-1]["timestamp"] - buffer[0]["timestamp"]).total_seconds()
        if time_span >= window_sec:
            df = pd.DataFrame(buffer).sort_values("timestamp")

            # 建立標準時間軸並插值
            start_time = df["timestamp"].iloc[0]
            target_times = [start_time + pd.Timedelta(milliseconds=1000 * i / sampling_rate) for i in range(window_size)]
            interp_df = pd.DataFrame({"timestamp": target_times})
            df_interp = pd.merge_asof(interp_df, df, on="timestamp", direction='nearest', tolerance=pd.Timedelta(milliseconds=50))
            df_interp[sensors] = df_interp[sensors].interpolate().bfill().ffill().round()

            # 特徵與預測
            feats = extract_features(df_interp)
            feature_json_str = json.dumps(feats.to_dict(orient="records")[0])
            timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            print("[預測結果]")
            for i in range(6):
                # pred = models[i].predict(feats.values)[0]
                # prob = models[i].predict_proba(feats.values)[0][pred]
                # print(f"Sensor {i+1}: 洩漏 {leak_levels[pred]}%｜機率 {prob:.2f} → {policy_map[pred]}")
                
                pred = models[i].predict(feats.values)[0]
                prob = models[i].predict_proba(feats.values)[0][pred]
                leakage_level = leak_levels[pred]
                policy = policy_map[pred]
                
                # sensor_id → machine_id 對應邏輯
                if i in [0, 1]:
                    machine_id = 158
                elif i in [2, 3]:
                    machine_id = 159
                else:
                    machine_id = 160

                # print(f"Sensor {i+1}: 洩漏 {leakage_level}% → {policy}")

                # 建立資料字典
                result = {
                    "timestamp": timestamp_str,
                    "sensor_id": i + 1,
                    "machine_id": machine_id,
                    "leakage_level": leakage_level,
                    "predicted_class": f"洩漏等級{pred}",
                    "maintenance_policy": policy,
                    "failure_probability": failure_probability_prediction(df_interp),
                    "feature_json": feature_json_str
                }
                # print(f"Sensor {i+1}: 洩漏 {leakage_level}% → {policy}｜故障率：{result['failure_probability']}%")
                print(f"寫入資料庫: {result}") 

            buffer.clear()  # 清空 11 秒資料重新開始

    time.sleep(0.01)  # 嘗試 100Hz 調用 read_sensor
