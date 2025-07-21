import pandas as pd
import numpy as np
import joblib

# === 1. 載入測試資料 ===
df = pd.read_csv("test.csv")
label_cols = [f"label_sensor{i}" for i in range(1, 7)]
feature_cols = [col for col in df.columns if col not in label_cols]
X_test = df[feature_cols].values

# === 2. 洩漏等級 → 保養建議對照表 ===
maintenance_policy = {
    0: "無需維修",
    1: "觀察壓力變化，定期巡檢",
    2: "安排停機檢查與氣密測試",
    3: "儘快更換管件或電磁閥元件",
    4: "緊急停機，立即維修並追蹤"
}

# === 3. 載入模型並預測 ===
all_predictions = []

for i in range(6):
    model = joblib.load(f"model_sensor_{i+1}.joblib")
    pred = model.predict(X_test)
    all_predictions.append(pred)

# === 4. 顯示預測結果與保養策略 ===
print("\n🔍 預測結果與保養建議：\n")

for idx in range(len(X_test)):
    print(f"📦 測試樣本 #{idx+1}")
    for sensor_id in range(6):
        level = all_predictions[sensor_id][idx]
        policy = maintenance_policy.get(level, "未知")
        print(f"  Sensor {sensor_id+1}: 洩漏等級 {level} → 保養建議：{policy}")
    print("-" * 50)
