import pandas as pd
import numpy as np

# 讀取 CSV
csv_path = "all_leakage_LSTM_dataset.csv"
df = pd.read_csv(csv_path)

# 提取 label 欄位
label_cols = [f"label_sensor{i}" for i in range(1, 7)]
y = df[label_cols].values  # shape: [samples, 6]

# 提取特徵欄位（排除 label 欄位）
X = df.drop(columns=label_cols).values  # shape: [samples, features]

# 轉換成 npz 並儲存
npz_path = "omg.npz"
np.savez_compressed(npz_path, X=X, y=y)
