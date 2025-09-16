import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ========== 參數設定 ==========
DATA_ROOT = "train/一根"  # 根目錄
SENSOR_LIST = [f"sensor{i}" for i in range(1, 7)]
LABEL_MAP = {"正常": 0, "7圈": 1, "10圈": 2}
PRESSURE_COLS = [f"psr_val_{i}" for i in range(6)]
TIME_COL = "si_ts"
PRESSURE_THRESHOLD = 0.2  # 保壓判斷門檻
OUTPUT_DIR = "features_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =============================

# ===== holding_time（不等間隔，線性插值） =====
def _duration_above_threshold_irregular(ts: pd.Series, x: np.ndarray, thr: float) -> float:
    t = pd.to_datetime(ts).astype("int64").to_numpy() / 1e9
    if len(t) < 2: return 0.0
    total = 0.0
    for i in range(len(t) - 1):
        t0, t1 = t[i], t[i + 1]
        x0, x1 = x[i], x[i + 1]
        dt = t1 - t0
        if dt <= 0: continue
        above0, above1 = (x0 > thr), (x1 > thr)
        if above0 and above1:
            total += dt
        elif above0 != above1:
            tau = (thr - x0) * dt / (x1 - x0) if x1 != x0 else dt / 2.0
            tau = max(0.0, min(dt, tau))
            total += (dt - tau) if (not above0 and above1) else tau
    return float(total)

# ===== 特徵 =====
def extract_features(df: pd.DataFrame, col: str):
    if df is None or df.empty or col not in df or TIME_COL not in df:
        return None
    x = df[col].astype(float).to_numpy()
    ts = pd.to_datetime(df[TIME_COL])
    if len(x) < 2 or ts.isna().any(): return None
    if not ts.is_monotonic_increasing:
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        x = df[col].astype(float).to_numpy()
        ts = pd.to_datetime(df[TIME_COL])

    x_max, x_min = float(np.max(x)), float(np.min(x))
    x_mean, x_std = float(np.mean(x)), float(np.std(x))
    x_range = float(x_max - x_min)
    total_sec = max((ts.iloc[-1] - ts.iloc[0]).total_seconds(), 1e-9)
    
    print(total_sec)
    
    holding_time = _duration_above_threshold_irregular(ts, x, thr=PRESSURE_THRESHOLD)
    return {"mean": x_mean, "std": x_std, "range": x_range, "holding_time": holding_time}

def process_sensor(sensor: str):
    records = []
    for label_folder in LABEL_MAP.keys():
        class_dir = os.path.join(DATA_ROOT, label_folder, sensor)
        print(class_dir)
        if not os.path.exists(class_dir):
            continue

        for file in tqdm(os.listdir(class_dir), desc=f"[{sensor}] {label_folder}"):
            if not file.endswith(".csv"):
                continue
            path = os.path.join(class_dir, file)
            df = pd.read_csv(path, encoding="utf-8-sig")
            sensor_idx = int(sensor[-1]) - 1  # sensor1 → psr_val_0

            pressure_col = f"psr_val_{sensor_idx}"
            if pressure_col not in df.columns or TIME_COL not in df.columns:
                continue

            features = extract_features(df, pressure_col)
            if features is None:
                continue
            features["label"] = LABEL_MAP[label_folder]
            features["file"] = file
            records.append(features)

    return pd.DataFrame(records)


def main():
    for sensor in SENSOR_LIST:
        df = process_sensor(sensor)
        out_path = os.path.join(OUTPUT_DIR, f"{sensor}_train.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[✓] 輸出：{out_path} ({len(df)} 筆樣本)")


if __name__ == "__main__":
    main()
