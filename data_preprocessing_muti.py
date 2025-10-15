import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ========== 參數設定 ==========
DATA_ROOT = "train"  # 根目錄
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
    if len(x) < 2 or ts.isna().any():
        return None

    # 保證時間遞增
    if not ts.is_monotonic_increasing:
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        x = df[col].astype(float).to_numpy()
        ts = pd.to_datetime(df[TIME_COL])

    # 計算保壓時間（全段）
    holding_time = _duration_above_threshold_irregular(ts, x, thr=PRESSURE_THRESHOLD)

    # ===== 只取保壓區間特徵 =====
    mask = x > PRESSURE_THRESHOLD
    if np.sum(mask) == 0:  # 若完全沒有保壓段
        return None

    x_hold = x[mask]
    x_max, x_min = float(np.max(x)), float(np.min(x))
    x_mean = float(np.mean(x_hold))
    x_std = float(np.std(x_hold))
    x_range = float(x_max - x_min)

    return { "mean": x_mean, "std": x_std, "range": x_range, "holding_time": holding_time }

# ===== 解析檔名（處理兩根洩漏） =====
def parse_filename_label(filename: str, default_label: str, sensor_idx: int) -> int:
    """
    filename: 檔名，如 cycle_001_12_7.csv
    default_label: 資料夾名稱的標籤 (正常/7圈/10圈)
    sensor_idx: 目前處理的感測器索引 (0~5)
    回傳該感測器的類別 (0=正常,1=7圈,2=10圈)
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) >= 4:
        leak_sensors = parts[2]  # "12" → 第1、2根
        leak_label = parts[3]    # "7"
        if str(sensor_idx+1) in leak_sensors:
            return LABEL_MAP.get(f"{leak_label}圈", LABEL_MAP[default_label])
        else:
            return LABEL_MAP["正常"]
    else:
        return LABEL_MAP[default_label]

# ===== 單一感測器處理 =====
def process_sensor(sensor: str, data_root: str):
    records = []
    for label_folder in LABEL_MAP.keys():
        class_dir = os.path.join(data_root, label_folder, sensor)
        print(f"[DEBUG] 處理資料夾: {class_dir}")
        if not os.path.exists(class_dir):
            continue

        for file in tqdm(os.listdir(class_dir), desc=f"[{os.path.basename(data_root)}] {sensor} {label_folder}"):
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

            # 判斷標籤：一根 → 用資料夾名稱；兩根 → 用檔名解析
            features["label"] = parse_filename_label(file, label_folder, sensor_idx)
            features["file"] = file
            records.append(features)

    return pd.DataFrame(records)

# ===== 主程式 =====
def main():
    data_roots = [os.path.join(DATA_ROOT, d) for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    print(f"發現資料來源：{data_roots}")

    for sensor in SENSOR_LIST:
        all_records = []
        for root in data_roots:
            df = process_sensor(sensor, root)
            if not df.empty:
                all_records.append(df)

        if all_records:
            merged = pd.concat(all_records, ignore_index=True)
            out_path = os.path.join(OUTPUT_DIR, f"{sensor}_train.csv")
            merged.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[✓] 輸出：{out_path} ({len(merged)} 筆樣本)")

if __name__ == "__main__":
    main()
