import os
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from modules import CycleDetector, ProbVoteBuffer, MultiLeakArbiter, send_sms, error_log, mysql_log, mqtt_log
from config.constants import *

load_dotenv()

USERNAME = os.getenv("TWSMS_USER")
PASSWORD = os.getenv("TWSMS_PASS")
API = os.getenv("TWSMS_API")
MOBILE = os.getenv("TWSMS_MOBILE")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
POLICY_DB = os.getenv("POLICY_DB")
POLICY_TABLE = os.getenv("POLICY_TABLE")
DATA_DB = os.getenv("DATA_DB")
TABLE_NAME = os.getenv("TABLE_NAME")

# ===== 參數 =====
OUTPUT_DIR = "cycles_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODELS_DIR = "models_many_normal"
CYCLE_ERROR = 0

MODEL_PATH = "models/rf_multioutput.pkl"

# ===== 時間 & 資料流 =====
def _to_datetime(val):
    try:
        ts = pd.to_datetime(val, errors="coerce")
        if pd.notna(ts): return ts
    except Exception:
        pass
    try:
        f = pd.to_numeric(val, errors="coerce")
        if pd.isna(f): return None
        if f > 1e15:   return pd.to_datetime(f, unit="ns", errors="coerce")
        elif f > 1e14: return pd.to_datetime(f, unit="us", errors="coerce")
        elif f > 1e11: return pd.to_datetime(f, unit="ms", errors="coerce")
        elif f > 1e9:  return pd.to_datetime(f, unit="s",  errors="coerce")
        else:          return pd.to_datetime(f, unit="s",  errors="coerce")
    except Exception:
        return None

def simulate_data_stream():
    df = pd.read_csv('row_data/兩根/7圈/第一四根七圈_1.csv', encoding="utf-8-sig")
    for _, row in df.iterrows():
        record = {
            'si_ts': _to_datetime(row['si_ts']),
            'psr_val_0': float(row['psr_val_0']),
            'psr_val_1': float(row['psr_val_1']),
            'psr_val_2': float(row['psr_val_2']),
            'psr_val_3': float(row['psr_val_3']),
            'psr_val_4': float(row['psr_val_4']),
            'psr_val_5': float(row['psr_val_5']),
        }
        # time.sleep(0.02)
        yield record

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
    # slope / stability 如需可打開
    holding_time = _duration_above_threshold_irregular(ts, x, thr=PRESSURE_THRESHOLD)
    return {"mean": x_mean, "std": x_std, "range": x_range, "holding_time": holding_time}
        
# ===== 跨感測器特徵（精簡示範，可再擴充）=====
# def build_cross_features(flat_feats: dict):
#     # 收集六根的均值
#     means = [flat_feats.get(f"sensor{i}_mean", 0.0) for i in range(1, 7)]
#     overall_mean = float(np.mean(means)) if len(means) else 0.0
#     # 例：最大-最小的均值差、均值的標準差、每根相對其他的差與比
#     cross = {
#         "max_mean_diff": float(np.max(means) - np.min(means)) if means else 0.0,
#         "std_of_means": float(np.std(means)) if means else 0.0,
#     }
#     for i in range(1, 7):
#         mi = flat_feats.get(f"sensor{i}_mean", 0.0)
#         cross[f"sensor{i}_diff_mean"] = mi - overall_mean
#         cross[f"sensor{i}_ratio_mean"] = (mi / overall_mean) if overall_mean != 0 else 0.0
#     # 排名（1~6，1=最低、6=最高）
#     ranks = pd.Series(means).rank(method="average").tolist()
#     for i in range(1, 7):
#         cross[f"sensor{i}_rank_mean"] = ranks[i-1]
#     return cross

def ensure_merged_index_header(path: str, model_feature_names: list[str]):
    if not os.path.exists(path):
        cols = ["cycle_id", "start_ts", "end_ts"]
        # 模型特徵（保存便於追溯）
        cols += list(model_feature_names)
        # 每根的預測與顯示名稱
        for i in range(1, 7):
            cols += [f"pred_sensor{i}", f"pred_name_sensor{i}"]
        pd.DataFrame(columns=cols).to_csv(path, index=False, encoding="utf-8-sig")
        
def label_name(cls_value: int) -> str:
    return LABEL_MAP.get(int(cls_value), str(cls_value))

def extract_features_all(cycle_dict: dict):
    """
    cycle_dict: { "psr_val_0": df0, ..., "psr_val_5": df5 }
    回傳: 只有六根感測器的單根特徵（mean/std/range/holding_time），不含跨感測器特徵
    """
    flat_feats = {}
    for i in range(6):
        key = f"psr_val_{i}"
        df = cycle_dict.get(key)
        f = extract_features(df, key) if df is not None else None
        if f:
            for k, v in f.items():
                flat_feats[f"sensor{i+1}_{k}"] = v
        else:
            # 該感測器沒有資料就補 0
            for k in FEATURE_COLS:
                flat_feats[f"sensor{i+1}_{k}"] = 0.0

    return flat_feats


# ===== 主流程 =====
def main():
    global CYCLE_ERROR
    detectors = {
        key: CycleDetector(low_th=LOW_END, high_th=HIGH_ON, sensor_key=key, mode=MODE_MAP[key], fixed_duration_sec=11.0)
        for key in MODE_MAP.keys()
    }

    cycle_counters = 1
    model = joblib.load(MODELS_DIR)   # 單一六合一模型

    features_index_path = os.path.join(OUTPUT_DIR, "features_index.csv")
    if not os.path.exists(features_index_path):
        pd.DataFrame(columns=[
            "cycle_id",
            *(f"sensor{i+1}_{c}" for i in range(6) for c in FEATURE_COLS),
            *(f"pred_sensor{i+1}" for i in range(6))
        ]).to_csv(features_index_path, index=False, encoding="utf-8-sig")

    # 初始化 pending_cycles
    pending_cycles = {f"psr_val_{i}": None for i in range(6)}

    for record in simulate_data_stream():
        for key, det in detectors.items():
            cycle_df = det.update(record)
            if cycle_df is not None and not cycle_df.empty:
                pending_cycles[key] = cycle_df
                
            sensor_name = SENSOR_NAME[key]
            
            if det.last_cycle_valid is False:
                CYCLE_ERROR += 1
                if CYCLE_ERROR >= 3:
                    # send_sms(USERNAME, PASSWORD, API, MOBILE, f"⚠️ {sensor_name} 連續三次異常，請檢查系統！")
                    print(f"⚠️ {sensor_name} 異常窗，略過（{det.last_cycle_reason}）") # 加入counter
                continue

        # Debug: 目前有幾根感測器已完成週期
        ready_count = sum(df is not None and not df.empty for df in pending_cycles.values())
        print(f"🔄 已完成週期的感測器數量: {ready_count}/6", end="\r")

        # 等到六根都有 cycle → 才做預測
        if all(df is not None and not df.empty for df in pending_cycles.values()):
            feats = extract_features_all(pending_cycles)
            X_row = pd.DataFrame([feats]).reindex(columns=model.feature_names_in_, fill_value=0.0)

            y_pred = model.predict(X_row)[0]

            row = {
                "cycle_id": cycle_counters,
                **feats,
                **{f"pred_sensor{i+1}": int(y_pred[i]) for i in range(6)}
            }

            pd.DataFrame([row]).to_csv(
                features_index_path, mode="a", header=False, index=False, encoding="utf-8-sig"
            )

            print(f"\n✅ 週期 #{cycle_counters} ｜預測結果: {y_pred}")

            cycle_counters += 1

            # 重置，等下一輪
            pending_cycles = {f"psr_val_{i}": None for i in range(6)}

if __name__ == "__main__":
    main()
