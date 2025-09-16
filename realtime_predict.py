import os
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from modules import MySQLConnector, CycleDetector, MultiLeakArbiter, send_sms, lifespan_estimation, error_log, mysql_log, mqtt_log
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
LIFE_LIMIT_TABLE = os.getenv("LIFE_LIMIT_TABLE")

OUTPUT_DIR = "cycles_out"
MODELS_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CYCLE_ERROR = 0

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

# ===== 推論輔助 =====
def _safe_vector_from_features(feats: dict, feature_cols: list[str], model=None):
    try:
        data = [[feats[c] for c in feature_cols]]
    except KeyError:
        return None
    # NaN/Inf 檢查
    for v in data[0]:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return None
    X = pd.DataFrame(data, columns=feature_cols)
    if model is not None and hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=list(model.feature_names_in_), fill_value=0.0)
    return X

def load_models() -> dict:
    models = {}
    for i in range(1, 7):
        name = f"sensor{i}"
        pkl_path = os.path.join(MODELS_DIR, f"{name}_rf_model.pkl")
        if os.path.exists(pkl_path):
            models[name] = joblib.load(pkl_path)
        else:
            print(f"⚠️ 找不到模型：{pkl_path}（此感測器將不做推論）")
    return models

def predict_with_model(model, X_df):
    pred = int(model.predict(X_df)[0])
    prob_map = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_df)[0]
        classes = list(getattr(model, "classes_", []))
        for cls, p in zip(classes, proba):
            prob_map[int(cls)] = float(p)
        # 補齊缺的類別
        for k in LABEL_MAP.keys():
            prob_map.setdefault(k, 0.0)
    return pred, LABEL_MAP.get(pred, str(pred)), prob_map

def ensure_features_index_header(path: str):
    if not os.path.exists(path):
        pd.DataFrame(columns=[
            "cycle_id", "sensor_key", "sensor_name",
            "start_ts", "end_ts", "n_points",
            *FEATURE_COLS,
            "pred_label", "pred_name",
            "p_0", "p_7", "p_10", "leak_score",
            "file"
        ]).to_csv(path, index=False, encoding="utf-8-sig")
        
def on_timeout(sensor_key, duration_sec, start_ts, now_ts):
    # 你也可以在這裡：寫入告警表、發通知、丟API等
    print(f"⚠️ 即時告警：{sensor_key} 週期已 {duration_sec:.2f}s（{start_ts} → {now_ts}）")

# ===== 主流程 =====
def main():
    global CYCLE_ERROR
    sql = MySQLConnector(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, DATA_DB)
    sql2 = MySQLConnector(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, POLICY_DB)
    
    detectors = {
        key: CycleDetector(low_th=LOW_END, high_th=HIGH_ON, sensor_key=key, mode=MODE_MAP[key], fixed_duration_sec=11.0)
        for key in MODE_MAP.keys()
    }
    
    cycle_counters = {key: 1 for key in MODE_MAP.keys()}
    models = load_models()

    features_index_path = os.path.join(OUTPUT_DIR, "features_index.csv")
    ensure_features_index_header(features_index_path)

    arbiter = MultiLeakArbiter(
        sensors=list(SENSOR_NAME.values()), 
        batch_sec=300,
    )

    while True:
        record = sql.get_latest_row(TABLE_NAME)
        record['si_ts'] = _to_datetime(record['si_ts'])
        
        do_counter_records = sql.get_latest_row(LIFE_LIMIT_TABLE)
        
        for key, det in detectors.items():
            do_count = do_counter_records[f'do_count_{int(key.split("_")[2]) + 1}']

            Y, cond_prob = lifespan_estimation(do_count)

            cycle_df = det.update(record)
            if cycle_df is None or cycle_df.empty:
                continue

            sensor_name = SENSOR_NAME[key]
            
            if det.last_cycle_valid is False:
                CYCLE_ERROR += 1
                if CYCLE_ERROR >= 3:
                    send_sms(USERNAME, PASSWORD, API, MOBILE, f"⚠️ {sensor_name} 連續三次異常，請檢查系統！")
                    # print(f"⚠️ {sensor_name} 異常窗，略過（{det.last_cycle_reason}）") # 加入counter
                continue
            
            cid = cycle_counters[key]
            cycle_file = os.path.join(OUTPUT_DIR, f"{sensor_name}_cycle_{cid:03d}.csv")
            cycle_df.to_csv(cycle_file, index=False, encoding="utf-8-sig")

            feats = extract_features(cycle_df, key) or {}
            model = models.get(sensor_name)
            X_row = _safe_vector_from_features(feats, FEATURE_COLS, model=model)

            pred_label, pred_name, prob_map = None, None, {}
            if model is not None and X_row is not None:
                pred_label, pred_name, prob_map = predict_with_model(model, X_row)

            p0 = prob_map.get(0, 0.0)
            p7 = prob_map.get(1, 0.0)
            p10 = prob_map.get(2, 0.0)
            leak_score = p7 + p10

            start_ts = pd.to_datetime(cycle_df[TIME_COL].iloc[0])
            end_ts   = pd.to_datetime(cycle_df[TIME_COL].iloc[-1])
            
            winners, details = arbiter.update(sensor_name, end_ts, pred_label, prob_map)
            
            if winners:
                send_sms(USERNAME, PASSWORD, API, MOBILE, f"{sensor_name} 洩漏！詳情：{details}")
                # print(f"仲裁後目前洩漏: {winners} ｜細節: {details}")
            
            result = {
                "sensor_id": int(key.split("_")[2]) + 1,
                "machine_id": record['si_ip'],
                "predicted_class": pred_name,
                "maintenance_policy": POLICY_MAP.get(pred_label, "未知"),
                "lifespan_estimation": f"閥門 {int(key.split('_')[2]) + 1}：已運作 {do_count:,} 次後，還能再 {Y:,} 次的機率：約 {cond_prob:.2%}",
            }

            row = {
                "cycle_id": cid,
                "sensor_key": key,
                "sensor_name": sensor_name,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "n_points": len(cycle_df),
                **{c: feats.get(c, None) for c in FEATURE_COLS},
                "pred_label": pred_label,
                "pred_name": pred_name,
                "p_0": p0, "p_7": p7, "p_10": p10,
                "leak_score": leak_score,
                "file": cycle_file,
            }
            pd.DataFrame([row]).to_csv(
                features_index_path, mode="a", header=False, index=False, encoding="utf-8-sig"
            )
            
            print(
                f"✅ {sensor_name} 週期 #{cid}｜{start_ts}→{end_ts}｜"
                f"{pred_name if pred_name else '—'}｜"
                f"p0={p0:.2f} p7={p7:.2f} p10={p10:.2f} leak={leak_score:.2f}｜"
                f"file={os.path.basename(cycle_file)}"
            )
            sql2.insert_data(POLICY_TABLE, result)
            cycle_counters[key] += 1

if __name__ == "__main__":
    main()
