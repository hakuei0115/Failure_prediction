import os
import time
import numpy as np
import pandas as pd
import joblib

# ===== 基本設定 =====
MYSQL_HOST = "localhost"
MYSQL_USER = "aict702"
MYSQL_PASSWORD = "aict702@Lab702"
POLICY_DB = "sensorTest"
POLICY_TABLENAME = "leakage_predictions"
DATA_DB = "mm_si"
TABLE_NAME = "si_prs"

# ===== 參數 =====
HIGH_ON = 0.2
LOW_END = 0.1
MAX_CYCLE_SEC = 50.0
ID_COL = "si_id"
TIME_COL = "si_ts"
PRESSURE_THRESHOLD = HIGH_ON
OUTPUT_DIR = "cycles_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAP = {0: "✅ 正常", 1: "⚠️ 洩漏（7圈）", 2: "🚨 洩漏（10圈）"}
# LABEL_MAP = {0: "洩漏等級0", 5: "洩漏等級1", 10: "洩漏等級2"}
# LEAKAGE_MAP = {0: 0, 5: 1, 10: 2}
FEATURE_COLS = ["mean", "std", "holding_time", "range"]
POLICY_MAP = {
    0: "無需維修",
    1: "安排停機檢查與氣密測試",
    2: "緊急停機，立即維修並追蹤"
}


MODE_MAP = {
    "psr_val_0": "single",
    "psr_val_1": "double",  # 感測器二
    "psr_val_2": "single",
    "psr_val_3": "single",
    "psr_val_4": "single",
    "psr_val_5": "single",
}
SENSOR_NAME = {
    "psr_val_0": "sensor1",
    "psr_val_1": "sensor2",
    "psr_val_2": "sensor3",
    "psr_val_3": "sensor4",
    "psr_val_4": "sensor5",
    "psr_val_5": "sensor6",
}
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
    df = pd.read_csv('row_data/一根/7圈/第三根七圈_1.csv', encoding="utf-8-sig")
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

# ===== 逐筆切週期（保留你的版本） =====
class CycleDetector:
    """
    逐筆偵測週期（即時串流）
    - 一般模式: mode="single"/"double" (原邏輯)
    - 固定長度模式: fixed_duration_sec>0 → 只要偵測到 低→高，就從「最後一筆低壓點」起算 fixed_duration_sec 秒並切段
    """
    def __init__(self, low_th=0.1, high_th=0.2, sensor_key="psr_val_0", mode="single",
                 max_cycle_sec=None, on_timeout=None, fixed_duration_sec=11.0):
        assert mode in ("single", "double")
        self.low_th = low_th
        self.high_th = high_th
        self.key = sensor_key
        self.mode = mode

        # 即時超時告警（沿用）
        self.max_cycle_sec = max_cycle_sec
        self.on_timeout = on_timeout

        # ---- ramp 追蹤（新版）----
        self.in_low = False
        self.last_low_rec = None   # 低壓區最後一筆（<low_th）
        self.ramp_start_rec = None # 離開低壓後，用 last_low_rec 作為起點

        # 一般模式所需
        self.state = "IDLE"
        self.cycle_buffer = []
        self.cycle_start_ts = None
        self.timeout_alerted = False

        # 固定時長模式
        self.fixed_duration_sec = fixed_duration_sec
        self.fixed_deadline_ts = None
        self.pre_buffer = []   # 已離開低壓到升破 high_th 之間的點（含 ramp_start_rec）
        self.fixed_state = "IDLE"  # IDLE → ARMED(離低壓等待升破) → ACTIVE(收集到截止) → WAIT_LOW

    def _val(self, record): 
        return float(record[self.key])

    def _track_ramp(self, record, val):
        """維護 last_low_rec / ramp_start_rec 與 pre_buffer。"""
        if val < self.low_th:
            # 還在低壓：更新最後一筆低壓點，清 ramp 與 pre_buffer
            self.in_low = True
            self.last_low_rec = record
            self.ramp_start_rec = None
            self.pre_buffer = []
        else:
            if self.in_low:
                # 剛離開低壓：起點=最後一筆低壓點
                self.ramp_start_rec = self.last_low_rec if self.last_low_rec is not None else record
                self.pre_buffer = [self.ramp_start_rec]
                self.in_low = False
            # 若已不在低壓且尚未升破 high_th，將點先放進 pre_buffer（為了不漏掉上升段）
            if self.ramp_start_rec is not None and val < self.high_th:
                self.pre_buffer.append(record)

    def _maybe_timeout_alert(self, record):
        if self.max_cycle_sec is None or self.cycle_start_ts is None or self.timeout_alerted:
            return
        now_ts = record.get(TIME_COL)
        if pd.isna(now_ts) or pd.isna(self.cycle_start_ts):
            return
        duration = (now_ts - self.cycle_start_ts).total_seconds()
        if duration > self.max_cycle_sec:
            self.timeout_alerted = True
            if callable(self.on_timeout):
                self.on_timeout(self.key, duration, self.cycle_start_ts, now_ts)
            else:
                print(f"⚠️ 即時告警：{self.key} 目前週期已 {duration:.2f}s (> {self.max_cycle_sec}s)")

    # ================= 固定時長模式 =================
    def _update_fixed(self, record, val):
        ts = record.get(TIME_COL)

        if self.fixed_state == "IDLE":
            # 等待：先離開低壓 → ramp_start_rec 成立 → 再升破 high_th 才啟動
            if (self.ramp_start_rec is not None) and (val >= self.high_th):
                # 起點=最後一筆低壓點；窗口=起點時間+fixed_duration
                self.cycle_buffer = list(self.pre_buffer) + [record]
                self.cycle_start_ts = self.ramp_start_rec.get(TIME_COL) if self.ramp_start_rec else ts
                self.fixed_deadline_ts = self.cycle_start_ts + pd.Timedelta(seconds=self.fixed_duration_sec)
                self.timeout_alerted = False
                self.fixed_state = "ACTIVE"
            return None

        if self.fixed_state == "ACTIVE":
            # 收集直到截止（包含 <= deadline 的點）
            if ts <= self.fixed_deadline_ts:
                self.cycle_buffer.append(record)
                # 可選：一般超時告警（與 fixed_duration 無關），若有設定也檢查
                self._maybe_timeout_alert(record)
                return None
            # 超過截止 → 輸出（丟棄超過截止的這筆）
            out = pd.DataFrame([r for r in self.cycle_buffer if r.get(TIME_COL) <= self.fixed_deadline_ts])
            # 切段後進入 WAIT_LOW：必須先回到低壓，再下一次低→高才會重新開始
            self.fixed_state = "WAIT_LOW"
            self.cycle_buffer = []
            self.cycle_start_ts = None
            self.fixed_deadline_ts = None
            self.ramp_start_rec = None
            self.pre_buffer = []
            return out

        if self.fixed_state == "WAIT_LOW":
            # 等待重新回到低壓（_track_ramp 會在 <low_th 時清/記錄 last_low_rec）
            # 什麼都不做，直到下次離低壓並升破 high_th 會回到 IDLE->ACTIVE 流程
            if val < self.low_th:
                # 回到低壓，之後離開時會重新 arm
                self.fixed_state = "IDLE"
            return None

        return None

    # ================= 原本的 single/double 模式 =================
    def _start_cycle_from(self, rec):
        self.cycle_buffer = [rec] if rec is not None else []
        ts0 = rec.get(TIME_COL) if rec else None
        self.cycle_start_ts = ts0 if pd.notna(ts0) else None
        self.timeout_alerted = False

    def _end_cycle_reset(self):
        self.cycle_start_ts = None
        self.timeout_alerted = False

    def _update_single(self, record, val):
        if self.state == "IDLE":
            if (self.ramp_start_rec is not None) and (val >= self.high_th):
                self._start_cycle_from(self.ramp_start_rec)
                self.state = "FIRST_HIGH"
            return None
        if self.state == "FIRST_HIGH":
            self.cycle_buffer.append(record)
            if val < self.low_th:
                self.state = "WAIT_SECOND_RISE"
            return None
        if self.state == "WAIT_SECOND_RISE":
            if val >= self.high_th:
                out = pd.DataFrame(self.cycle_buffer[:-1]) if self.cycle_buffer else None
                self.state = "FIRST_HIGH"
                if self.ramp_start_rec is not None:
                    self._start_cycle_from(self.ramp_start_rec)
                else:
                    self.cycle_buffer = []
                    self._end_cycle_reset()
                return out
            self.cycle_buffer.append(record)
            return None
        return None

    def _update_double(self, record, val):
        if self.state == "IDLE":
            if (self.ramp_start_rec is not None) and (val >= self.high_th):
                self._start_cycle_from(self.ramp_start_rec)
                self.state = "HIGH1"
            return None
        if self.state == "HIGH1":
            self.cycle_buffer.append(record); 
            if val < self.low_th: self.state = "LOW1"
            return None
        if self.state == "LOW1":
            self.cycle_buffer.append(record); 
            if val >= self.high_th: self.state = "HIGH2"
            return None
        if self.state == "HIGH2":
            self.cycle_buffer.append(record); 
            if val < self.low_th: self.state = "LOW2"
            return None
        if self.state == "LOW2":
            self.cycle_buffer.append(record)
            if val >= self.high_th:
                out = pd.DataFrame(self.cycle_buffer[:-1]) if self.cycle_buffer else None
                self.state = "HIGH1"
                if self.ramp_start_rec is not None:
                    self._start_cycle_from(self.ramp_start_rec)
                else:
                    self.cycle_buffer = []
                    self._end_cycle_reset()
                return out
            return None
        return None

    # ================= 公用入口 =================
    def update(self, record):
        val = self._val(record)
        self._track_ramp(record, val)
        if self.fixed_duration_sec and self.fixed_duration_sec > 0:
            return self._update_fixed(record, val)
        # 否則走原本 single/double 規則
        out = self._update_single(record, val) if self.mode == "single" else self._update_double(record, val)
        # 在一般模式下也可做即時超時檢查
        if self.state in ("FIRST_HIGH", "WAIT_SECOND_RISE", "HIGH1", "LOW1", "HIGH2", "LOW2"):
            self._maybe_timeout_alert(record)
        return out

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
        
def on_timeout(sensor_key, duration_sec, start_ts, now_ts):
    # 你也可以在這裡：寫入告警表、發通知、丟API等
    print(f"⚠️ 即時告警：{sensor_key} 週期已 {duration_sec:.2f}s（{start_ts} → {now_ts}）")
    
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
    detectors = {
        key: CycleDetector(low_th=LOW_END, high_th=HIGH_ON, sensor_key=key,
                           mode=MODE_MAP[key], max_cycle_sec=MAX_CYCLE_SEC,
                           on_timeout=on_timeout, fixed_duration_sec=11.0)
        for key in MODE_MAP.keys()
    }

    cycle_counters = 1
    model = joblib.load("models/rf_multioutput.pkl")   # 單一六合一模型

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
