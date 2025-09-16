import pandas as pd
from modules.detect_cycle import CycleDetector
from config.constants import HIGH_ON, LOW_END, MODE_MAP

def _to_datetime(val):
    try:
        ts = pd.to_datetime(val, errors="coerce")
        if pd.notna(ts):
            return ts
    except Exception:
        pass
    try:
        f = pd.to_numeric(val, errors="coerce")
        if pd.isna(f):
            return None
        if f > 1e15:   return pd.to_datetime(f, unit="ns", errors="coerce")
        elif f > 1e14: return pd.to_datetime(f, unit="us", errors="coerce")
        elif f > 1e11: return pd.to_datetime(f, unit="ms", errors="coerce")
        elif f > 1e9:  return pd.to_datetime(f, unit="s",  errors="coerce")
        else:          return pd.to_datetime(f, unit="s",  errors="coerce")
    except Exception:
        return None

def simulate_data_stream():
    df = pd.read_csv('row_data/一根/10圈/第六根十圈.csv', encoding="utf-8-sig")

    for index, row in df.iterrows():
        record = {
            'si_ts': _to_datetime(row['si_ts']),
            'psr_val_0': float(row['psr_val_0']),
            'psr_val_1': float(row['psr_val_1']),
            'psr_val_2': float(row['psr_val_2']),
            'psr_val_3': float(row['psr_val_3']),
            'psr_val_4': float(row['psr_val_4']),
            'psr_val_5': float(row['psr_val_5']),
        }
        
        yield record  # 返回當前記錄

i = 6

detector = CycleDetector(
        low_th=LOW_END,
        high_th=HIGH_ON,
        sensor_key=f"psr_val_{i-1}",
        mode=MODE_MAP[f"psr_val_{i-1}"],
        fixed_duration_sec=11.0
    )

cycle_id = 1
for record in simulate_data_stream():
    result = detector.update(record)
    if result is not None:
        if detector.last_cycle_valid is False:
            print(f"⚠️ Psr_val_{i-1} 異常窗，略過（{detector.last_cycle_reason}）")
            continue
        
        print(f"✅ 偵測到週期 #{cycle_id}，共 {len(result)} 筆")
        print(result.head(2))
        # 儲存或後處理可加這行：

        result.to_csv(f"train/cycle_{cycle_id:03d}.csv", index=False, encoding="utf-8-sig")
        cycle_id += 1