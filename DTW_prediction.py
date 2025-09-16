import os
import numpy as np
import pandas as pd
from dtaidistance import dtw

# ========== 可調參數 ==========
INPUT_PATH  = "row_data/一根/10圈/第一根十圈_1.csv"
TS_COL      = "si_ts"
VAL_COLS = [f"psr_val_{i}" for i in range(6)]
HIGH_ON = 0.2      # 上升沿門檻（進入週期）
HIGH_OFF = 0.1     # 下降沿門檻（離開週期）
LOW_END = 0.05     # 低壓區界線（完整週期起點）
MIN_DURATION_SEC = 8.0  # 最短週期時間（秒）
K = 5               # 滑動平均視窗大小
# =============================

def find_sensor_cycles(x: np.ndarray, low_th: float, high_th: float, ts):
    """
    偵測從低壓區 → 高壓 → 回到低壓的完整波形區段
    回傳 [(start_idx, end_idx), ...]
    """
    cycles = []
    i = 0
    while i < len(x):
        # Step 1: 找起點：從低壓區起升
        while i < len(x) and x[i] > low_th:
            i += 1
        while i < len(x) and x[i] < high_th:
            i += 1
        start = i - 1

        # Step 2: 找終點：回到低壓區
        while i < len(x) and x[i] > low_th:
            i += 1
        # end = i if i < len(x) else len(x) - 1
        
        # 找下一次高壓，作為週期終點
        while i < len(x) and x[i] < high_th:
            i += 1
        end = i - 1

        # 確保不是空週期
        if start < end:
            if ts is not None:
                duration = (ts[end] - ts[start]).total_seconds()
                if duration < MIN_DURATION_SEC:
                    continue
            cycles.append((start, end))

    return cycles

def find_sensor2_cycles(x: np.ndarray, low_th: float, high_th: float, ts):
    """
    偵測感測器二的雙波週期：
    需包含「上升→下降→上升→下降→上升」為一個完整週期
    回傳 [(start_idx, end_idx), ...]
    """
    cycles = []
    i = 0
    while i < len(x):
        # 找第一次上升
        while i < len(x) and x[i] > low_th:
            i += 1
        while i < len(x) and x[i] < high_th:
            i += 1
        start = i - 1

        # 第一次下降
        while i < len(x) and x[i] > low_th:
            i += 1

        # 第二次上升
        while i < len(x) and x[i] < high_th:
            i += 1

        # 第二次下降
        while i < len(x) and x[i] > low_th:
            i += 1

        # 第三次上升（視為週期結束）
        while i < len(x) and x[i] < high_th:
            i += 1
        end = i - 1

        if start < end:
            cycles.append((start, end))
        
        if start < end:
            if ts is not None:
                duration = (ts[end] - ts[start]).total_seconds()
                if duration < MIN_DURATION_SEC:
                    continue
            cycles.append((start, end))
            
    return cycles

def to_binary(x: float) -> str:
    if x < 0.2:
        return "00"
    elif x < 0.3:
        return "01"
    elif x < 0.4:
        return "10"
    else:
        return "11"

def bitstring_to_sequence(bitstring: str):
    mapping = {"00": 0, "01": 1, "10": 2, "11": 3}
    seq = [mapping[bitstring[i:i+2]] for i in range(0, len(bitstring), 2)]
    return seq

def classify_dtw_distance(distance: float) -> str:
    if distance < 4:
        return "正常"
    elif distance < 10:
        return "疑似洩漏（五圈）"
    else:
        return "洩漏（十圈）"

def main():
    df = pd.read_csv(INPUT_PATH)
    df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")

    result_rows = []

    for i, col in enumerate(VAL_COLS):
        print(f"\n=== Sensor {i+1} 處理中 ===")

        cycle_fn = find_sensor2_cycles if i == 1 else find_sensor_cycles
        x = df[col].astype(float).values
        cycles = cycle_fn(x, LOW_END, HIGH_ON, ts=df[TS_COL])

        if not cycles:
            print(f"⚠️ Sensor {i+1} 無法偵測週期")
            continue

        # 載入對應 template
        template_path = f"template/sensor{i+1}_template.txt"
        if not os.path.exists(template_path):
            print(f"⚠️ 找不到 template：{template_path}")
            continue
        with open(template_path, "r") as f:
            template_str = f.read().strip()
        template_seq = bitstring_to_sequence(template_str)

        # 比對每個週期
        for idx, (s, e) in enumerate(cycles, start=1):
            seg = df.iloc[s:e+1].copy().reset_index(drop=True)

            # 滑動平均（也可選用 seg[col] 作為原始值）
            ma_col = f"{col}_ma"
            seg[ma_col] = seg[col].rolling(window=K, min_periods=K).mean()
            seg = seg.dropna().reset_index(drop=True)

            # 二位元轉換
            BIN_COL = "binary_2bit"
            seg[BIN_COL] = seg[ma_col].apply(to_binary)
            binary_string = ''.join(seg[BIN_COL].tolist())
            seq = bitstring_to_sequence(binary_string)

            # 計算 DTW
            dtw_score = dtw.distance(seq, template_seq)
            
            classification = classify_dtw_distance(dtw_score)
            print(f"Cycle {idx}: DTW 距離 = {dtw_score:.2f} → {classification}")

            # 儲存結果
            result_rows.append({
                "sensor_id": i + 1,
                "cycle_id": idx,
                "dtw_distance": round(dtw_score, 2),
                "start_ts": seg[TS_COL].iloc[0],
                "end_ts": seg[TS_COL].iloc[-1],
                "duration_sec": (seg[TS_COL].iloc[-1] - seg[TS_COL].iloc[0]).total_seconds(),
                "length": len(seq)
            })

    # 輸出 CSV
    output_df = pd.DataFrame(result_rows)
    os.makedirs("dtw_result", exist_ok=True)
    output_path = os.path.join("dtw_result", "dtw_scores_10.csv")
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ DTW 比對完成，結果儲存於：{output_path}")


if __name__ == "__main__":
    main()
