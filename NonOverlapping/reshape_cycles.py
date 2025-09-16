import os
import numpy as np
import pandas as pd

now_dir = os.path.dirname(os.path.abspath(__file__))

# ========== 可調參數 ==========
INPUT_PATH  = "row_data/兩根/7圈/第一二根七圈.csv"
TS_COL      = "si_ts"
VAL_COL     = "psr_val_5"
HIGH_ON = 0.2      # 上升沿門檻（進入週期）
HIGH_OFF = 0.1     # 下降沿門檻（離開週期）
LOW_END = 0.05     # 低壓區界線（完整週期起點）
OUT_DIR     = f"{now_dir}/cycles_out_simple_12_7"
# =============================

def find_sensor_cycles(x: np.ndarray, low_th: float, high_th: float):
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
            cycles.append((start, end))
    return cycles

def find_sensor2_cycles(x: np.ndarray, low_th: float, high_th: float):
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
    return cycles


def main():
    # 輸入
    df = pd.read_csv(INPUT_PATH)
    df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")
    raw = df[VAL_COL].astype(float).values

    # 偵測週期
    if VAL_COL == "psr_val_1":
        cycle_fn = find_sensor2_cycles
    else:
        cycle_fn = find_sensor_cycles

    # 呼叫週期偵測
    all_cycles = cycle_fn(raw, LOW_END, HIGH_ON)

    # 準備輸出資料夾
    os.makedirs(OUT_DIR, exist_ok=True)

    # 輸出每個週期 CSV，並建立索引
    index_rows = []
    for idx, (s, e) in enumerate(all_cycles, start=1):
        seg = df.iloc[s:e+1].copy().reset_index(drop=True)
        seg_path = os.path.join(OUT_DIR, f"cycle_{idx:03d}.csv")
        seg.to_csv(seg_path, index=False, encoding="utf-8-sig")

        index_rows.append({
            "cycle_id": idx,
            "start_idx": s,
            "end_idx": e,
            "start_ts": seg[TS_COL].iloc[0],
            "end_ts": seg[TS_COL].iloc[-1],
            "file": seg_path
        })

    index_df = pd.DataFrame(index_rows)
    index_path = os.path.join(OUT_DIR, "index.csv")
    index_df.to_csv(index_path, index=False, encoding="utf-8-sig")

    # 總結
    print("=== 週期擷取完成（簡化版）===")
    print(f"來源檔：{INPUT_PATH}")
    print(f"總共擷取週期數：{len(all_cycles)}")
    print(f"輸出資料夾：{OUT_DIR}")

if __name__ == "__main__":
    main()
