#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np

# ===== 可調整區 =====
TRAIN_ROOT = "train/一根"                  # 你的 train 根目錄
TIME_COL   = "si_ts"                  # 週期檔裡的時間欄位名
CSV_PATTERN = "*.csv"                 # 檔名樣式
EXCLUDE_NAMES = {"index.csv"}         # 要略過的檔名（例如索引檔）
OUT_FILES_CSV  = "train_summary_files.csv"
OUT_GROUPS_CSV = "train_summary_groups.csv"

def safe_read_csv(path: str):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        print(f"✖ 讀取失敗: {path} | {e}")
        return None

def describe_csv(path: str, condition: str, sensor: str) -> dict:
    info = {
        "condition": condition,
        "sensor": sensor,
        "file": os.path.basename(path),
        "path": path,
        "rows": None,
        "start_ts": None,
        "end_ts": None,
        "duration_sec": None,
        "error": None,
    }

    df = safe_read_csv(path)
    if df is None:
        info["error"] = "read_error"
        return info

    if TIME_COL not in df.columns:
        info["error"] = f"missing column '{TIME_COL}'"
        return info

    ts = pd.to_datetime(df[TIME_COL], errors="coerce")
    ok = ts.notna()
    if not ok.any():
        info["error"] = "all timestamps NaT"
        return info

    ts = ts[ok]
    if not ts.is_monotonic_increasing:
        ts = ts.sort_values()

    start_ts = ts.iloc[0]
    end_ts   = ts.iloc[-1]
    duration_sec = (end_ts - start_ts).total_seconds()

    info.update({
        "rows": int(ok.sum()),
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "duration_sec": float(round(duration_sec, 6)),
    })
    return info

def walk_train_tree(root: str):
    """
    假設結構：
      train/
        正常/
          sensor1/
            *.csv
          ...
        7圈/
          sensor1/
          ...
        10圈/
          sensor1/
          ...
    會回傳 (condition, sensor, csv_path) 的 generator
    """
    if not os.path.isdir(root):
        print(f"✖ 找不到資料夾：{root}")
        return

    for condition in sorted(os.listdir(root)):
        cond_dir = os.path.join(root, condition)
        if not os.path.isdir(cond_dir):
            continue
        # 僅抓 sensor* 資料夾
        sensor_dirs = [d for d in sorted(os.listdir(cond_dir)) if d.startswith("sensor")]
        for sensor in sensor_dirs:
            sensor_dir = os.path.join(cond_dir, sensor)
            if not os.path.isdir(sensor_dir):
                continue
            # 列出 csv 檔
            for path in sorted(glob.glob(os.path.join(sensor_dir, CSV_PATTERN))):
                name = os.path.basename(path)
                if name in EXCLUDE_NAMES:
                    continue
                yield (condition, sensor, path)

def summarize_groups(df_files: pd.DataFrame) -> pd.DataFrame:
    # 僅取無錯的檔案做分組統計
    ok = df_files["error"].isna()
    g = df_files[ok].groupby(["condition", "sensor"], as_index=False)
    out = g.agg(
        n_files=("file", "count"),
        total_rows=("rows", "sum"),
        total_duration_sec=("duration_sec", "sum"),
        mean_duration_sec=("duration_sec", "mean"),
        median_duration_sec=("duration_sec", "median"),
        min_duration_sec=("duration_sec", "min"),
        max_duration_sec=("duration_sec", "max"),
    )
    # 數值欄位四捨五入
    num_cols = [c for c in out.columns if c.endswith("_sec")]
    out[num_cols] = out[num_cols].apply(lambda s: np.round(s.astype(float), 6))
    return out

def main():
    records = []
    for condition, sensor, path in walk_train_tree(TRAIN_ROOT):
        info = describe_csv(path, condition, sensor)
        records.append(info)
        if info.get("error"):
            print(f"✖ {condition}/{sensor}/{info['file']} | {info['error']}")
        else:
            print(f"✓ {condition}/{sensor}/{info['file']} | "
                  f"{info['start_ts']} → {info['end_ts']} | {info['duration_sec']} s (rows={info['rows']})")

    if not records:
        print("（沒有找到任何 CSV 檔）")
        return

    df_files = pd.DataFrame(records)
    df_files.to_csv(OUT_FILES_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] 檔案明細 → {OUT_FILES_CSV}")

    df_groups = summarize_groups(df_files)
    df_groups.to_csv(OUT_GROUPS_CSV, index=False, encoding="utf-8-sig")
    print(f"[Saved] 分組統計 → {OUT_GROUPS_CSV}")

    # 額外印一個總覽
    if not df_groups.empty:
        print("\n=== 分組統計（前幾列）===")
        print(df_groups.head().to_string(index=False))

if __name__ == "__main__":
    main()
