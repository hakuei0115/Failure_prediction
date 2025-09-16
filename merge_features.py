import os
import pandas as pd

INPUT_DIR = "features_output"
OUTPUT_FILE = "features_output/features_merged.csv"
SENSOR_LIST = [f"sensor{i}" for i in range(1, 7)]

def merge_features():
    # 讀取所有感測器的特徵表
    dfs = []
    for i, sensor in enumerate(SENSOR_LIST, start=1):
        path = os.path.join(INPUT_DIR, f"{sensor}_train.csv")
        if not os.path.exists(path):
            print(f"⚠️ 找不到檔案: {path}")
            continue

        df = pd.read_csv(path, encoding="utf-8-sig")
        
        # 重命名欄位，避免重複
        df = df.rename(columns={
            "mean": f"{sensor}_mean",
            "std": f"{sensor}_std",
            "range": f"{sensor}_range",
            "holding_time": f"{sensor}_holding_time",
            "label": f"label_{i}",
        })
        dfs.append(df)

    if not dfs:
        print("❌ 沒有讀到任何特徵檔案")
        return

    # 依照 file 欄位合併
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="file", how="inner")

    # ===== 輸出 =====
    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[✓] 已輸出合併特徵檔案：{OUTPUT_FILE} ({len(merged)} 筆樣本)")

if __name__ == "__main__":
    merge_features()
