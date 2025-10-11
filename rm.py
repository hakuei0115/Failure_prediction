import os

# === 參數設定 ===
DATA_ROOT = "train/兩根/正常"   # 根目錄（請改成你的資料夾路徑）
TARGET_RING = "7"     # 要刪除的圈數

deleted_files = []

# === 遞迴搜尋所有檔案 ===
for root, dirs, files in os.walk(DATA_ROOT):
    for file in files:
        if not file.endswith(".csv"):
            continue

        parts = file.replace(".csv", "").split("_")

        # 預期命名格式：
        # cycle_<週期>_<感測器>_<圈數>[_(編號)]
        # 例如：cycle_026_5_5.csv 或 cycle_026_5_5_1.csv
        if len(parts) >= 4:
            ring = parts[3]  # 第3個欄位 = 圈數
            if ring == TARGET_RING:
                full_path = os.path.join(root, file)
                try:
                    os.remove(full_path)
                    deleted_files.append(full_path)
                except Exception as e:
                    print(f"刪除失敗：{full_path} ({e})")

# === 結果輸出 ===
print(f"✅ 共刪除 {len(deleted_files)} 個檔案：")
for f in deleted_files:
    print(" -", f)
