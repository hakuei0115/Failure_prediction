import pandas as pd
import os

now_dir = os.path.dirname(os.path.abspath(__file__))

# === 參數設定 ===
INPUT_PATH = f"{now_dir}/cycle_ma_clean/normal_D.csv"
VAL_COL = "psr_val_0_ma"
TS_COL = "si_ts"
BIN_COL = "binary_2bit"
OUTPUT_DIR = f"{now_dir}/cycle_binary_output"

STRING_FILE = "normal_D_binary_string.txt"

# === 建立資料夾 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 離散化函式 ===
def to_binary(x: float) -> str:
    if x < 0.2:
        return "00"
    elif x < 0.3:
        return "01"
    elif x < 0.4:
        return "10"
    else:
        return "11"

# === 資料處理 ===
df = pd.read_csv(INPUT_PATH)

# 檢查欄位是否存在
if VAL_COL not in df.columns or TS_COL not in df.columns:
    raise ValueError(f"欄位 {VAL_COL} 或 {TS_COL} 不存在於檔案中")

# 加入離散化欄位
df[BIN_COL] = df[VAL_COL].apply(to_binary)

# 將所有二位元拼成一長字串
binary_string = ''.join(df[BIN_COL].tolist())

# === 輸出字串 ===
with open(os.path.join(OUTPUT_DIR, STRING_FILE), "w") as f:
    f.write(binary_string)

print("✅ 離散化完成！")
print(f"🧾 二位元字串：{os.path.join(OUTPUT_DIR, STRING_FILE)}")
