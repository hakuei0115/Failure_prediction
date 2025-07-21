import pandas as pd

# === 1. 載入原始 Excel 檔 ===
input_path = "data/一根/全開/第一根.xlsx"
df = pd.read_excel(input_path)

# === 2. 確保 timestamp 格式正確並排序（不改欄位名）===
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values(by='timestamp').reset_index(drop=True)

# === 3. 找出壓力欄位，進行補值 ===
sensor_cols = [col for col in df.columns if col.startswith("pressure")]

for col in sensor_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')     # 轉換為 float
    df[col] = df[col].replace(-1, pd.NA)                  # -1 當作缺失值
    df[col] = df[col].interpolate(method='linear', limit_direction='both')
    df[col] = df[col].bfill().ffill()
    df[col] = df[col].infer_objects(copy=False).round()   # 消除警告，轉 float → int（若可）

# === 4. 儲存為新檔案 ===
output_path = "第一根(全開)_補值後.xlsx"
df.to_excel(output_path, index=False, engine='openpyxl')
print(f"✅ 已補值並儲存為：{output_path}")
