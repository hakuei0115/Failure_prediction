import pandas as pd
import os

now_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = f"{now_dir}/cycle_ma_clean"

input_path = "row_data/第一根五圈有流量計-test.xlsx"
df = pd.read_excel(input_path)

# 設定滑動平均參數
k = 5
val_col = "psr_val_0"
ts_col = "si_ts"
ma_col = f"{val_col}_ma"

# 確認欄位存在
if val_col not in df.columns or ts_col not in df.columns:
    raise ValueError(f"缺少必要欄位：{val_col} 或 {ts_col}")

# 嚴格滑動平均（每個完整視窗都取平均）
df_ma = df[val_col].rolling(window=k, min_periods=k).mean().reset_index(drop=True)

# 移除前 k-1 筆，並對齊時間戳
df_out = pd.DataFrame({
    ts_col: df[ts_col].iloc[k-1:].reset_index(drop=True),
    ma_col: df_ma.dropna().reset_index(drop=True)
})

# 驗證長度是否為 N - (k - 1)
original_len = len(df)
expected_len = original_len - (k - 1)
actual_len = len(df_out)
assert actual_len == expected_len, f"處理後長度 {actual_len} 不等於預期長度 {expected_len}"

# 儲存處理後的結果
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "normal_D.csv")
df_out.to_csv(output_path, index=False, encoding="utf-8-sig")

# 顯示前幾筆處理結果
df_out.head()
