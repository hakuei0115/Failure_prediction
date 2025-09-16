# 🧠 Sensor Cycle Analysis & DTW Matching

本專案針對壓力感測器資料進行：

✅ 完整週期偵測  
✅ 滑動平均處理  
✅ 二位元離散化（模擬 A/D 轉換）  
✅ DTW 動態時間比對驗證樣本相似性

---

## 📂 專案檔案結構
├── reshape_cycles.py # 從原始感測資料擷取完整週期（0→高→0）
├── cycles_ma.py # 對每段週期做滑動平均
├── discretize_to_binary.py # 將平滑資料離散化為二位元並拼接成字串
├── dtw.py # 使用原生 DTW 比對兩組週期樣本相似度
├── row_data/ # 原始感測器 Excel 檔
├── cycles_out_simple1/ # 擷取出的每段週期（CSV 格式）
├── cycle_ma_clean/ # 每段週期滑動平均結果
├── cycle_binary_output/ # 二位元字串結果與拼接字串

---

## 🧩 腳本說明

### 1️⃣ `reshape_cycles.py` - 擷取完整週期

- 根據「上升門檻 → 高壓平台 → 下降門檻」切出一段週期  
- 僅當訊號回到低壓 (< 0.05) 時才視為完整波型結束

```python
HIGH_ON  = 0.4     # 上升沿門檻
HIGH_OFF = 0.24    # 下降沿門檻
LOW_END  = 0.05    # 判定結束為低壓
```

🔹 輸出：
- cycle_001.csv, cycle_002.csv…
- index.csv（週期起迄資訊）

### 2️⃣ `cycles_ma.py` - 每步重算的滑動平均

- 對單一週期 CSV 進行滑動平均處理
- 僅保留平均後欄位與對應時間

```python
k = 5  # 每 5 筆取平均，會重疊（sliding）
```

🔹輸出：
- cycle_ma_clean/cycle_001.csv

### 3️⃣ `discretize_to_binary.py` - 轉換為 2-bit 序列

- 將平滑後的壓力值依區間轉為 2-bit 離散值：

| 壓力值區間         | 2-bit 表示 |
| ------------- | -------- |
| x ≤ 0.2       | 00       |
| 0.2 < x ≤ 0.3 | 01       |
| 0.3 < x ≤ 0.4 | 10       |
| x > 0.4       | 11       |

🔹輸出：
- cycle_001_discretized.csv：時間＋2bit值
- cycle_001_binary_string.txt：純字串，例如：111110111011...

### 4️⃣ `dtw.py` - 使用原生 DTW 比對二位元序列

- 將每組 binary string 轉為整數序列（00→0, 11→3）
- 比對兩週期樣本的對齊距離

```python=
DTW Distance: 1.0  # 越小表示越相似
```

## ✅ 執行流程建議

```python=
# Step 1: 從原始資料切出完整週期
python reshape_cycles.py

# Step 2: 對每段週期進行滑動平均
python cycles_ma.py

# Step 3: 離散化為二位元字串
python discretize_to_binary.py

# Step 4: 使用 DTW 比對不同週期樣本
python dtw.py
```