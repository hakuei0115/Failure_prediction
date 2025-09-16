import os
import numpy as np
import pandas as pd

now_dir = os.path.dirname(os.path.abspath(__file__))

# 自訂 DTW 實作
def dtw_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])  # L1 distance
            dtw[i, j] = cost + min(
                dtw[i - 1, j],    # insertion
                dtw[i, j - 1],    # deletion
                dtw[i - 1, j - 1] # match
            )

    return dtw[n, m]

# 讀取 bitstring 並轉為序列
def bitstring_to_sequence(bitstring: str):
    mapping = {"00": 0, "01": 1, "10": 2, "11": 3}
    seq = [mapping[bitstring[i:i+2]] for i in range(0, len(bitstring), 2)]
    return seq

# 載入 bitstring
path_1 = f"{now_dir}/cycle_binary_output/normal_D_binary_string.txt"
path_2 = f"{now_dir}/cycle_binary_output/cycle_001_binary_string.txt"

# 載入 bitstring
with open(path_1, "r") as f1, open(path_2, "r") as f2:
    target_str = f1.read().strip()
    base_str = f2.read().strip()

base_seq = bitstring_to_sequence(base_str)
target_seq = bitstring_to_sequence(target_str)

# 滑動比對：重疊方式，每次移動 1 個單位
window_size = len(base_seq)
distances = []

for i in range(0, len(target_seq) - window_size + 1):
    sub_seq = target_seq[i:i + window_size]
    dist = dtw_distance(base_seq, sub_seq)
    distances.append(dist)

df = pd.DataFrame({
    "dtw_distance": distances
})

csv_path = f"{now_dir}/dtw_result.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"✅ DTW 結果已儲存：{csv_path}")