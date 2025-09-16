import os
import numpy as np
from scipy.spatial.distance import euclidean

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
path_1 = f"{now_dir}/cycle_binary_output/cycle_001_binary_string.txt"
path_2 = f"{now_dir}/cycle_binary_output/cycle_001_3_10_binary_string.txt"

with open(path_1, "r") as f1, open(path_2, "r") as f2:
    bstr1 = f1.read().strip()
    bstr2 = f2.read().strip()

seq1 = bitstring_to_sequence(bstr1)
seq2 = bitstring_to_sequence(bstr2)

# 計算 DTW 距離
dtw_dist = dtw_distance(seq1, seq2)

print(len(seq1), len(seq2))
print("DTW 距離：", dtw_dist)