import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 使用微軟正黑體顯示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# 方法一：條件期望剩餘壽命
def conditional_expected_rul(mu, sigma, N):
    ZN = (N - mu) / sigma
    phi = norm.pdf(ZN)
    Phi = norm.cdf(ZN)
    expected_X_given_X_gt_N = mu + sigma * (1 - Phi) / phi
    return expected_X_given_X_gt_N - N

# 方法二：未來 k 次的失效機率
def near_term_failure_probability(mu, sigma, N, k):
    ZN = (N - mu) / sigma
    ZNk = (N + k - mu) / sigma
    Phi_ZN = norm.cdf(ZN)
    Phi_ZNk = norm.cdf(ZNk)
    return (Phi_ZNk - Phi_ZN) / (1 - Phi_ZN)

# 參數設定
mu = 7000000
sigma = 70000
# sigma = [30000, 70000, 100000]
# k = [300, 3000, 30000]
k = 30000
P_thresh = 0.05

# N 值序列
N_values = np.arange(6800000, 7000001, 10000) # 從6800000到7000000，每10000一個點 方法一
# rul_values_sigma1 = []
# rul_values_sigma2 = []
# rul_values_sigma3 = []
# risk_values = []
# risk_values_k1 = []
# risk_values_k2 = []
# risk_values_k3 = []
replacement_thresholds = []

# 計算第一方法的值
# for N in N_values:
#     for s in sigma:
#         rul = conditional_expected_rul(mu, s, N)
#         rul_log = np.log(rul) if rul > 0 else np.nan
#         if s == 30000:
#             rul_values_sigma1.append(rul_log)
#         elif s == 70000:
#             rul_values_sigma2.append(rul_log)
#         elif s == 100000:
#             rul_values_sigma3.append(rul_log)

# 計算第二方法的值
# for N in N_values:
#     for k_value in k:
#         risk = near_term_failure_probability(mu, sigma[1], N, k_value)
#         if k_value == 300:
#             risk_values_k1.append(risk)
#         elif k_value == 3000:
#             risk_values_k2.append(risk)
#         elif k_value == 30000:
#             risk_values_k3.append(risk)

# # 方法三的輸出：單一點
replacement_point = find_preventive_replacement_point(mu, sigma, k, P_thresh)

# # 畫三張圖
# fig1, ax1 = plt.subplots(figsize=(8, 5))
# ax1.plot(N_values, rul_values_sigma1, color='blue', label='預估 (σ=30,000)')
# ax1.plot(N_values, rul_values_sigma2, color='green', label='預估 (σ=70,000)')
# ax1.plot(N_values, rul_values_sigma3, color='red', label='預估 (σ=100,000)')
# ax1.set_title("方法一: 剩餘壽命 (RUL) 預估")
# ax1.set_xlabel("電磁閥累積起閉次數 (N)")
# ax1.set_ylabel("期望剩餘壽命")
# ax1.grid(True)
# ax1.legend()

# fig2, ax2 = plt.subplots(figsize=(8, 5))
# ax2.plot(N_values, risk_values_k1, color='blue', label='k=300')
# ax2.plot(N_values, risk_values_k2, color='green', label='k=3000')
# ax2.plot(N_values, risk_values_k3, color='red', label='k=30000')
# ax2.set_title("方法二: 未來 k 次操作中故障機率")
# ax2.set_xlabel("電磁閥累積起閉次數 (N)")
# ax2.set_ylabel("故障機率")
# ax2.grid(True)
# ax2.legend()

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.axvline(replacement_point, color='red', linestyle='--', label='Preventive Replacement Point')
ax3.set_title(f"Method 3: Preventive Replacement Threshold\n(at Risk ≤ {P_thresh:.0%})")
ax3.set_xlabel("Usage (N)")
ax3.set_ylabel("Marker Only (Vertical Line)")
ax3.set_xlim(min(N_values), max(N_values))
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
