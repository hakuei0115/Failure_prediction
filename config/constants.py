# ===== 偵測週期參數 =====
HIGH_ON = 0.2 # 高於此值視為「高壓」
LOW_END = 0.1 # 低於此值視為「低壓」
PRESSURE_THRESHOLD = HIGH_ON

# ===== 資料庫欄位參數 =====
ID_COL = "si_id"
TIME_COL = "si_ts"

# ===== 特徵欄位參數 =====
FEATURE_COLS = ["mean", "std", "holding_time", "range"]

# ===== 模型參數 =====
# LABEL_MAP = {0: "✅ 正常", 1: "⚠️ 洩漏（7圈）", 2: "🚨 洩漏（10圈）"}
LABEL_MAP = {0: "洩漏等級0", 1: "洩漏等級1", 2: "洩漏等級2"}

POLICY_MAP = {
    0: "無需維修",
    1: "安排停機檢查與氣密測試",
    2: "緊急停機，立即維修並追蹤"
}

MODE_MAP = {
    "psr_val_0": "single",
    "psr_val_1": "double",  # 感測器二
    "psr_val_2": "single",
    "psr_val_3": "single",
    "psr_val_4": "single",
    "psr_val_5": "single",
}
SENSOR_NAME = {
    "psr_val_0": "sensor1",
    "psr_val_1": "sensor2",
    "psr_val_2": "sensor3",
    "psr_val_3": "sensor4",
    "psr_val_4": "sensor5",
    "psr_val_5": "sensor6",
}

# ===== 壽限估計參數 =====
MEAN_LIFE = 10_000_000
STD_LIFE = 1_000_000
Y = 5500000  # 希望再撐的次數