import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 訓練資料位置
DATA_DIR = "features_output"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 所有感測器
SENSOR_LIST = [f"sensor{i}" for i in range(1, 7)]

# 要使用的特徵欄位
# FEATURE_COLS = ["max", "min", "mean", "std", "slope", "stability", "holding_time"]
FEATURE_COLS = ["mean", "std", "holding_time", "range"]

for sensor in SENSOR_LIST:
    print(f"\n==== 🚀 訓練 {sensor} 的模型 ====")

    # 讀取資料
    csv_path = os.path.join(DATA_DIR, f"{sensor}_train.csv")
    df = pd.read_csv(csv_path)

    # 拆分資料集
    X = df[FEATURE_COLS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 建立並訓練模型
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 預測與評估
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[✓] 準確率：{acc:.4f}")
    print("[📊] 分類報告：")
    print(classification_report(y_test, y_pred))

    # 儲存模型
    model_path = os.path.join(OUTPUT_DIR, f"{sensor}_rf_model.pkl")
    joblib.dump(clf, model_path)
    print(f"[💾] 模型已儲存：{model_path}")
