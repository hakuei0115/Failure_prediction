import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ========== [1] 載入資料 ========== #
df = pd.read_csv("extracted_sensor_features.csv")

# 提取 label 與特徵欄位
label_cols = [f"label_sensor{i}" for i in range(1, 7)]
feature_cols = [col for col in df.columns if col not in label_cols]

X = df[feature_cols].values

# 將洩漏百分比轉為分類 index
leak_levels = [0.0, 2.78, 27.78, 55.56, 100.0]
label_encoder = {v: i for i, v in enumerate(leak_levels)}
label_decoder = {i: v for v, i in label_encoder.items()}
Y = df[label_cols].applymap(lambda x: label_encoder.get(x, 0)).values

# ========== [2] 模型訓練與評估 ========== #
for i in range(6):
    y = Y[:, i]
    print(f"\n===== Sensor {i+1} 分類報告 =====")
    
    # 資料切分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 模型建立
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 預測與評估
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    # 特徵重要性圖
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10

    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances[indices][:top_n], y=np.array(feature_cols)[indices][:top_n])
    plt.title(f"Sensor {i+1} - Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # ===== Normalized 混淆矩陣（每列為 100%）=====
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(leak_levels))), normalize='true') * 100

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[str(lv) for lv in leak_levels],
                yticklabels=[str(lv) for lv in leak_levels])
    plt.title(f"Sensor {i+1} - Confusion Matrix (%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()
