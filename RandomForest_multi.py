import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ===== 參數 =====
DATA_FILE = "features_output/features_merged.csv"
MODEL_FILE = "models/rf_multioutput.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    # 讀資料
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    
    # 取出 label 欄位（六個感測器）
    label_cols = [c for c in df.columns if c.startswith("label_")]
    X = df.drop(columns=["file"] + label_cols)
    y = df[label_cols]

    # 分割訓練 / 測試
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 建立模型
    base_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model = MultiOutputClassifier(base_rf)

    # 訓練
    print("訓練中...")
    model.fit(X_train, y_train)

    # 預測
    y_pred = model.predict(X_test)

    # 評估
    print("\n===== 評估結果 =====")
    for i, col in enumerate(label_cols):
        print(f"\n[{col}]")
        print(classification_report(y_test.iloc[:, i], y_pred[:, i], digits=4))

    # 逐根感測器 accuracy
    for i, col in enumerate(label_cols):
        acc_i = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
        print(f"{col} 準確率: {acc_i:.4f}")

    # 整體 sample-level accuracy（六根都正確才算正確）
    sample_acc = (y_test.values == y_pred).all(axis=1).mean()
    print(f"\n整體樣本準確率: {sample_acc:.4f}")


    # 存模型
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"[✓] 模型已儲存：{MODEL_FILE}")

if __name__ == "__main__":
    main()
