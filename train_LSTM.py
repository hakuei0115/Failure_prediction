import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# ========== [1] 載入資料 ==========
data = np.load("train_dataset.npz")
X, y = data["X"], data["y"]

# ========== [2] 類別編碼 ==========
leak_levels = [0.0, 2.78, 27.78, 55.56, 100.0]
label_encoder = {v: i for i, v in enumerate(leak_levels)}
y_encoded = np.array([[label_encoder[val] for val in row] for row in y])  # shape: (2609, 6)

# ========== [3] 資料切割 ==========
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ========== [4] One-hot 編碼 ==========
def one_hot_all(y_data):
    return [to_categorical(y_data[:, i], num_classes=len(leak_levels)) for i in range(6)]

y_train_split = one_hot_all(y_train)
y_val_split = one_hot_all(y_val)
y_test_split = one_hot_all(y_test)

# ========== [5] 模型建構 ==========
input_layer = Input(shape=(X.shape[1], X.shape[2]))
x = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(64))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
outputs = [Dense(5, activation="softmax", name=f"sensor{i+1}_output")(x) for i in range(6)]

model = Model(inputs=input_layer, outputs=outputs)
model.compile(
    optimizer="adam",
    loss=["categorical_crossentropy"] * 6,
    metrics=["accuracy"]
)

# ========== [6] 訓練 ==========
model.fit(
    X_train, y_train_split,
    validation_data=(X_val, y_val_split),
    epochs=200,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
)

# ========== [7] 儲存模型 ==========
# model.save("lstm_sensor_multiclass_model.keras")

# ========== [8] 預測與評估 ==========
y_pred = model.predict(X_test)
y_pred_label = [np.argmax(p, axis=1) for p in y_pred]
y_true_label = [np.argmax(t, axis=1) for t in y_test_split]  # ✅ 修正：y_val → y_test_split

# ========== [9] 輸出分類報告 ==========
for i in range(6):
    print(f"\nSensor {i+1} 分類報告:")
    print(classification_report(
        y_true_label[i], y_pred_label[i],
        target_names=[str(lv) for lv in leak_levels]
    ))