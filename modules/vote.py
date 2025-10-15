import pandas as pd
from collections import defaultdict, Counter
import threading
import time

class MultiLeakArbiter:
    """
    多感測器洩漏仲裁器（長週期統計版，無輸出）
    -------------------------------------------------
    功能：
      - 在 batch_sec 內累積各感測器洩漏紀錄（pred_label != 0）
      - 每次結算只回傳：
          {
            "top1": {"sensor": <感測器>, "class": <洩漏類別>},
            "top2": {"sensor": <感測器>, "class": <洩漏類別>}  # 可選
          }
      - 不印出、不告警、不紀錄
    """

    def __init__(self, sensors, batch_sec=300, on_batch_result=None):
        self.sensors = sensors
        self.batch_sec = batch_sec
        self.buffers = {s: [] for s in sensors}
        self.lock = threading.Lock()
        self._running = False
        self._last_results = {}
        self.on_batch_result = on_batch_result  # 可選 callback

    # ===== 啟動與停止 =====
    def start(self):
        """啟動背景結算執行緒"""
        self._running = True
        threading.Thread(target=self._batch_loop, daemon=True).start()

    def stop(self):
        """停止背景結算"""
        self._running = False

    # ===== 新增資料 =====
    def update(self, sensor_name, ts, pred_label=None, prob_map=None):
        """
        加入一筆預測結果。
        - 只統計 pred_label = 1 或 2（洩漏類別）
        """
        with self.lock:
            label = pred_label
            # 若有 prob_map，取最大機率類別
            if prob_map and isinstance(prob_map, dict):
                label = max(prob_map, key=prob_map.get)
            if label != 0:
                self.buffers[sensor_name].append(label)

    # ===== 週期性結算 =====
    def _batch_loop(self):
        while self._running:
            time.sleep(self.batch_sec)
            results = self._finalize_batch()
            if self.on_batch_result:
                self.on_batch_result(results)

    # ===== 統計核心 =====
    def _finalize_batch(self):
        with self.lock:
            leak_counts = defaultdict(int)   # 感測器洩漏次數
            leak_modes = {}                  # 感測器主要洩漏類別

            for s, labels in self.buffers.items():
                if not labels:
                    continue
                leak_counts[s] = len(labels)
                leak_modes[s] = Counter(labels).most_common(1)[0][0]

            # 若無洩漏事件 → 清空並回傳 {}
            if not leak_counts:
                self._last_results = {}
                self.buffers = {s: [] for s in self.sensors}
                return {}

            # 排序：洩漏次數高到低
            sorted_sensors = sorted(leak_counts.items(), key=lambda x: x[1], reverse=True)
            top1_sensor, top1_count = sorted_sensors[0]
            top2_sensor, top2_count = (None, 0)
            if len(sorted_sensors) > 1:
                top2_sensor, top2_count = sorted_sensors[1]

            # 準備回傳結果
            results = {
                "top1": {
                    "sensor": top1_sensor,
                    "class": leak_modes[top1_sensor]
                }
            }
            if top2_sensor and top2_count >= top1_count / 2.0:
                results["top2"] = {
                    "sensor": top2_sensor,
                    "class": leak_modes[top2_sensor]
                }

            self._last_results = results
            self.buffers = {s: [] for s in self.sensors}
            return results

    # ===== 主動取結果 =====
    def fetch_results(self):
        """主程式可隨時取得最新一批結果"""
        return self._last_results