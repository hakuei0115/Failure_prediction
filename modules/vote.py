import numpy as np
import pandas as pd
from collections import deque, defaultdict

class ProbVoteBuffer:
    """單一感測器：保留最近 window_sec 內的機率分佈並做加總。"""
    def __init__(self, window_sec=30.0, maxlen=None):
        self.window_sec = float(window_sec)
        self.maxlen = maxlen
        self.q = deque()  # (ts: Timestamp, prob_map: {0/5/10 -> prob})

    def add(self, ts, prob_map, fallback_label = None):
        if not prob_map:
            prob_map = {}
            if fallback_label is not None:
                prob_map[int(fallback_label)] = 1.0  # one-hot 後備
        clean = {}
        for k in [0, 1, 2]:
            v = float(prob_map.get(k, 0.0))
            if np.isfinite(v) and v >= 0.0:
                clean[k] = v
        self.q.append((ts, clean))
        if self.maxlen is not None:
            while len(self.q) > self.maxlen:
                self.q.popleft()
        self._trim(ts)

    def _trim(self, now: pd.Timestamp):
        cutoff = now - pd.Timedelta(seconds=self.window_sec)
        while self.q and self.q[0][0] < cutoff:
            self.q.popleft()

    def counts(self) -> dict:
        c = defaultdict(float)
        for _, pm in self.q:
            for k, p in pm.items():
                c[k] += p
        return dict(c)

    def leak_score(self) -> float:
        c = self.counts()
        return float(c.get(1, 0.0) + c.get(2, 0.0))

    def samples(self) -> int:
        return len(self.q)

class MultiLeakArbiter:
    def __init__(self, sensors, batch_sec=300, min_votes=2, leak_threshold=1.2, drop_threshold=0.8, max_winners=2):
        self.buffers = {s: [] for s in sensors}   # 改成 list 累積
        self.batch_sec = batch_sec
        self.min_votes = int(min_votes)
        self.leak_threshold = float(leak_threshold)
        self.drop_threshold = float(drop_threshold)
        self.max_winners = int(max_winners)
        self.batch_start_ts = None  # 批次起始時間

    def update(self, sensor_name: str, ts, pred_label, prob_map):
        if self.batch_start_ts is None:
            self.batch_start_ts = ts

        # 收集一筆
        self.buffers[sensor_name].append((ts, prob_map or {pred_label: 1.0}))

        # 判斷是否到達批次結算時間
        if (ts - self.batch_start_ts).total_seconds() < self.batch_sec:
            return [], {}  # 還沒到時間，不輸出

        # ===== 投票邏輯 =====
        scores, samples = {}, {}
        for s, data in self.buffers.items():
            counts = defaultdict(float)
            for _, pm in data:
                for k, p in pm.items():
                    counts[k] += p
            scores[s] = counts.get(1, 0.0) + counts.get(2, 0.0)
            samples[s] = len(data)

        candidates = [
            s for s in self.buffers.keys()
            if samples[s] >= self.min_votes and scores[s] >= self.leak_threshold
        ]
        
        pool_sorted = sorted(candidates, key=lambda s: scores[s], reverse=True)
        winners = set(pool_sorted[:self.max_winners])

        details = {"scores": scores, "samples": samples, "candidates": candidates, "winners": list(winners)}

        # ===== 重設，開始下一個批次 =====
        self.buffers = {s: [] for s in self.buffers.keys()}
        self.batch_start_ts = ts

        return list(winners), details
