import pandas as pd
from config.constants import TIME_COL

class CycleDetector:
    def __init__(self, low_th=0.1, high_th=0.2, sensor_key="psr_val_0", mode="single", on_timeout=None, fixed_duration_sec=11.0):
        assert mode in ("single", "double")
        self.low_th = low_th
        self.high_th = high_th
        self.key = sensor_key
        self.mode = mode

        # 即時超時告警（沿用）
        self.on_timeout = on_timeout

        # ---- ramp 追蹤（新版）----
        self.in_low = False
        self.last_low_rec = None   # 低壓區最後一筆（<low_th）
        self.ramp_start_rec = None # 離開低壓後，用 last_low_rec 作為起點

        # 一般模式所需
        self.state = "IDLE"
        self.cycle_buffer = []
        self.cycle_start_ts = None

        # 固定時長模式
        self.fixed_duration_sec = fixed_duration_sec
        self.fixed_deadline_ts = None
        self.pre_buffer = []   # 已離開低壓到升破 high_th 之間的點（含 ramp_start_rec）
        self.fixed_state = "IDLE"  # IDLE → ARMED(離低壓等待升破) → ACTIVE(收集到截止) → WAIT_LOW
        
        # === 每窗 pattern 與驗證結果 ===
        self.pattern_flags = {"up1": False, "down": False, "up2": False}
        self.last_cycle_valid = None
        self.last_cycle_reason = None

    def _val(self, record): 
        return float(record[self.key])

    def _track_ramp(self, record, val):
        """維護 last_low_rec / ramp_start_rec 與 pre_buffer。"""
        if val < self.low_th:
            # 還在低壓：更新最後一筆低壓點，清 ramp 與 pre_buffer
            self.in_low = True
            self.last_low_rec = record
            self.ramp_start_rec = None
            self.pre_buffer = []
        else:
            if self.in_low:
                # 剛離開低壓：起點=最後一筆低壓點
                self.ramp_start_rec = self.last_low_rec if self.last_low_rec is not None else record
                self.pre_buffer = [self.ramp_start_rec]
                self.in_low = False
            # 若已不在低壓且尚未升破 high_th，將點先放進 pre_buffer（為了不漏掉上升段）
            if self.ramp_start_rec is not None and val < self.high_th:
                self.pre_buffer.append(record)

    def _maybe_timeout_alert(self, record):
        if self.max_cycle_sec is None or self.cycle_start_ts is None or self.timeout_alerted:
            return
        now_ts = record.get(TIME_COL)
        if pd.isna(now_ts) or pd.isna(self.cycle_start_ts):
            return
        duration = (now_ts - self.cycle_start_ts).total_seconds()
        if duration > self.max_cycle_sec:
            self.timeout_alerted = True
            if callable(self.on_timeout):
                self.on_timeout(self.key, duration, self.cycle_start_ts, now_ts)
            else:
                print(f"⚠️ 即時告警：{self.key} 目前週期已 {duration:.2f}s (> {self.max_cycle_sec}s)")

    # ================= 固定時長模式 =================
    def _update_fixed(self, record, val):
        ts = record.get(TIME_COL)

        if self.fixed_state == "IDLE":
            if (self.ramp_start_rec is not None) and (val >= self.high_th):
                self.cycle_buffer = list(self.pre_buffer) + [record]
                self.cycle_start_ts = self.ramp_start_rec.get(TIME_COL) if self.ramp_start_rec else ts
                self.fixed_deadline_ts = self.cycle_start_ts + pd.Timedelta(seconds=self.fixed_duration_sec)
                self.fixed_state = "ACTIVE"

                # reset 本窗的狀態紀錄
                self.pattern_flags = {"up1": False, "down": False}
                self.last_cycle_valid = None
                self.last_cycle_reason = None
            return None

        if self.fixed_state == "ACTIVE":
            if ts <= self.fixed_deadline_ts:
                self.cycle_buffer.append(record)

                # ===== 在 11s 窗內更新狀態 =====
                # up1：第一次進入高壓
                if (not self.pattern_flags["up1"]) and (val >= self.high_th):
                    self.pattern_flags["up1"] = True

                # down：up1 之後回到低壓
                if self.pattern_flags["up1"] and (not self.pattern_flags["down"]) and (val <= self.low_th):
                    self.pattern_flags["down"] = True

                return None

            # === 超過截止：產出本窗，並驗證狀態 ===
            out = pd.DataFrame([r for r in self.cycle_buffer if r.get(TIME_COL) <= self.fixed_deadline_ts])

            pf = self.pattern_flags
            if not pf["up1"]:
                self.last_cycle_valid = False
                self.last_cycle_reason = "no_high_within_window"   # 低壓沒到高壓
            elif pf["up1"] and not pf["down"]:
                self.last_cycle_valid = False
                self.last_cycle_reason = "no_low_after_high"       # 高壓後沒降下來
            else:
                self.last_cycle_valid = True
                self.last_cycle_reason = "ok"

            if not self.last_cycle_valid:
                print(f"⚠️ Cycle 未完成（{self.last_cycle_reason}）")

            # 切段後 → 進入 WAIT_LOW
            self.fixed_state = "WAIT_LOW"
            self.cycle_buffer = []
            self.cycle_start_ts = None
            self.fixed_deadline_ts = None
            self.ramp_start_rec = None
            self.pre_buffer = []
            return out

        if self.fixed_state == "WAIT_LOW":
            if val < self.low_th:
                self.fixed_state = "IDLE"
            return None

        return None

    # ================= 原本的 single/double 模式 =================
    def _start_cycle_from(self, rec):
        self.cycle_buffer = [rec] if rec is not None else []
        ts0 = rec.get(TIME_COL) if rec else None
        self.cycle_start_ts = ts0 if pd.notna(ts0) else None
        self.timeout_alerted = False

    def _end_cycle_reset(self):
        self.cycle_start_ts = None
        self.timeout_alerted = False

    def _update_single(self, record, val):
        if self.state == "IDLE":
            if (self.ramp_start_rec is not None) and (val >= self.high_th):
                self._start_cycle_from(self.ramp_start_rec)
                self.state = "FIRST_HIGH"
            return None
        if self.state == "FIRST_HIGH":
            self.cycle_buffer.append(record)
            if val < self.low_th:
                self.state = "WAIT_SECOND_RISE"
            return None
        if self.state == "WAIT_SECOND_RISE":
            if val >= self.high_th:
                out = pd.DataFrame(self.cycle_buffer[:-1]) if self.cycle_buffer else None
                self.state = "FIRST_HIGH"
                if self.ramp_start_rec is not None:
                    self._start_cycle_from(self.ramp_start_rec)
                else:
                    self.cycle_buffer = []
                    self._end_cycle_reset()
                return out
            self.cycle_buffer.append(record)
            return None
        return None

    def _update_double(self, record, val):
        if self.state == "IDLE":
            if (self.ramp_start_rec is not None) and (val >= self.high_th):
                self._start_cycle_from(self.ramp_start_rec)
                self.state = "HIGH1"
            return None
        if self.state == "HIGH1":
            self.cycle_buffer.append(record); 
            if val < self.low_th: self.state = "LOW1"
            return None
        if self.state == "LOW1":
            self.cycle_buffer.append(record); 
            if val >= self.high_th: self.state = "HIGH2"
            return None
        if self.state == "HIGH2":
            self.cycle_buffer.append(record); 
            if val < self.low_th: self.state = "LOW2"
            return None
        if self.state == "LOW2":
            self.cycle_buffer.append(record)
            if val >= self.high_th:
                out = pd.DataFrame(self.cycle_buffer[:-1]) if self.cycle_buffer else None
                self.state = "HIGH1"
                if self.ramp_start_rec is not None:
                    self._start_cycle_from(self.ramp_start_rec)
                else:
                    self.cycle_buffer = []
                    self._end_cycle_reset()
                return out
            return None
        return None

    # ================= 公用入口 =================
    def update(self, record):
        val = self._val(record)
        self._track_ramp(record, val)
        if self.fixed_duration_sec and self.fixed_duration_sec > 0:
            return self._update_fixed(record, val)
        # 否則走原本 single/double 規則
        out = self._update_single(record, val) if self.mode == "single" else self._update_double(record, val)
        # 在一般模式下也可做即時超時檢查
        if self.state in ("FIRST_HIGH", "WAIT_SECOND_RISE", "HIGH1", "LOW1", "HIGH2", "LOW2"):
            self._maybe_timeout_alert(record)
        return out