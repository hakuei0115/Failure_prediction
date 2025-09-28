import os
import re
import glob
import pandas as pd
from modules.detect_cycle import CycleDetector
from config.constants import HIGH_ON, LOW_END, MODE_MAP

# ============= 公用：時間欄轉換 =============
def _to_datetime(val):
    try:
        ts = pd.to_datetime(val, errors="coerce")
        if pd.notna(ts):
            return ts
    except Exception:
        pass
    try:
        f = pd.to_numeric(val, errors="coerce")
        if pd.isna(f):
            return None
        if f > 1e15:   return pd.to_datetime(f, unit="ns", errors="coerce")
        elif f > 1e14: return pd.to_datetime(f, unit="us", errors="coerce")
        elif f > 1e11: return pd.to_datetime(f, unit="ms", errors="coerce")
        elif f > 1e9:  return pd.to_datetime(f, unit="s",  errors="coerce")
        else:          return pd.to_datetime(f, unit="s",  errors="coerce")
    except Exception:
        return None

# ============= 中文數字轉整數（支援到 99，足夠這裡使用） =============
_CN_DIGIT = {"零":0,"〇":0,"一":1,"二":2,"兩":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
def cn_num_to_int(s: str) -> int:
    s = s.strip()
    if not s:
        return 0
    # 純數字直接回傳
    if re.fullmatch(r"\d+", s):
        return int(s)

    # 只是一位（例：五、七）
    if s in _CN_DIGIT:
        return _CN_DIGIT[s]

    # 十、十一、二十、二十五...
    if "十" in s:
        left, _, right = s.partition("十")
        tens = _CN_DIGIT.get(left, 1) if left else 1
        ones = _CN_DIGIT.get(right, 0) if right else 0
        return tens * 10 + ones

    # 其他情況（很少用到）
    return 0

# ============= 檔名→洩漏資訊解析（加入中文圈數與資料夾備援） =============
def parse_leak_info(filename: str, folder_status: str = None):
    """
    例：
    '第一根十圈.csv'   -> ([1], 10)
    '第一二根七圈.csv' -> ([1,2], 7)
    '正常.csv'         -> ([], 0)

    若檔名抓不到圈數，回退用 folder_status（例如 '10圈'）。
    """
    # 感測器（中文）
    round_index = 0
    leak_sensors = []
    m = re.search(r"第([一二三四五六]+)根", filename)
    
    mround = re.search(r"_(\d+)\.csv$", filename)
    if mround:
        round_index = int(mround.group(1))
        
    if m:
        mapping = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6}
        for ch in m.group(1):
            if ch in mapping:
                leak_sensors.append(mapping[ch])
        leak_sensors = sorted(set(leak_sensors))

    # 圈數：先抓阿拉伯數字，其次抓中文，最後回退資料夾
    circle = 0
    md = re.search(r"(\d+)\s*圈", filename)
    if md:
        circle = int(md.group(1))
    else:
        mcn = re.search(r"([零〇一二兩三四五六七八九十]+)\s*圈", filename)
        if mcn:
            circle = cn_num_to_int(mcn.group(1))

    # 回退：用資料夾 '5圈'/'7圈'/'10圈'
    if circle == 0 and folder_status and folder_status != "正常":
        ms = re.search(r"(\d+)", folder_status)
        if ms:
            circle = int(ms.group(1))

    return leak_sensors, circle, round_index

# ============= 檔名編號：避免覆寫 =============
_counters = {}  # key=(out_dir, suffix) -> next_id:int

def _scan_existing_start_id(out_dir: str, suffix: str, pure_normal: bool) -> int:
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "cycle_*.csv") if pure_normal else os.path.join(out_dir, f"cycle_*{suffix}.csv")
    mx = 0
    for p in glob.glob(pattern):
        b = os.path.basename(p)
        m = re.match(r"cycle_(\d{3})", b)
        if m:
            try:
                mx = max(mx, int(m.group(1)))
            except:
                pass
    return mx + 1

def next_cycle_id(out_dir: str, suffix: str, pure_normal: bool) -> int:
    key = (out_dir, suffix if not pure_normal else "")
    if key not in _counters:
        _counters[key] = _scan_existing_start_id(out_dir, suffix, pure_normal)
    val = _counters[key]
    _counters[key] += 1
    return val

# ============= 核心：單一感測器切週期並輸出 =============
def detect_and_save_cycles(df: pd.DataFrame, sensor_id: int, out_dir: str, pure_normal_source: bool, leak_suffix: str):
    """
    pure_normal_source:
      True  -> 來源在 row_data/.../正常/，命名 cycle_{id:03d}.csv
      False -> 來源在 row_data/.../(5|7|10)圈/，命名 cycle_{id:03d}{leak_suffix}.csv
               （同檔的「正常感測器」也要帶 suffix）
    """
    key = f"psr_val_{sensor_id-1}"
    detector = CycleDetector(
        low_th=LOW_END,
        high_th=HIGH_ON,
        sensor_key=key,
        mode=MODE_MAP[key],
        fixed_duration_sec=10.0
    )

    os.makedirs(out_dir, exist_ok=True)

    for _, row in df.iterrows():
        # 構建單筆紀錄（缺值直接略過）
        try:
            record = {
                "si_ts": _to_datetime(row["si_ts"]),
                "psr_val_0": float(row["psr_val_0"]),
                "psr_val_1": float(row["psr_val_1"]),
                "psr_val_2": float(row["psr_val_2"]),
                "psr_val_3": float(row["psr_val_3"]),
                "psr_val_4": float(row["psr_val_4"]),
                "psr_val_5": float(row["psr_val_5"]),
            }
        except Exception:
            continue

        result = detector.update(record)
        if result is None or detector.last_cycle_valid is False:
            continue

        # 取號 & 儲存
        cid = next_cycle_id(out_dir, leak_suffix, pure_normal_source)
        if pure_normal_source:
            fname = f"cycle_{cid:03d}.csv"
        else:
            fname = f"cycle_{cid:03d}{leak_suffix}.csv"
        out_path = os.path.join(out_dir, fname)
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ {out_path}  ({len(result)} rows)")

# ============= 主流程：遍歷 row_data =============
def process_all_data(base_dir="row_data", out_base="train"):
    """
    結構：
    row_data/
      ├─ 一根/
      │   ├─ 正常/
      │   ├─ 5圈/
      │   ├─ 7圈/
      │   └─ 10圈/
      └─ 兩根/
          ├─ 正常/
          ├─ 5圈/
          ├─ 7圈/
          └─ 10圈/
    """
    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.lower().endswith(".csv"):
                continue

            file_path = os.path.join(root, file)
            rel_dir = os.path.relpath(root, base_dir)  # e.g. '一根/7圈'
            parts = rel_dir.split(os.sep)
            if len(parts) < 2:
                print(f"⚠️ 路徑層級異常，略過：{file_path}")
                continue

            group = parts[0]   # '一根' 或 '兩根'
            status = parts[1]  # '正常' 或 '5圈'/'7圈'/'10圈'

            # 載入資料
            try:
                df = pd.read_csv(file_path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="utf-8")
            except Exception as e:
                print(f"⚠️ 讀檔失敗：{file_path} ({e})")
                continue

            # 是否為純正常來源
            pure_normal_source = (status == "正常")

            # 解析洩漏資訊（加入 folder_status 傳入，以便回退）
            leak_sensors, leak_circle, round_index = parse_leak_info(file, folder_status=status)

            # 非正常來源但解析不到 → 再保險一次用 folder_status 的數字
            if not pure_normal_source and (leak_circle == 0 or not leak_sensors):
                # 感測器沒解析到，至少圈數會從資料夾層級取得；
                # 但若真的檔名沒有「第..根」，就無法知道是哪幾根洩漏 → 這裡直接用錯誤提示
                if not leak_sensors:
                    print(f"⚠️ 非正常來源但檔名缺少『第…根』資訊（無法辨識洩漏根），跳過：{file_path}")
                    continue
                if leak_circle == 0:
                    print(f"⚠️ 非正常來源但圈數解析失敗（檔名/資料夾皆無法辨識），跳過：{file_path}")
                    continue

            # 構造 suffix
            leak_suffix = ""
            if not pure_normal_source:
                leak_str = "".join(map(str, sorted(leak_sensors)))
                if round_index > 0:
                    leak_suffix = f"_{leak_str}_{leak_circle}_{round_index}"
                else:
                    leak_suffix = f"_{leak_str}_{leak_circle}"


            # 逐一感測器輸出
            for sensor_id in range(1, 7):
                if pure_normal_source:
                    out_dir = os.path.join(out_base, group, "正常", f"sensor{sensor_id}")
                    detect_and_save_cycles(df, sensor_id, out_dir, True, "")
                else:
                    # 洩漏來源：洩漏→圈數資料夾；其餘正常→正常資料夾，但檔名帶 suffix
                    if sensor_id in leak_sensors:
                        out_dir = os.path.join(out_base, group, f"{leak_circle}圈", f"sensor{sensor_id}")
                    else:
                        out_dir = os.path.join(out_base, group, "正常", f"sensor{sensor_id}")
                    detect_and_save_cycles(df, sensor_id, out_dir, False, leak_suffix)

if __name__ == "__main__":
    process_all_data()
