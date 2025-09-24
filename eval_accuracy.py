#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_accuracy.py  (v1.2)

新增：
- --pred-mode {auto,class,label}：
    auto  ：自動判斷 pred_label 是否為 0/1/2（class）或 0/7/10（label），必要時自動映射。
    class ：預期 pred_label 已是 0/1/2。
    label ：預期 pred_label 是 0/7/10，會映射為 0/1/2。
- 忽略常見回合資料夾（R1/R2/Round1/Round2），不會誤解析。
- 保留 --verbose 與 --zero-based 與診斷輸出。
"""
import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Iterable

import numpy as np
import pandas as pd


# ===== 類別定義：0->0圈, 1->7圈, 2->10圈 =====
CLASS_ORDER = [0, 7, 10]
LABEL_TO_CLASS = {0: 0, 7: 1, 10: 2}
CLASS_TO_LABEL = {v: k for k, v in LABEL_TO_CLASS.items()}

# ===== 中文數字對照 =====
CH_NUM = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

ROUND_NAMES = {"r1", "r2", "round1", "round2"}


def parse_chinese_int(s: str) -> Optional[int]:
    if s is None:
        return None
    s = s.strip()
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    if s in CH_NUM:
        return CH_NUM[s]
    if len(s) == 2 and s[0] == "十" and s[1] in CH_NUM:
        return 10 + CH_NUM[s[1]]
    if len(s) == 2 and s[1] == "十" and s[0] in CH_NUM and CH_NUM[s[0]] > 0:
        return CH_NUM[s[0]] * 10
    return None


def extract_sensor_indices_from_text(s: str) -> List[int]:
    s_nospace = re.sub(r"\s+", "", s)
    idxs: List[int] = []

    m = re.search(r"第([^根支]{1,30})(?:根|支)", s_nospace)
    if m:
        seg = m.group(1)
        idxs += [int(x) for x in re.findall(r"(\d+)", seg)]
        for ch in seg:
            if ch in CH_NUM:
                v = CH_NUM[ch]
                if v and v not in idxs:
                    idxs.append(v)

    idxs += [int(x) for x in re.findall(r"(?:sensor|s)[_\-]?(\d+)", s, flags=re.IGNORECASE)]

    if re.search(r"(根|支)", s):
        idxs += [int(x) for x in re.findall(r"(\d+)", s)]

    seen = set()
    uniq = []
    for i in idxs:
        if i not in seen and i > 0:
            seen.add(i)
            uniq.append(i)
    return uniq


def extract_label_from_text(s: str) -> Optional[int]:
    s0 = s.replace(" ", "")
    if re.search(r"(正常|無洩|0圈|零圈|〇圈)", s0):
        return 0
    if re.search(r"(七圈|7圈)", s0):
        return 7
    if re.search(r"(十圈|10圈)", s0):
        return 10
    m = re.search(r"(\d+)\s*圈", s)
    if m:
        v = int(m.group(1))
        if v in (0, 7, 10):
            return v
    m = re.search(r"([零〇一二兩三四五六七八九十])\s*圈", s)
    if m:
        v = parse_chinese_int(m.group(1))
        if v in (0, 7, 10):
            return v
    return None


def is_round_component(name: str) -> bool:
    return name.strip().lower() in ROUND_NAMES


def parse_component_to_mapping(name: str) -> Tuple[Dict[int, int], Optional[int]]:
    s = name.strip()
    if not s or is_round_component(s):
        return {}, None

    sensors = extract_sensor_indices_from_text(s)
    label = extract_label_from_text(s)
    if sensors and (label is not None):
        return {i: label for i in sensors}, None
    if (not sensors) and (label == 0):
        return {}, 0

    tokens = re.split(r"[\s,_、，&/\+\-]+", s)
    tokens = [t for t in tokens if t and not is_round_component(t)]

    mapping: Dict[int, int] = {}
    sensors_accum: List[int] = []
    labels_found: List[int] = []

    for t in tokens:
        s_list = extract_sensor_indices_from_text(t)
        lab = extract_label_from_text(t)
        if s_list and (lab is not None):
            for idx in s_list:
                mapping[idx] = lab
    if mapping:
        return mapping, None

    for t in tokens:
        s_list = extract_sensor_indices_from_text(t)
        if s_list:
            for idx in s_list:
                if idx not in sensors_accum:
                    sensors_accum.append(idx)
        lab = extract_label_from_text(t)
        if lab is not None:
            labels_found.append(lab)
    labels_found = list(dict.fromkeys(labels_found))

    if sensors_accum and len(labels_found) == 1:
        lab = labels_found[0]
        return {i: lab for i in sensors_accum}, None

    if (not sensors_accum) and (labels_found == [0]):
        return {}, 0

    return {}, None


def parse_folder_assignments(dir_path: Path) -> Tuple[Dict[int, int], Optional[int], Path]:
    mapping: Dict[int, int] = {}
    default_label: Optional[int] = None
    anchor = dir_path

    for comp in reversed(dir_path.parts):
        m, d = parse_component_to_mapping(comp)
        if m:
            mapping = m
            break
        if (default_label is None) and (d is not None):
            default_label = d
            break

    return mapping, default_label, anchor


def get_sensor_index_from_row(row: pd.Series, zero_based: bool, id_col: str) -> Optional[int]:
    """
    從資料列抓感測器編號（1-based）。
    id_col:
      - "sensor_key"  ：只看 sensor_key 欄位
      - "sensor_name" ：只看 sensor_name 欄位
      - "auto"        ：聰明選擇（預設規則）
    zero_based: 若所選欄位是 0-based 命名（例如 psr_val_0），啟用後會 +1。
    """
    def _extract_first_int(s: str) -> Optional[int]:
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None

    # Helper：依字串判斷是否「看起來」是 0-based（僅在 zero_based=True 時才會 +1）
    def _looks_zero_based(col_name: str, val: str) -> bool:
        val_lower = str(val).lower()
        if col_name == "sensor_key":
            return True  # 通常 key 像 psr_val_0 是 0-based
        # 其它啟發式：含 val/psr/prs/_0
        if ("val" in val_lower) or ("psr" in val_lower) or ("prs" in val_lower) or re.search(r"(?:^|[_-])0$", val_lower):
            return True
        # 像 sensor1 / s1 則傾向 1-based
        if re.search(r"(?:^|[^a-z])(sensor|s)\s*\d+", val_lower):
            return False
        return False

    chosen_col = None
    raw_val = None
    idx = None

    cols = []
    if id_col == "sensor_key":
        cols = ["sensor_key"]
    elif id_col == "sensor_name":
        cols = ["sensor_name"]
    else:  # auto
        cols = ["sensor_name", "sensor_key"] if not zero_based else ["sensor_key", "sensor_name"]

    for col in cols:
        if col in row and pd.notna(row[col]):
            v = str(row[col])
            iv = _extract_first_int(v)
            if iv is not None:
                chosen_col = col
                raw_val = v
                idx = iv
                break

    if idx is None:
        return None

    if zero_based and _looks_zero_based(chosen_col or "", raw_val or ""):
        idx += 1

    return idx


def class_id_from_label(label: int) -> Optional[int]:
    return LABEL_TO_CLASS.get(label, None)

def expected_class_for_row(sensor_idx: Optional[int],
                           mapping: Dict[int, int],
                           default_label: Optional[int]) -> Optional[int]:
    """
    決定單列的真實類別：
      - 若 mapping 指定了該 sensor → 使用其 label
      - 否則若 default_label 有值（通常為 0/正常） → 使用它
      - 否則預設為 0（正常）
    回傳類別索引（0/1/2）或 None。
    """
    if sensor_idx is None:
        return None
    if sensor_idx in mapping:
        lab = mapping[sensor_idx]
    elif default_label is not None:
        lab = default_label
    else:
        lab = 0
    return class_id_from_label(lab)


def map_pred_values(values: Iterable[Optional[float]], mode: str) -> List[Optional[int]]:
    """
    將 pred_label 映射為類別索引 0/1/2。
    mode='class'：期待值 ∈ {0,1,2}；若為其他值視為無效。
    mode='label'：期待值 ∈ {0,7,10}；映射 0->0,7->1,10->2；其他值無效。
    mode='auto' ：若非 NaN 的唯一值子集 ⊆ {0,1,2} → 視為 'class'；
                  若子集 ⊆ {0,7,10} → 視為 'label'；
                  否則視為 'class'（保守）。
    """
    arr = list(values)
    uniq = sorted(set([int(v) for v in arr if v is not None]))
    chosen = mode
    if mode == "auto":
        if set(uniq).issubset({0, 1, 2}):
            chosen = "class"
        elif set(uniq).issubset({0, 7, 10}):
            chosen = "label"
        else:
            chosen = "class"

    mapped: List[Optional[int]] = []
    for v in arr:
        if v is None:
            mapped.append(None)
            continue
        iv = int(v)
        if chosen == "class":
            mapped.append(iv if iv in (0, 1, 2) else None)
        else:  # 'label'
            if iv in (0, 7, 10):
                mapped.append(LABEL_TO_CLASS[iv])
            else:
                mapped.append(None)
    return mapped


def evaluate_root(root: Path, out_dir: Path, verbose: bool, zero_based: bool, pred_mode: str, id_col: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_total = 0
    overall_correct = 0
    cm_overall = np.zeros((3, 3), dtype=int)

    by_sensor_stats: Dict[int, Dict[str, int]] = {}
    folder_rows: List[Dict] = []

    diag = {
        "csv_found": 0,
        "csv_skipped_no_truth": 0,
        "csv_skipped_no_pred_col": 0,
        "rows_skipped_no_sensor_id": 0,
        "rows_skipped_bad_pred": 0,
        "rows_skipped_truth_none": 0,
        "rows_used": 0,
    }

    csv_paths = list(root.rglob("features_index.csv"))
    diag["csv_found"] = len(csv_paths)
    if verbose:
        print(f"[INFO] 掃描到 {len(csv_paths)} 個 features_index.csv")

    if not csv_paths:
        print(f"[WARN] 根目錄下找不到任何 features_index.csv：{root}")

    for csv_path in csv_paths:
        mapping, default_label, anchor = parse_folder_assignments(csv_path.parent)

        if not mapping and default_label is None:
            diag["csv_skipped_no_truth"] += 1
            if verbose:
                print(f"[WARN] 無法推出真實標籤（未指名感測器、亦非『正常/0圈』），略過：{csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            if verbose:
                print(f"[ERROR] 讀取失敗：{csv_path} -> {e}")
            continue

        if "pred_label" not in df.columns:
            diag["csv_skipped_no_pred_col"] += 1
            if verbose:
                print(f"[WARN] 缺 pred_label 欄位，略過：{csv_path}")
            continue

        if verbose:
            print(f"\n[INFO] 處理：{csv_path}")
            map_str = "; ".join([f"s{sid}->{lbl}" for sid, lbl in sorted(mapping.items())]) if mapping else "(none)"
            def_str = f"{default_label}" if default_label is not None else "(none)"
            print(f"       mapping = {map_str}, default = {def_str}")
            print(f"       使用感測器 ID 欄位: {id_col}")
            try:
                print(f"       pred_label 值分佈（原始）:\n{df['pred_label'].value_counts(dropna=False).to_string()}")
            except Exception:
                pass
            for col in ["sensor_name", "sensor_key"]:
                if col in df.columns:
                    ex = df[col].dropna().astype(str).head(5).tolist()
                    print(f"       {col} 範例：{ex}")

        # 解析 pred_label 為整數
        raw_pred = pd.to_numeric(df["pred_label"], errors="coerce").astype("Int64")
        mapped_pred = map_pred_values(list(raw_pred.astype(object).where(raw_pred.notna(), None)), mode=pred_mode)
        df["_pred_class"] = mapped_pred  # 0/1/2 或 None

        used_rows = 0
        correct_csv = 0

        for _, row in df.iterrows():
            sensor_idx = get_sensor_index_from_row(row, zero_based=zero_based, id_col=id_col)
            if sensor_idx is None:
                diag["rows_skipped_no_sensor_id"] += 1
                continue

            y_t = expected_class_for_row(sensor_idx, mapping, default_label)
            y_p = row["_pred_class"]

            if y_t is None:
                diag["rows_skipped_truth_none"] += 1
                continue
            if (y_p is None) or (y_p not in (0, 1, 2)):
                diag["rows_skipped_bad_pred"] += 1
                continue

            used_rows += 1
            diag["rows_used"] += 1

            if y_t == y_p:
                correct_csv += 1

            overall_total += 1
            overall_correct += int(y_t == y_p)
            cm_overall[y_t, y_p] += 1

            if sensor_idx not in by_sensor_stats:
                by_sensor_stats[sensor_idx] = {"total": 0, "correct": 0}
            by_sensor_stats[sensor_idx]["total"] += 1
            by_sensor_stats[sensor_idx]["correct"] += int(y_t == y_p)

        acc_csv = correct_csv / used_rows if used_rows > 0 else np.nan

        folder_rows.append({
            "folder": str(csv_path.parent),
            "file": str(csv_path.name),
            "mapping": ";".join([f"s{sid}->{lbl}" for sid, lbl in sorted(mapping.items())]) if mapping else "(none)",
            "default_label": f"{default_label}" if default_label is not None else "(none)",
            "rows_used": used_rows,
            "correct": correct_csv,
            "wrong": max(0, used_rows - correct_csv),
            "accuracy": acc_csv
        })

    # ===== 輸出：overall =====
    overall_acc = overall_correct / overall_total if overall_total > 0 else np.nan
    overall_df = pd.DataFrame([{
        "total": overall_total,
        "correct": overall_correct,
        "wrong": max(0, overall_total - overall_correct),
        "accuracy": overall_acc
    }])
    overall_df.to_csv(out_dir / "overall_summary.csv", index=False, encoding="utf-8-sig")

    cm_df = pd.DataFrame(cm_overall, index=[f"true_{k}" for k in CLASS_ORDER],
                         columns=[f"pred_{k}" for k in CLASS_ORDER])
    cm_df.to_csv(out_dir / "overall_confusion_matrix.csv", encoding="utf-8-sig")

    sensor_rows = []
    for s_idx in sorted(by_sensor_stats.keys(), key=lambda x: (x is None, x)):
        tot = by_sensor_stats[s_idx]["total"]
        cor = by_sensor_stats[s_idx]["correct"]
        acc = cor / tot if tot > 0 else np.nan
        sensor_rows.append({
            "sensor_idx": s_idx,
            "total": tot,
            "correct": cor,
            "wrong": max(0, tot - cor),
            "accuracy": acc
        })
    pd.DataFrame(sensor_rows).to_csv(out_dir / "by_sensor_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(folder_rows).to_csv(out_dir / "by_folder_summary.csv", index=False, encoding="utf-8-sig")

    print("==== 評估完成 ====")
    if overall_total:
        print(f"Overall: total={overall_total}, correct={overall_correct}, wrong={overall_total - overall_correct}, acc={overall_acc:.4f}")
    else:
        print("Overall: (no data)")
    print(f"報表已輸出至：{out_dir}")

    print("\n==== 診斷摘要 ====")
    print(f"找到 CSV 數：{diag['csv_found']}")
    print(f"略過（資料夾未能推出真實標籤）CSV 數：{diag['csv_skipped_no_truth']}")
    print(f"略過（缺 pred_label 欄位）CSV 數：{diag['csv_skipped_no_pred_col']}")
    print(f"使用的列數：{diag['rows_used']}")
    print(f"略過列（無 sensor_id）數：{diag['rows_skipped_no_sensor_id']}")
    print(f"略過列（pred_label 格式不符）數：{diag['rows_skipped_bad_pred']}")
    print(f"略過列（真實標籤為 None）數：{diag['rows_skipped_truth_none']}")


def main():
    parser = argparse.ArgumentParser(description="從 features_index.csv 計算各感測器/整體準確率與誤判數（支援多根 + R1/R2 + 自動 pred 映射）")
    parser.add_argument("--root", required=True, type=str, help="資料根目錄（底下會遞迴搜尋 features_index.csv）")
    parser.add_argument("--out", type=str, default=None, help="輸出資料夾（預設：root/_eval_report）")
    parser.add_argument("--verbose", action="store_true", help="顯示詳盡處理過程與樣本")
    parser.add_argument("--zero-based", action="store_true", help="感測器欄位（如 prs_val_0 / sensor0）為 0-based，啟用後自動 +1")
    parser.add_argument("--pred-mode", choices=["auto", "class", "label"], default="auto",
                        help="pred_label 模式：auto=自動判斷；class=0/1/2；label=0/7/10 → 0/1/2")
    parser.add_argument("--id-col", choices=["auto", "sensor_key", "sensor_name"], default="sensor_key", help="用哪個欄位取感測器編號（預設只看 sensor_key）")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = root.resolve()
    out_dir = Path(args.out).resolve() if args.out else (root / "_eval_report")

    evaluate_root(root, out_dir, verbose=args.verbose, zero_based=args.zero_based, pred_mode=args.pred_mode, id_col=args.id_col)


if __name__ == "__main__":
    main()
