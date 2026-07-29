"""各类工具函数。"""

import time
import json
from typing import Dict, Optional, List
from pathlib import Path


def format_metrics_table(results: Dict[str, Dict]) -> str:
    """将多个模型的评估结果格式化为 Markdown 表格。

    Args:
        results: { model_name: {"P": float, "R": float, "F1": float, ...} }

    Returns:
        Markdown 表格字符串
    """
    headers = ["模型", "P", "R", "F1", "OOV-R", "IV-R", "OOV%", "POS-Acc"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([":---:"] * len(headers)) + "|",
    ]
    for name, m in results.items():
        row = [
            name,
            f"{m.get('P', 0):.4f}",
            f"{m.get('R', 0):.4f}",
            f"{m.get('F1', 0):.4f}",
            f"{m.get('OOV-R', 0):.4f}",
            f"{m.get('IV-R', 0):.4f}",
            f"{m.get('OOV%', 0):.4f}",
            f"{m.get('POS-Acc', 0):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def aggregate_cv_results(
    fold_results: List[Dict[str, Dict]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """将各折结果按 {方法: {指标: {mean, std}}} 聚合。

    Args:
        fold_results: 长度为 k 的列表，每项是 {方法名: {指标: 值}}（单折结果）

    Returns:
        {方法名: {指标: {"mean": float, "std": float, "values": [...]}}}
    """
    import math

    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not fold_results:
        return agg
    method_names = list(fold_results[0].keys())
    metric_keys = ["P", "R", "F1", "OOV-R", "IV-R", "OOV%", "POS-Acc"]

    for name in method_names:
        agg[name] = {}
        for key in metric_keys:
            vals = [fr[name][key] for fr in fold_results
                    if name in fr and key in fr[name]]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            agg[name][key] = {
                "mean": mean,
                "std": math.sqrt(var),
                "values": vals,
            }
    return agg


def format_cv_table(agg: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """将 CV 聚合结果格式化为 Markdown 表格（均值 ± 标准差）。"""
    headers = ["模型", "P", "R", "F1", "OOV-R", "IV-R", "OOV%", "POS-Acc"]
    metric_keys = ["P", "R", "F1", "OOV-R", "IV-R", "OOV%", "POS-Acc"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([":---:"] * len(headers)) + "|",
    ]
    for name, metrics in agg.items():
        row = [name]
        for key in metric_keys:
            if key in metrics:
                row.append(f"{metrics[key]['mean']:.4f}±{metrics[key]['std']:.4f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_category_table(cat_results: Dict[str, Dict[str, Dict]]) -> str:
    """将按文献类别拆分的评估结果格式化为 Markdown 表格。

    Args:
        cat_results: { model_name: { category: {"P": float, "R": float,
                        "F1": float, "n_sents": int, ...} } }

    Returns:
        Markdown 表格字符串
    """
    headers = ["模型", "类别", "P", "R", "F1", "OOV-R", "IV-R", "OOV%", "POS-Acc", "句子数"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([":---:"] * len(headers)) + "|",
    ]
    for model_name, cats in cat_results.items():
        for cat, metrics in cats.items():
            row = [
                model_name,
                cat,
                f"{metrics.get('P', 0):.4f}",
                f"{metrics.get('R', 0):.4f}",
                f"{metrics.get('F1', 0):.4f}",
                f"{metrics.get('OOV-R', 0):.4f}",
                f"{metrics.get('IV-R', 0):.4f}",
                f"{metrics.get('OOV%', 0):.4f}",
                f"{metrics.get('POS-Acc', 0):.4f}",
                str(metrics.get("n_sents", "-")),
            ]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def aggregate_cv_category_results(
    fold_cat_results: List[Dict[str, Dict[str, Dict]]],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """将各折类别结果按 {方法: {类别: {指标: {mean, std}}}} 聚合。

    Args:
        fold_cat_results: 长度为 k 的列表，每项是 {方法名: {类别: {指标: 值}}}

    Returns:
        {方法名: {类别: {指标: {"mean": float, "std": float, "values": [...]}}}}
    """
    import math
    agg: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    metric_keys = ["P", "R", "F1", "OOV-R", "IV-R", "OOV%", "POS-Acc"]

    for fold_cats in fold_cat_results:
        for model_name, cats in fold_cats.items():
            if model_name not in agg:
                agg[model_name] = {}
            for cat, metrics in cats.items():
                if cat not in agg[model_name]:
                    agg[model_name][cat] = {
                        key: {"values": [], "n_sents_total": 0}
                        for key in metric_keys
                    }
                    agg[model_name][cat]["n_sents"] = {"n_sents_total": 0, "values": []}
                for key in metric_keys:
                    if key in metrics:
                        agg[model_name][cat][key]["values"].append(metrics[key])
                if "n_sents" in metrics:
                    agg[model_name][cat]["n_sents"]["n_sents_total"] += metrics["n_sents"]
                    agg[model_name][cat]["n_sents"]["values"].append(metrics["n_sents"])

    # 计算 mean/std
    for model_name, cats in agg.items():
        for cat, mdict in cats.items():
            for key in metric_keys:
                vals = mdict[key]["values"]
                if vals:
                    mean = sum(vals) / len(vals)
                    var = sum((v - mean) ** 2 for v in vals) / len(vals)
                    mdict[key]["mean"] = mean
                    mdict[key]["std"] = math.sqrt(var)

    return agg


def format_cv_category_table(
    agg_cat: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
) -> str:
    """将 CV 类别聚合结果格式化为 Markdown 表格（均值 ± 标准差）。"""
    headers = ["模型", "类别", "P", "R", "F1", "OOV-R", "IV-R", "OOV%", "POS-Acc", "句子数"]
    metric_keys = ["P", "R", "F1", "OOV-R", "IV-R", "OOV%", "POS-Acc"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([":---:"] * len(headers)) + "|",
    ]
    for model_name, cats in agg_cat.items():
        for cat, mdict in cats.items():
            row = [model_name, cat]
            for key in metric_keys:
                if key in mdict and "mean" in mdict[key]:
                    row.append(f"{mdict[key]['mean']:.4f}±{mdict[key]['std']:.4f}")
                else:
                    row.append("-")
            n_sents = mdict.get("n_sents", {}).get("n_sents_total", "-")
            row.append(str(n_sents))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_results(results: Dict, path: Path) -> None:
    """将评估结果保存为 JSON 文件。"""
    serializable = {}
    for name, metrics in results.items():
        serializable[name] = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def append_run_log(
    log_path: Path,
    corpus_path: str,
    max_sentences: Optional[int],
    methods_config: Dict[str, Dict],
    results: Dict[str, Dict],
    elapsed: float,
) -> None:
    """将一次运行记录追加到 JSONL 日志文件。

    每条记录包含: 时间戳、语料、截断数、各方法配置、结果、耗时。
    """
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus": str(corpus_path),
        "max_sentences": max_sentences,
        "elapsed_seconds": round(elapsed, 2),
        "methods_config": methods_config,
        "results": {
            name: {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in m.items()
            }
            for name, m in results.items()
        },
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Timer:
    """简单的计时上下文管理器。"""

    def __init__(self, name: str = ""):
        self.name = name
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
