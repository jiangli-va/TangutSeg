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


def save_inference_model(
    methods: Dict,
    method_name: str,
    base_dir: Path,
    lexicon_extractor=None,
) -> None:
    """保存完整推理模型到指定目录。

    支持 TangutEncoderSegmenter (.pt) 和 CRFSegmenter (.joblib):
        - TEnc: <name>_model.pt + <name>_lexicon.pkl + <name>_gap.pkl
        - CRF:  <name>_model.joblib + <name>_lexicon.pkl + <name>_gap.pkl

    Args:
        methods: {方法名: Segmenter实例}
        method_name: 要保存的方法名
        base_dir: 保存根目录 (如 BASE / "saved_models")
        lexicon_extractor: 备选词典提取器 (当 segmenter 内部没有时使用)
    """
    # lazy import 避免循环依赖
    from models.crf import CRFSegmenter
    try:
        from pretrain.segmenter import TangutEncoderSegmenter
    except ImportError:
        TangutEncoderSegmenter = None

    saved_models_dir = base_dir / "saved_models"
    saved_models_dir.mkdir(parents=True, exist_ok=True)

    if method_name not in methods:
        print(f"[WARN] 方法 '{method_name}' 未找到，不可保存。可用方法: {list(methods.keys())}")
        return

    segmenter = methods[method_name]
    is_crf = isinstance(segmenter, CRFSegmenter)
    is_tenc = (TangutEncoderSegmenter is not None
               and isinstance(segmenter, TangutEncoderSegmenter))

    if not is_crf and not is_tenc:
        print(f"[WARN] '{method_name}' 类型不支持保存为推理模型 "
              f"(需要 CRFSegmenter 或 TangutEncoderSegmenter)")
        return

    print(f"\n{'=' * 60}")
    print(f"  Saving inference model: {method_name}")
    print(f"{'=' * 60}\n")

    # 1) 保存模型
    if is_crf:
        model_path = saved_models_dir / f"{method_name}_model.joblib"
        segmenter.save(str(model_path))
    else:
        model_path = saved_models_dir / f"{method_name}_model.pt"
        segmenter.save(str(model_path))
    print(f"  ✓ Model saved to {model_path}")

    # 2) 保存词典特征提取器
    if is_crf:
        # CRF fit 时直接修改了 _lexicon_extractor
        extractor_for_inference = getattr(segmenter, '_lexicon_extractor', None)
    else:
        extractor_for_inference = getattr(segmenter, '_extractor_for_inference', None)

    if extractor_for_inference is not None:
        lex_path = saved_models_dir / f"{method_name}_lexicon.pkl"
        extractor_for_inference.save(str(lex_path))
        print(f"  ✓ Lexicon extractor saved to {lex_path}")
    elif lexicon_extractor is not None:
        lex_path = saved_models_dir / f"{method_name}_lexicon.pkl"
        lexicon_extractor.save(str(lex_path))
        print(f"  ✓ Lexicon extractor (base) saved to {lex_path}")

    # 3) 保存 gap (unlabeled) 特征提取器
    try:
        from models.unlabeled_stats import UnlabeledStatsExtractor
        unlabeled = getattr(segmenter, '_unlabeled_extractor', None)
        if unlabeled is not None and isinstance(unlabeled, UnlabeledStatsExtractor):
            gap_path = saved_models_dir / f"{method_name}_gap.pkl"
            unlabeled.save(str(gap_path))
            print(f"  ✓ Gap extractor saved to {gap_path}")
    except ImportError:
        pass

    print(f"\n  Inference model saved to {saved_models_dir}/")
    print(f"  Files: {', '.join(p.name for p in saved_models_dir.glob(f'{method_name}_*'))}")


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
