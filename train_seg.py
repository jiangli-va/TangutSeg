"""分词模型入口 —— 全流程运行：语料解析 → 数据划分 → 训练/预测 → 评估对比。

用法示例:
    python run_seg.py                                    # 运行全部方法
    python run_seg.py --methods dict                     # 只跑词典法(3种)
    python run_seg.py --methods crf,crf5,crf8            # 只跑指定 CRF 变体
    python run_seg.py --methods bilstm,bilstm5           # 只跑指定 BiLSTM-CRF 变体
    python run_seg.py --methods dict,crf,bilstm,ltp      # 组合多种方法
    python run_seg.py --max 2000                         # 截断语料（快速调试）
    python run_seg.py --cv                               # K 折交叉验证，报告均值±标准差
    python run_seg.py --cv --folds 10                    # 10 折交叉验证
    python run_seg.py --skip-tools                       # 跳过所有外部分词工具(LTP/HanLP/Stanza/Trankit)

可用的 --methods 值 (逗号分隔, 大小写敏感):
    ┌──────────────────┬─────────────────────────────────────────────────────┐
    │ 简写             │ 对应全称 / 说明                                       │
    ├──────────────────┼─────────────────────────────────────────────────────┤
    │ dict / dict1/2/3 │ Dict1(corpus) / Dict2(predef) / Dict3(combine)      │
    │ crf              │ CRF (无词典特征, level=0)                            │
    │ crf0 ~ crf9      │ CRF 词典特征消融 (0=无,1=BIE,2=+rel_seen,           │
    │                  │   3=+rel_all(=dict_core),4=+meta,5=dict_full,        │
    │                  │   6=dict_full+freq, 7=dict_full+freq+assoc,          │
    │                  │   8=dict_full+dist_all, 9=dict_full+freq+entropy)    │
    │ bilstm           │ BiLSTM-CRF (无词典特征, level=0)                     │
    │ bilstm0 ~ bilstm9│ BiLSTM-CRF 词典特征消融                              │
    │                  │   0=无,1=BIE,2=+rel_seen,3=+rel_all(dict_core),      │
    │                  │   4=dict_full(20维,同CRF5),5=+internal,6=+domain,    │
    │                  │   7~9=+gap (同CRF 6~8)                               │
    │ ltp              │ LTP 外部分词工具 (需安装 ltp)                        │
    │ hanlp            │ HanLP 外部分词工具 (需安装 hanlp)                    │
    │ stanza           │ Stanza 外部分词工具 (需安装 stanza)                   │
    │ trankit          │ Trankit 外部分词工具 (需安装 trankit)                 │
    └──────────────────┴─────────────────────────────────────────────────────┘

    注意: "dict" 会匹配 dict1/2/3 全部三种词典来源。
          "crf" 会匹配 crf + crf0~crf8 全部 CRF 变体。
          "bilstm" 会匹配 bilstm + bilstm0~bilstm9 全部 BiLSTM-CRF 变体。
          可用 "crf5" 或 "crf6,crf7,crf8" 精确指定某几个变体。

流程概览:
    1. 解析语料 (CorpusParser)
    2. 8:1:1 划分训练/开发/测试集
    3. 对每种方法: fit() → predict_batch() → evaluate()
    4. 汇总对比表，保存到 output/
"""

import sys
import argparse
from pathlib import Path
import random
from typing import Dict, List, Set, Optional, Tuple

# 确保项目根目录在 sys.path 中
BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import config

# ---- 固定全局随机种子, 保证可复现 ----
random.seed(config.RANDOM_SEED)
import numpy as np
np.random.seed(config.RANDOM_SEED)
import torch
torch.manual_seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    # 关闭 benchmark 以消除非确定性; 如需速度可以注释掉
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
from data.corpus import CorpusParser
from data.dataset import split_dataset, make_kfolds, compute_vocabulary_stats
from evaluation.metrics import evaluate
from utils.helpers import (format_metrics_table, format_category_table,
                           save_results, append_run_log,
                           aggregate_cv_results, format_cv_table,
                           aggregate_cv_category_results,
                           format_cv_category_table, Timer,
                           save_inference_model)
from models.base import Segmenter


# ======================== 注册所有方法 ========================

def build_methods(which: Optional[List[str]] = None) -> Dict[str, 'Segmenter']:
    """构建一个 {方法名: Segmenter实例} 的字典。

    'which' 是一个可选的筛选列表，如 ['dict', 'crf']。
    None 表示运行所有方法。
    """
    all_methods = {}

    # ---- 词典法 (三种词典来源) ----
    predefined_path = str(BASE / "corpus" / "西夏文词典.json")

    if which is None or "dict" in which or "dict1" in which or "dict2" in which or "dict3" in which:
        from models.dictionary import DictionarySegmenter
        if which is None or "dict" in which or "dict1" in which:
            all_methods["Dict1(corpus)"] = DictionarySegmenter(
                mode=config.DICT_MATCH_MODE, dict_source="corpus")
        if which is None or "dict" in which or "dict2" in which:
            all_methods["Dict2(predef)"] = DictionarySegmenter(
                mode=config.DICT_MATCH_MODE, dict_source="predefined",
                predefined_path=predefined_path)
        if which is None or "dict" in which or "dict3" in which:
            all_methods["Dict3(combine)"] = DictionarySegmenter(
                mode=config.DICT_MATCH_MODE, dict_source="combined",
                predefined_path=predefined_path)

    # ---- gap 特征 (从四行对译 JSON 提取, CRF 和 BiLSTM 共享) ----
    gap_json_path = config.UNLABELED_JSON_PATH
    shared_unlabeled: Optional[object] = None
    if gap_json_path and Path(gap_json_path).exists():
        from models.unlabeled_stats import UnlabeledStatsExtractor
        shared_unlabeled = UnlabeledStatsExtractor(bigram_metric=config.BIGRAM_METRIC)
        shared_unlabeled.load(gap_json_path)

    # 当前关联度量名称 (用于 label)
    _metric_label = {"dpmi": "dPMI", "dice": "Dice", "t_score": "t-score"}.get(
        config.BIGRAM_METRIC, config.BIGRAM_METRIC
    )

    # ---- CRF (含词典格网特征消融变体 v2 + 无标注 gap 特征) ----
    # 0=baseline, 1=BIE, 2=BIE+rel_seen, 3=BIE+rel_all(=dict_core, 17维),
    # 4=BIE+meta, 5=dict_full(BIE+rel_all+meta, 20维),
    # 6=dict_full+freq, 7=dict_full+freq+assoc, 8=dict_full+dist_all, 9=dict_full+freq+entropy
    crf_levels = {
        "crf": 0,
        "crf0": 0, "crf1": 1, "crf2": 2,
        "crf3": 3, "crf4": 4, "crf5": 5,
        "crf6": 6, "crf7": 7, "crf8": 8,
        "crf9": 9,
    }
    crf_requested = which is None or any(k in crf_levels and k in which for k in crf_levels)

    if crf_requested:
        from models.crf import CRFSegmenter
        from models.lexicon import LexiconFeatureExtractor
        # 共享同一个词典特征提取器
        shared_lexicon = LexiconFeatureExtractor()
        shared_lexicon.load(predefined_path)

        level_label = {
            0: "CRF", 1: "CRF+BIE", 2: "CRF+BIE+rel_seen",
            3: "CRF+BIE+rel_all", 4: "CRF+BIE+meta", 5: "CRF+dict_full",
            6: f"CRF+dict_full+freq", 7: f"CRF+dict_full+freq+{_metric_label}",
            8: f"CRF+dict_full+dist_all",
            9: f"CRF+dict_full+freq+entropy",
        }
        for key, level in crf_levels.items():
            if which is None or key in which or "crf" in which:
                if level <= 5:
                    # 纯词典特征消融 (0-5)
                    use_lexicon = shared_lexicon if level > 0 else None
                    all_methods[level_label[level]] = CRFSegmenter(
                        c1=config.CRF_PARAMS["c1"],
                        c2=config.CRF_PARAMS["c2"],
                        max_iterations=config.CRF_PARAMS["max_iterations"],
                        all_possible_transitions=config.CRF_PARAMS["all_possible_transitions"],
                        lexicon_extractor=use_lexicon,
                        dict_feature_level=level,
                    )
                else:
                    # gap 特征消融 (6-9): dict=5 (dict_full), gap=level-5
                    # 6=freq(1), 7=freq+assoc(2), 8=all(3), 9=freq+entropy(4)
                    gap_lv = level - 5  # 1=freq, 2=freq+assoc, 3=dist_all, 4=freq+entropy
                    all_methods[level_label[level]] = CRFSegmenter(
                        c1=config.CRF_PARAMS["c1"],
                        c2=config.CRF_PARAMS["c2"],
                        max_iterations=config.CRF_PARAMS["max_iterations"],
                        all_possible_transitions=config.CRF_PARAMS["all_possible_transitions"],
                        lexicon_extractor=shared_lexicon,
                        dict_feature_level=5,
                        unlabeled_extractor=shared_unlabeled,
                        gap_feature_level=gap_lv,
                    )

    # ---- BiLSTM-CRF (与 CRF 对应的词典特征消融变体 v4) ----
    # 0=baseline, 1=BIE, 2=+rel_seen, 3=+rel_all(=dict_core, 17维),
    # 4=dict_full(20维, 同CRF level 5), 5=+internal, 6=+domain,
    # 7=+gap=freq, 8=+gap=freq+assoc, 9=+gap=freq+assoc+entropy
    bilstm_levels = {
        "bilstm": 0,
        "bilstm0": 0, "bilstm1": 1, "bilstm2": 2, "bilstm3": 3,
        "bilstm4": 4, "bilstm5": 5, "bilstm6": 6,
        "bilstm7": 7, "bilstm8": 8, "bilstm9": 9,
    }
    bilstm_requested = which is None or any(k in bilstm_levels and k in which for k in bilstm_levels)
    if bilstm_requested:
        from models.bilstm_crf import BiLSTMCRFSegmenter
        from models.lexicon import LexiconFeatureExtractor as BiLSTMLexicon
        from models.lexicon import build_internal_only_trie
        # 与 CRF 共享同一个词典特征提取器 (公平对比)
        bilstm_shared_lexicon = BiLSTMLexicon()
        bilstm_shared_lexicon.load(predefined_path)

        bilstm_label = {
            0: "BiLSTM-0", 1: "BiLSTM-BIE",
            2: "BiLSTM-rel_seen", 3: "BiLSTM-rel_all",
            4: "BiLSTM+dict_full", 5: "BiLSTM+dict_full+internal",
            6: "BiLSTM+dict_full+internal+domain",
            7: f"BiLSTM+dict_full+internal+domain+freq",
            8: f"BiLSTM+dict_full+internal+domain+freq+{_metric_label}",
            9: f"BiLSTM+dict_full+internal+domain+freq+{_metric_label}+entropy",
        }
        for key, level in bilstm_levels.items():
            if which is None or key in which or "bilstm" in which:
                use_lexicon = bilstm_shared_lexicon if level > 0 else None
                # level >= 5: 需要 internal trie; 先传一个空占位, fit() 时会 OOF 重建
                internal_trie = bilstm_shared_lexicon.trie if level >= 5 else None
                domain_dist_dim = config.BILSTM_CRF_PARAMS.get("domain_dist_dim", 2)
                jingshu_loss_weight = config.BILSTM_CRF_PARAMS.get("jingshu_loss_weight", 1.0)
                # gap 特征: level 7-9 对应 gap_level 1-3 (freq / freq+assoc / all)
                gap_lv = max(0, level - 6)
                use_gap_extractor = shared_unlabeled if gap_lv > 0 else None
                all_methods[bilstm_label[level]] = BiLSTMCRFSegmenter(
                    embedding_dim=config.BILSTM_CRF_PARAMS["embedding_dim"],
                    hidden_dim=config.BILSTM_CRF_PARAMS["hidden_dim"],
                    num_layers=config.BILSTM_CRF_PARAMS["num_layers"],
                    dropout=config.BILSTM_CRF_PARAMS["dropout"],
                    learning_rate=config.BILSTM_CRF_PARAMS["learning_rate"],
                    batch_size=config.BILSTM_CRF_PARAMS["batch_size"],
                    epochs=config.BILSTM_CRF_PARAMS["epochs"],
                    device=config.BILSTM_CRF_PARAMS["device"],
                    lr_patience=config.BILSTM_CRF_PARAMS["lr_patience"],
                    lr_factor=config.BILSTM_CRF_PARAMS["lr_factor"],
                    early_stop_patience=config.BILSTM_CRF_PARAMS["early_stop_patience"],
                    grad_clip=config.BILSTM_CRF_PARAMS["grad_clip"],
                    lexicon_extractor=use_lexicon,
                    dict_feature_level=level,
                    dict_dropout=config.BILSTM_CRF_PARAMS["dict_dropout"],
                    internal_trie=internal_trie,
                    domain_dist_dim=domain_dist_dim,
                    jingshu_loss_weight=jingshu_loss_weight,
                    unlabeled_extractor=use_gap_extractor,
                    gap_feature_level=gap_lv,
                )

    # ---- 外部分词工具 (可选) ----
    if which is None or "ltp" in which:
        try:
            from models.tools import LTPSegmenter
            all_methods["LTP"] = LTPSegmenter(model_path=config.TOOL_PATHS.get("ltp", "LTP"))
        except Exception as e:
            print(f"[WARN] LTP 加载失败: {e}")

    if which is None or "hanlp" in which:
        try:
            from models.tools import HanLPSegmenter
            all_methods["HanLP"] = HanLPSegmenter()
        except Exception as e:
            print(f"[WARN] HanLP 加载失败: {e}")

    if which is None or "stanza" in which:
        try:
            from models.tools import StanzaSegmenter
            all_methods["Stanza"] = StanzaSegmenter(lang=config.TOOL_PATHS.get("stanza", "zh"))
        except Exception as e:
            print(f"[WARN] Stanza 加载失败: {e}")

    if which is None or "trankit" in which:
        try:
            from models.tools import TrankitSegmenter
            all_methods["Trankit"] = TrankitSegmenter(lang=config.TOOL_PATHS.get("trankit", "chinese"))
        except Exception as e:
            print(f"[WARN] Trankit 加载失败: {e}")

    return all_methods


# ======================== 主流程 ========================

def train_eval_dataset(
    methods: Dict[str, 'Segmenter'],
    dataset,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, Dict]]]:
    """在给定 dataset 上训练并评估所有方法。

    Returns:
        (overall_results, category_results)
        overall_results: {方法名: {指标}}  —— 全测试集
        category_results: {方法名: {类别: {指标}}}  —— 按类别拆分
    """
    train_vocab: Set[str] = set()
    for ws in dataset.train_words:
        train_vocab.update(ws)

    test_categories = getattr(dataset, 'categories', [])
    has_categories = bool(test_categories) and len(test_categories) == len(dataset.test_words)

    results: Dict[str, Dict] = {}
    cat_results: Dict[str, Dict[str, Dict]] = {}  # {方法: {类别: {指标}}}

    for name, segmenter in methods.items():
        if verbose:
            print(f"\n--- {name} ---")

        with Timer(f"{name}.fit"):
            if "BiLSTM" in name:
                # v4: 为 BiLSTM (level 5) 传递 train/dev_categories → fit() 内部计算领域分布向量
                train_cats = getattr(dataset, 'train_categories', None)
                dev_cats = getattr(dataset, 'dev_categories', None)
                segmenter.fit(dataset.train_words, dataset.train_tags,
                             dev_words=dataset.dev_words, dev_tags=dataset.dev_tags,
                             train_categories=train_cats,
                             dev_categories=dev_cats)
            elif "CRF" in name:
                # CRF 使用 dev 数据进行早停
                segmenter.fit(dataset.train_words, dataset.train_tags,
                             dev_words=dataset.dev_words, dev_tags=dataset.dev_tags)
            else:
                segmenter.fit(dataset.train_words, dataset.train_tags)
        if verbose:
            print(f"  fit: 完成")

        with Timer(f"{name}.predict"):
            # v4: 为 BiLSTM+internal+domain 传递测试集的 categories
            test_cats = None
            if "BiLSTM" in name:
                test_cats = getattr(dataset, 'test_categories', None)
            if test_cats:
                pred_result = segmenter.predict_batch(
                    dataset.test_sents, categories=test_cats)
            else:
                pred_result = segmenter.predict_batch(dataset.test_sents)
        if isinstance(pred_result, tuple):
            pred_words, pred_pos = pred_result
        else:
            pred_words = pred_result
            pred_pos = None
        if verbose:
            print(f"  predict: {len(pred_words)} 句完成")

        # ---- 全测试集评估 ----
        metrics = evaluate(pred_words, dataset.test_words, train_vocab=train_vocab,
                          pred_pos_list=pred_pos, gold_pos_list=dataset.test_tags)
        if verbose:
            print(f"  评估: {metrics}")

        results[name] = {
            "P": metrics.precision,
            "R": metrics.recall,
            "F1": metrics.f1,
            "OOV-R": metrics.oov_recall,
            "IV-R": metrics.iv_recall,
            "OOV%": metrics.oov_rate,
            "POS-Acc": metrics.pos_accuracy,
        }

        # ---- 按类别拆分评估 ----
        if has_categories:
            cat_results[name] = {}
            unique_cats = sorted(set(test_categories))
            for cat in unique_cats:
                indices = [i for i, c in enumerate(test_categories) if c == cat]
                cat_pred = [pred_words[i] for i in indices]
                cat_gold = [dataset.test_words[i] for i in indices]
                cat_pred_pos = [pred_pos[i] for i in indices] if pred_pos else None
                cat_gold_pos = [dataset.test_tags[i] for i in indices] if pred_pos else None
                cat_metrics = evaluate(cat_pred, cat_gold, train_vocab=train_vocab,
                                      pred_pos_list=cat_pred_pos, gold_pos_list=cat_gold_pos)
                cat_results[name][cat] = {
                    "P": cat_metrics.precision,
                    "R": cat_metrics.recall,
                    "F1": cat_metrics.f1,
                    "OOV-R": cat_metrics.oov_recall,
                    "IV-R": cat_metrics.iv_recall,
                    "OOV%": cat_metrics.oov_rate,
                    "POS-Acc": cat_metrics.pos_accuracy,
                    "n_sents": len(indices),
                }
            if verbose:
                for cat in unique_cats:
                    c = cat_results[name][cat]
                    print(f"    [{cat}] P={c['P']:.4f} R={c['R']:.4f} F1={c['F1']:.4f} "
                          f"OOV-R={c['OOV-R']:.4f} IV-R={c['IV-R']:.4f} (n={c['n_sents']})")

    return results, cat_results


def run_cross_validation(all_words, all_tags, all_categories, which, k, run_timer):
    """K 折交叉验证流程：对所有方法报告均值±标准差。"""
    # ---- 划分 K 折 ----
    print("\n" + "=" * 60)
    print(f"Step 2/5: 生成 {k} 折交叉验证数据集 (分层) ...")
    print("=" * 60)
    folds = make_kfolds(all_words, all_tags, k=k, seed=config.RANDOM_SEED,
                        categories=all_categories)
    for i, ds in enumerate(folds):
        cats = getattr(ds, 'categories', [])
        cat_str = ""
        if cats:
            from collections import Counter
            cc = Counter(cats)
            cat_str = " | " + ", ".join(f"{c}:{n}" for c, n in cc.most_common())
        print(f"  -> Fold {i+1}: train={len(ds.train_words)} "
              f"dev={len(ds.dev_words)} test={len(ds.test_words)}{cat_str}")

    # ---- 逐折训练评估 ----
    print("\n" + "=" * 60)
    print(f"Step 3-4/5: {k} 折训练 & 评估 ...")
    print("=" * 60)
    fold_results: List[Dict[str, Dict]] = []
    fold_cat_results: List[Dict[str, Dict[str, Dict]]] = []
    for i, ds in enumerate(folds):
        print("\n" + "#" * 60)
        print(f"# Fold {i+1}/{k}")
        print("#" * 60)
        methods = build_methods(which)
        if not methods:
            print("  [ERROR] 没有可用的分词方法，退出。")
            return
        res, cat_res = train_eval_dataset(methods, ds, verbose=True)
        fold_results.append(res)
        fold_cat_results.append(cat_res)
        print(f"\n[Fold {i+1} 结果]")
        print(format_metrics_table(res))
        if cat_res:
            print(format_category_table(cat_res))

    # ---- 聚合 ----
    print("\n" + "=" * 60)
    print(f"Step 5/5: {k} 折汇总（均值 ± 标准差）")
    print("=" * 60)
    agg = aggregate_cv_results(fold_results)
    cv_table = format_cv_table(agg)
    print("\n### 全测试集")
    print(cv_table)

    # 分类汇总
    if fold_cat_results and any(fold_cat_results):
        agg_cat = aggregate_cv_category_results(fold_cat_results)
        cat_cv_table = format_cv_category_table(agg_cat)
        print("\n### 按文献类别")
        print(cat_cv_table)

    # ---- 保存 ----
    output_path = config.OUTPUT_DIR / "results_cv.json"
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "k": k,
            "seed": config.RANDOM_SEED,
            "fold_results": fold_results,
            "aggregate": agg,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")

    md_path = config.OUTPUT_DIR / "results_cv.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 分词+词性标注 {k} 折交叉验证结果\n\n")
        f.write(f"折数: {k} | seed: {config.RANDOM_SEED}\n\n")
        f.write("## 汇总（均值 ± 标准差）\n\n")
        f.write("### 全测试集\n\n")
        f.write(cv_table + "\n\n")
        if fold_cat_results and any(fold_cat_results):
            f.write("### 按文献类别\n\n")
            f.write(cat_cv_table + "\n\n")
        for i, res in enumerate(fold_results):
            f.write(f"## Fold {i+1}\n\n")
            f.write(format_metrics_table(res) + "\n\n")
            if i < len(fold_cat_results) and fold_cat_results[i]:
                f.write(format_category_table(fold_cat_results[i]) + "\n\n")
    print(f"Markdown 表格已保存到: {md_path}")

    run_timer.__exit__(None, None, None)
    total_elapsed = run_timer.elapsed
    log_path = config.OUTPUT_DIR / "run_log.jsonl"
    methods_config = {
        "cv_folds": k,
        "dict": {"mode": config.DICT_MATCH_MODE},
        "crf": {k2: v for k2, v in config.CRF_PARAMS.items()},
        "bilstm_crf": {k2: v for k2, v in config.BILSTM_CRF_PARAMS.items()
                       if k2 not in ("device",)},
    }
    append_run_log(
        log_path=log_path,
        corpus_path=config.CORPUS_PATH,
        max_sentences=config.MAX_SENTENCES,
        methods_config=methods_config,
        results={name: {kk: vv["mean"] for kk, vv in m.items()}
                 for name, m in agg.items()},
        elapsed=total_elapsed,
    )
    print(f"运行日志已追加到: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="分词全流程实验 (纯分词模型)")
    parser.add_argument("--methods", type=str, default=None,
                        help="逗号分隔的方法列表, e.g. 'dict,crf,ltp'")
    parser.add_argument("--max", type=int, default=config.MAX_SENTENCES,
                        help="最大句子数（截断语料，快速调试）")
    parser.add_argument("--skip-tools", action="store_true",
                        help="跳过所有外部分词工具")
    parser.add_argument("--cv", action="store_true",
                        help="使用 K 折交叉验证（对所有方法），报告均值±标准差")
    parser.add_argument("--folds", type=int, default=5,
                        help="交叉验证折数（默认 5）")
    parser.add_argument("--save-model", type=str, default=None,
                        help="训练完成后保存指定方法的推理模型, e.g. 'CRF+dict_full'")
    args = parser.parse_args()

    # 解析方法列表
    which = [m.strip() for m in args.methods.split(",")] if args.methods else None
    if args.skip_tools:
        if which is None:
            which = ["dict", "crf", "bilstm"]
        else:
            which = [m for m in which if m in ("dict", "crf", "bilstm")]

    max_sentences = args.max

    # 全局计时
    run_timer = Timer("total")
    run_timer.__enter__()

    # ==================== Step 1: 解析语料 ====================
    print("=" * 60)
    print("Step 1/5: 解析西夏文语料 ...")
    print("=" * 60)
    parser_obj = CorpusParser(config.CORPUS_PATH)
    with Timer("解析语料"):
        all_words, all_tags, all_categories = parser_obj.parse_file_with_categories(
            max_sentences=max_sentences)
    stats = compute_vocabulary_stats(all_words)
    print(f"  -> 句子数: {stats['num_sentences']:,}")
    print(f"  -> 词语数: {stats['num_tokens']:,}")
    print(f"  -> 字数:   {stats['num_chars']:,}")
    print(f"  -> 不同词: {stats['unique_words']:,}")
    print(f"  -> 不同字: {stats['unique_chars']:,}")
    print(f"  -> 平均句长: {stats['avg_sentence_len_tokens']:.1f} 词 / "
          f"{stats['avg_sentence_len_chars']:.1f} 字")
    # 类别分布
    from collections import Counter
    cat_counts = Counter(all_categories)
    for cat, cnt in cat_counts.most_common():
        print(f"  -> {cat}: {cnt} 句")

    # ==================== 交叉验证分支 ====================
    if args.cv:
        run_cross_validation(all_words, all_tags, all_categories, which, args.folds, run_timer)
        return

    # ==================== Step 2: 划分数据集 ====================
    print("\n" + "=" * 60)
    print("Step 2/5: 划分训练/开发/测试集 (8:1:1, 分层抽样) ...")
    print("=" * 60)
    dataset = split_dataset(
        all_words, all_tags,
        train_ratio=config.TRAIN_RATIO,
        dev_ratio=config.DEV_RATIO,
        seed=config.RANDOM_SEED,
        categories=all_categories,
    )
    print(f"  -> 训练集: {len(dataset.train_words):,} 句")
    print(f"  -> 开发集: {len(dataset.dev_words):,} 句")
    print(f"  -> 测试集: {len(dataset.test_words):,} 句")
    if dataset.categories:
        test_cat_counts = Counter(dataset.categories)
        for cat, cnt in test_cat_counts.most_common():
            print(f"     {cat}: {cnt} 句")

    # ==================== Step 3: 构建方法 ====================
    print("\n" + "=" * 60)
    print("Step 3/5: 初始化分词方法 ...")
    print("=" * 60)
    methods = build_methods(which)
    for name in methods:
        print(f"  -> {name} ✓")
    if not methods:
        print("  [ERROR] 没有可用的分词方法，退出。")
        return

    # ==================== Step 4: 训练 + 预测 ====================
    print("\n" + "=" * 60)
    print("Step 4/5: 训练 & 预测 ...")
    print("=" * 60)

    results, cat_results = train_eval_dataset(methods, dataset, verbose=True)

    # ---- 保存推理模型 (--save-model) ----
    if args.save_model:
        save_inference_model(methods, args.save_model, BASE)

    # ==================== Step 5: 汇总对比 ====================
    print("\n" + "=" * 60)
    print("Step 5/5: 汇总对比")
    print("=" * 60)

    print("\n### 全测试集")
    table = format_metrics_table(results)
    print(table)

    if cat_results:
        print("\n### 按文献类别")
        cat_table = format_category_table(cat_results)
        print(cat_table)

    # 保存结果
    output_path = config.OUTPUT_DIR / "results.json"
    save_results(results, output_path)
    print(f"\n结果已保存到: {output_path}")

    if cat_results:
        cat_output_path = config.OUTPUT_DIR / "results_by_category.json"
        import json
        with open(cat_output_path, "w", encoding="utf-8") as f:
            json.dump(cat_results, f, ensure_ascii=False, indent=2)
        print(f"分类结果已保存到: {cat_output_path}")

    # 保存 Markdown 表格
    md_path = config.OUTPUT_DIR / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 分词+词性标注评估结果\n\n")
        f.write(f"训练集: {len(dataset.train_words):,} 句 | ")
        f.write(f"测试集: {len(dataset.test_words):,} 句 | ")
        f.write(f"seed: {config.RANDOM_SEED}\n\n")
        f.write("## 全测试集\n\n")
        f.write(table)
        if cat_results:
            f.write("\n\n## 按文献类别\n\n")
            f.write(cat_table)
    print(f"Markdown 表格已保存到: {md_path}")

    # 结束全局计时
    run_timer.__exit__(None, None, None)
    total_elapsed = run_timer.elapsed

    # 追加运行日志到 JSONL
    log_path = config.OUTPUT_DIR / "run_log.jsonl"
    methods_config = {
        "dict": {"mode": config.DICT_MATCH_MODE},
        "crf": {k: v for k, v in config.CRF_PARAMS.items()},
        "bilstm_crf": {k: v for k, v in config.BILSTM_CRF_PARAMS.items()
                       if k not in ("device",)},
    }
    append_run_log(
        log_path=log_path,
        corpus_path=config.CORPUS_PATH,
        max_sentences=config.MAX_SENTENCES,
        methods_config=methods_config,
        results=results,
        elapsed=total_elapsed,
    )
    print(f"运行日志已追加到: {log_path}")


if __name__ == "__main__":
    main()
