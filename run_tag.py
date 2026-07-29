"""联合分词+词性标注模型入口 —— 使用 BI+POS 大标签集一次性完成分词和词性标注。

使用方法:
    python run_tag.py                         # 全部联合模型
    python run_tag.py --methods crf_joint5    # 只跑指定变体
    python run_tag.py --max 2000              # 截断语料（快速调试）
    python run_tag.py --cv                    # 启用 K 折交叉验证
    python run_tag.py --folds 5               # 指定折数

可用的联合模型:
    crf_joint0, crf_joint5          - CRF 联合标注 (无/全词典特征)
    crf_joint6, crf_joint7, crf_joint8  - CRF 联合标注 + gap 消融
    bilstm_joint0, bilstm_joint5    - BiLSTM-CRF 联合标注 (无/全词典特征)
    bilstm_joint6, bilstm_joint7, bilstm_joint8  - BiLSTM-CRF 联合标注 + gap 消融
"""

import sys
import argparse
from pathlib import Path
import random
from typing import Dict, List, Set, Optional, Tuple

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import config

# ---- 固定全局随机种子 ----
random.seed(config.RANDOM_SEED)
import numpy as np
np.random.seed(config.RANDOM_SEED)
import torch
torch.manual_seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
from data.corpus import CorpusParser
from data.dataset import split_dataset, make_kfolds, compute_vocabulary_stats
from evaluation.metrics import evaluate
from utils.helpers import (format_metrics_table, format_category_table,
                           save_results, append_run_log,
                           aggregate_cv_results, format_cv_table,
                           aggregate_cv_category_results,
                           format_cv_category_table, Timer)
from models.base import Segmenter


# ======================== 注册联合词性标注方法 ========================

def build_methods(which: Optional[List[str]] = None) -> Dict[str, 'Segmenter']:
    """构建联合分词+词性标注方法字典。

    只包含 tag_model 下的联合模型，不含纯分词模型。
    """
    all_methods = {}
    predefined_path = str(BASE / "corpus" / "西夏文词典.json")

    # ---- 共享的 UnlabeledStatsExtractor (gap features) ----
    gap_json_path = config.UNLABELED_JSON_PATH
    shared_unlabeled: Optional[object] = None
    if gap_json_path and Path(gap_json_path).exists():
        from models.unlabeled_stats import UnlabeledStatsExtractor
        shared_unlabeled = UnlabeledStatsExtractor(bigram_metric=config.BIGRAM_METRIC)
        shared_unlabeled.load(gap_json_path)

    # 当前关联度量名称
    _metric_label = {"dpmi": "dPMI", "dice": "Dice", "t_score": "t-score"}.get(
        config.BIGRAM_METRIC, config.BIGRAM_METRIC
    )

    # ---- CRF 联合标注 ----
    crf_joint_levels = {
        "crf_joint0": 0,
        "crf_joint5": 5,
        "crf_joint6": 6,
        "crf_joint7": 7,
        "crf_joint8": 8,
        "crf_joint": 5,   # 默认 = full
    }
    crf_joint_requested = which is None or any(
        k in crf_joint_levels and k in which for k in crf_joint_levels)
    if crf_joint_requested:
        from tag_model.crf_joint import CRFJointTagger
        from models.lexicon import LexiconFeatureExtractor
        joint_level_label = {
            0: "CRF-Joint-0",
            5: "CRF-Joint-full",
            6: f"CRF-Joint-full+freq",
            7: f"CRF-Joint-full+freq+{_metric_label}",
            8: f"CRF-Joint-full+freq+{_metric_label}+entropy",
        }
        joint_lexicon = None
        for key, level in crf_joint_levels.items():
            if which is None or key in which:
                if level > 0:
                    if joint_lexicon is None:
                        joint_lexicon = LexiconFeatureExtractor()
                        joint_lexicon.load(predefined_path)
                    use_lexicon = joint_lexicon
                else:
                    use_lexicon = None
                # gap feature: level 6-8 => gap_level = level - 5
                gap_lv = level - 5 if level >= 6 else 0
                use_gap_extractor = shared_unlabeled if gap_lv > 0 else None
                all_methods[joint_level_label[level]] = CRFJointTagger(
                    c1=config.CRF_PARAMS["c1"],
                    c2=config.CRF_PARAMS["c2"],
                    max_iterations=config.CRF_PARAMS["max_iterations"],
                    all_possible_transitions=config.CRF_PARAMS["all_possible_transitions"],
                    lexicon_extractor=use_lexicon,
                    dict_feature_level=level,
                    unlabeled_extractor=use_gap_extractor,
                    gap_feature_level=gap_lv,
                )

    # ---- BiLSTM-CRF 联合标注 ----
    bilstm_joint_levels = {
        "bilstm_joint0": 0,
        "bilstm_joint5": 5,
        "bilstm_joint6": 6,
        "bilstm_joint7": 7,
        "bilstm_joint8": 8,
        "bilstm_joint": 5,   # 默认 = full
    }
    bilstm_joint_requested = which is None or any(
        k in bilstm_joint_levels and k in which for k in bilstm_joint_levels)
    if bilstm_joint_requested:
        from tag_model.bilstm_crf_joint import BiLSTMCRFJointTagger
        from models.lexicon import LexiconFeatureExtractor
        bilstm_joint_label = {
            0: "BiLSTM-Joint-0",
            5: "BiLSTM-Joint-full",
            6: f"BiLSTM-Joint-full+freq",
            7: f"BiLSTM-Joint-full+freq+{_metric_label}",
            8: f"BiLSTM-Joint-full+freq+{_metric_label}+entropy",
        }
        bilstm_lexicon = None
        jingshu_loss_weight = config.BILSTM_CRF_PARAMS.get("jingshu_loss_weight", 1.0)
        for key, level in bilstm_joint_levels.items():
            if which is None or key in which:
                if level > 0:
                    if bilstm_lexicon is None:
                        bilstm_lexicon = LexiconFeatureExtractor()
                        bilstm_lexicon.load(predefined_path)
                    use_lexicon = bilstm_lexicon
                else:
                    use_lexicon = None
                # gap feature: level 6-8 => gap_level = level - 5
                gap_lv = level - 5 if level >= 6 else 0
                use_gap_extractor = shared_unlabeled if gap_lv > 0 else None
                all_methods[bilstm_joint_label[level]] = BiLSTMCRFJointTagger(
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
                    jingshu_loss_weight=jingshu_loss_weight,
                    unlabeled_extractor=use_gap_extractor,
                    gap_feature_level=gap_lv,
                )

    return all_methods


# ======================== 训练 & 评估 ========================

def train_eval_dataset(
    methods: Dict[str, 'Segmenter'],
    dataset,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, Dict]]]:
    """在给定 dataset 上训练并评估所有联合标注方法。"""
    train_vocab: Set[str] = set()
    for ws in dataset.train_words:
        train_vocab.update(ws)

    test_categories = getattr(dataset, 'categories', [])
    has_categories = bool(test_categories) and len(test_categories) == len(dataset.test_words)

    results: Dict[str, Dict] = {}
    cat_results: Dict[str, Dict[str, Dict]] = {}

    for name, segmenter in methods.items():
        if verbose:
            print(f"\n--- {name} ---")

        with Timer(f"{name}.fit"):
            # 联合模型传递 train/dev 数据给 fit
            if "BiLSTM" in name:
                train_cats = getattr(dataset, 'train_categories', None)
                dev_cats = getattr(dataset, 'dev_categories', None)
                segmenter.fit(dataset.train_words, dataset.train_tags,
                             dev_words=dataset.dev_words, dev_tags=dataset.dev_tags,
                             train_categories=train_cats,
                             dev_categories=dev_cats)
            else:
                segmenter.fit(dataset.train_words, dataset.train_tags)
        if verbose:
            print(f"  fit: 完成")

        with Timer(f"{name}.predict"):
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

        # ---- 按类别拆分 ----
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
    """K 折交叉验证。"""
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
            print("  [ERROR] 没有可用的联合标注方法，退出。")
            return
        res, cat_res = train_eval_dataset(methods, ds, verbose=True)
        fold_results.append(res)
        fold_cat_results.append(cat_res)
        print(f"\n[Fold {i+1} 结果]")
        print(format_metrics_table(res))
        if cat_res:
            print(format_category_table(cat_res))

    print("\n" + "=" * 60)
    print(f"Step 5/5: {k} 折汇总（均值 ± 标准差）")
    print("=" * 60)
    agg = aggregate_cv_results(fold_results)
    cv_table = format_cv_table(agg)
    print("\n### 全测试集")
    print(cv_table)

    if fold_cat_results and any(fold_cat_results):
        agg_cat = aggregate_cv_category_results(fold_cat_results)
        cat_cv_table = format_cv_category_table(agg_cat)
        print("\n### 按文献类别")
        print(cat_cv_table)

    output_path = config.OUTPUT_DIR / "results_tag_cv.json"
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "k": k,
            "seed": config.RANDOM_SEED,
            "fold_results": fold_results,
            "aggregate": agg,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")

    md_path = config.OUTPUT_DIR / "results_tag_cv.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 联合分词+词性标注 {k} 折交叉验证结果\n\n")
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
    log_path = config.OUTPUT_DIR / "run_log_tag.jsonl"
    methods_config = {
        "cv_folds": k,
        "crf_joint": {k2: v for k2, v in config.CRF_PARAMS.items()},
        "bilstm_crf_joint": {k2: v for k2, v in config.BILSTM_CRF_PARAMS.items()
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


# ======================== 主入口 ========================

def main():
    parser = argparse.ArgumentParser(description="联合分词+词性标注实验")
    parser.add_argument("--methods", type=str, default=None,
                        help="逗号分隔的方法列表, e.g. 'crf_joint5,bilstm_joint5'")
    parser.add_argument("--max", type=int, default=config.MAX_SENTENCES,
                        help="最大句子数（截断语料，快速调试）")
    parser.add_argument("--cv", action="store_true",
                        help="使用 K 折交叉验证，报告均值±标准差")
    parser.add_argument("--folds", type=int, default=5,
                        help="交叉验证折数（默认 5）")
    args = parser.parse_args()

    which = [m.strip() for m in args.methods.split(",")] if args.methods else None
    max_sentences = args.max

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
    print("Step 3/5: 初始化联合词性标注方法 ...")
    print("=" * 60)
    methods = build_methods(which)
    for name in methods:
        print(f"  -> {name} ✓")
    if not methods:
        print("  [ERROR] 没有可用的联合标注方法，退出。")
        return

    # ==================== Step 4/5: 训练 + 评估 ====================
    print("\n" + "=" * 60)
    print("Step 4/5: 训练 & 预测 ...")
    print("=" * 60)
    results, cat_results = train_eval_dataset(methods, dataset, verbose=True)

    # ==================== 汇总 ====================
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

    # 保存
    output_path = config.OUTPUT_DIR / "results_tag.json"
    save_results(results, output_path)
    print(f"\n结果已保存到: {output_path}")

    if cat_results:
        cat_output_path = config.OUTPUT_DIR / "results_tag_by_category.json"
        import json
        with open(cat_output_path, "w", encoding="utf-8") as f:
            json.dump(cat_results, f, ensure_ascii=False, indent=2)
        print(f"分类结果已保存到: {cat_output_path}")

    md_path = config.OUTPUT_DIR / "results_tag.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 联合分词+词性标注评估结果\n\n")
        f.write(f"训练集: {len(dataset.train_words):,} 句 | ")
        f.write(f"测试集: {len(dataset.test_words):,} 句 | ")
        f.write(f"seed: {config.RANDOM_SEED}\n\n")
        f.write("## 全测试集\n\n")
        f.write(table)
        if cat_results:
            f.write("\n\n## 按文献类别\n\n")
            f.write(cat_table)
    print(f"Markdown 表格已保存到: {md_path}")

    run_timer.__exit__(None, None, None)
    total_elapsed = run_timer.elapsed

    log_path = config.OUTPUT_DIR / "run_log_tag.jsonl"
    methods_config = {
        "crf_joint": {k: v for k, v in config.CRF_PARAMS.items()},
        "bilstm_crf_joint": {k: v for k, v in config.BILSTM_CRF_PARAMS.items()
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
