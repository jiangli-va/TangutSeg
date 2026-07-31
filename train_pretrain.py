"""TangutEncoder 预训练 + 下游 BIES-CRF 分词 实验入口。

用法:
    # 1. 预训练 MLM
    python run_pretrain.py --pretrain --mask-mode mixed

    # 2. 预训练 MLM (single mask only)
    python run_pretrain.py --pretrain --mask-mode single

    # 3. 训练 Word2Vec 字符向量
    python run_pretrain.py --train-w2v --w2v-model output/pretrain/w2v_char_192.pt

    # 4. 下游分词: 预训练 vs 随机 vs Word2Vec 对照 (5折CV)
    python run_pretrain.py --seg --cv --pretrained-model ... --w2v-model ...

    # 5. 全部一步完成 (W2V + MLM + 下游)
    python run_pretrain.py --train-w2v --pretrain --seg --cv

    # 6. 已保存最优模型：
    --pretrained-model output/pretrain/tangut_encoder_mixed/best_model.pt

实验对照 (自动同时运行):
    TEnc-Random        随机初始化 Transformer + BIES-CRF
    TEnc-Char2Vec      Word2Vec 字符向量初始化 + BIES-CRF (新增)
    TEnc-MLM           预训练 TangutEncoder + BIES-CRF
    TEnc-Random+dict   随机初始化 + 17维词典特征
    TEnc-MLM+dict      预训练 + 17维词典特征
    TEnc-MLM+dict+gap  预训练 + 词典(17维) + gap(8维)
    CRF-U              当前最优 CRF (full + freq + 关联度量 + entropy)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
from data.dataset import split_dataset, make_kfolds
from evaluation.metrics import evaluate
from utils.helpers import (
    format_metrics_table, format_category_table,
    save_results,
    aggregate_cv_results, format_cv_table,
    aggregate_cv_category_results, format_cv_category_table,
    Timer,
)
from models.base import Segmenter

# ============================================================
# 路径配置
# ============================================================

OUTPUT_DIR = BASE / "output"
PRETRAIN_DIR = OUTPUT_DIR / "pretrain"
PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

LEXICON_PATH = str(BASE / "corpus" / "西夏文词典.json")
PRETRAIN_JSON_PATH = str(BASE / "corpus" / "提取四行对译中的西夏字（以典籍的图片为单位）.json")


# ============================================================
# 构建实验方法
# ============================================================

def build_pretrain_methods(
    pretrained_path: Optional[str] = None,
    lexicon_extractor=None,
    w2v_path: Optional[str] = None,
) -> Dict[str, Segmenter]:
    """构建下游分词对照实验方法。

    包含:
        TEnc-Random, TEnc-Char2Vec, TEnc-MLM,
        TEnc-Random+dict, TEnc-MLM+dict, TEnc-MLM+dict+gap,
        CRF-U
    """
    from pretrain.segmenter import TangutEncoderSegmenter
    from models.crf import CRFSegmenter
    from models.unlabeled_stats import UnlabeledStatsExtractor

    methods = {}

    # ---- 共享的 UnlabeledStatsExtractor (CRF-U 和 TEnc gap 特征需要) ----
    shared_unlabeled = None
    if Path(PRETRAIN_JSON_PATH).exists():
        shared_unlabeled = UnlabeledStatsExtractor(bigram_metric=config.BIGRAM_METRIC)
        shared_unlabeled.load(PRETRAIN_JSON_PATH)

    # ---- TEnc-Random: 随机 Transformer, 无词典 ----
    methods["TEnc-Random"] = TangutEncoderSegmenter(
        random_encoder=True,
        dict_feature_level=0,
    )

    # ---- TEnc-Char2Vec: Word2Vec 字符向量初始化, 无词典 ----
    if w2v_path is not None:
        methods["TEnc-Char2Vec"] = TangutEncoderSegmenter(
            random_encoder=True,
            w2v_path=w2v_path,
            dict_feature_level=0,
        )

    # ---- TEnc-MLM: 预训练 Transformer, 无词典 ----
    if pretrained_path is not None:
        methods["TEnc-MLM"] = TangutEncoderSegmenter(
            pretrained_path=pretrained_path,
            dict_feature_level=0,
        )

    # ---- TEnc-Random+dict: 随机 Transformer, 17维词典特征 ----
    if lexicon_extractor is not None:
        methods["TEnc-Random+dict"] = TangutEncoderSegmenter(
            random_encoder=True,
            dict_feature_level=3,
            lexicon_extractor=lexicon_extractor,
        )

    # ---- TEnc-MLM+dict: 预训练 Transformer, 17维词典特征 ----
    if pretrained_path is not None and lexicon_extractor is not None:
        methods["TEnc-MLM+dict"] = TangutEncoderSegmenter(
            pretrained_path=pretrained_path,
            dict_feature_level=3,
            lexicon_extractor=lexicon_extractor,
        )

    # ---- TEnc-MLM+dict+gap: 预训练 + 词典(17维) + gap(8维) ----
    if pretrained_path is not None and lexicon_extractor is not None and shared_unlabeled is not None:
        methods["TEnc-MLM+dict+gap"] = TangutEncoderSegmenter(
            pretrained_path=pretrained_path,
            dict_feature_level=3,
            lexicon_extractor=lexicon_extractor,
            gap_feature_level=3,
            unlabeled_extractor=shared_unlabeled,
        )

    # ---- CRF-U: 当前最优 CRF ----
    if lexicon_extractor is not None and shared_unlabeled is not None:
        methods["CRF-U"] = CRFSegmenter(
            c1=config.CRF_PARAMS["c1"],
            c2=config.CRF_PARAMS["c2"],
            max_iterations=config.CRF_PARAMS["max_iterations"],
            all_possible_transitions=config.CRF_PARAMS["all_possible_transitions"],
            lexicon_extractor=lexicon_extractor,
            dict_feature_level=5,
            unlabeled_extractor=shared_unlabeled,
            gap_feature_level=3,
        )

    return methods


def filter_methods(methods: Dict[str, Segmenter], selected: Optional[str]) -> Dict[str, Segmenter]:
    """按逗号分隔的列表过滤实验方法。None 或空字符串表示全部保留。"""
    if not selected or not selected.strip():
        return methods
    keep = {s.strip() for s in selected.split(",") if s.strip()}
    unknown = keep - set(methods.keys())
    if unknown:
        print(f"[WARN] Unknown methods ignored: {','.join(sorted(unknown))}")
    return {k: v for k, v in methods.items() if k in keep}


# ============================================================
# 训练 & 评估 (复用现有模式)
# ============================================================

def _reset_seed(seed: int) -> None:
    """重置所有随机源，使每个方法从相同的 RNG 状态开始，与运行顺序解耦。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_eval_dataset(
    methods: Dict[str, Segmenter],
    dataset,
    verbose: bool = True,
    seed: int = config.RANDOM_SEED,
) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, Dict]]]:
    """在给定 dataset 上训练并评估所有方法。"""
    train_vocab: Set[str] = set()
    for ws in dataset.train_words:
        train_vocab.update(ws)

    test_categories = getattr(dataset, 'categories', [])
    has_categories = bool(test_categories) and len(test_categories) == len(dataset.test_words)

    results = {}
    cat_results = {}

    for name, segmenter in methods.items():
        if verbose:
            print(f"\n--- {name} ---")

        # 每个方法重置随机种子，使结果与方法数量/顺序无关
        _reset_seed(seed)

        with Timer(f"{name}.fit"):
            if hasattr(segmenter, '_early_stop_patience'):
                # TangutEncoderSegmenter: 传入 dev 集用于早停
                segmenter.fit(
                    dataset.train_words, dataset.train_tags,
                    dev_words=dataset.dev_words,
                    dev_tags=dataset.dev_tags,
                )
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

        metrics = evaluate(
            pred_words, dataset.test_words, train_vocab=train_vocab,
            pred_pos_list=pred_pos, gold_pos_list=dataset.test_tags,
        )
        if verbose:
            print(f"  评估: {metrics}")

        results[name] = {
            "P": metrics.precision, "R": metrics.recall, "F1": metrics.f1,
            "OOV-R": metrics.oov_recall, "IV-R": metrics.iv_recall,
            "OOV%": metrics.oov_rate, "POS-Acc": metrics.pos_accuracy,
        }

        if has_categories:
            cat_results[name] = evaluate_per_category(
                pred_words, dataset.test_words, dataset.categories,
                train_vocab=train_vocab,
            )

    return results, cat_results


def evaluate_per_category(
    pred_words_list, gold_words_list, categories, train_vocab,
) -> Dict[str, Dict]:
    """按类别分组评估。"""
    from collections import defaultdict
    cat_pred = defaultdict(list)
    cat_gold = defaultdict(list)
    for pw, gw, c in zip(pred_words_list, gold_words_list, categories):
        cat_pred[c].append(pw)
        cat_gold[c].append(gw)

    cat_metrics = {}
    for c in sorted(cat_pred.keys()):
        m = evaluate(cat_pred[c], cat_gold[c], train_vocab=train_vocab)
        cat_metrics[c] = {
            "P": m.precision, "R": m.recall, "F1": m.f1,
            "OOV-R": m.oov_recall, "IV-R": m.iv_recall,
            "OOV%": m.oov_rate,
        }
    return cat_metrics


# ============================================================
# 保存推理模型
# ============================================================

def _save_inference_model(methods, method_name, lexicon_extractor=None):
    """保存完整推理模型到 saved_models/ 目录 (委托到 utils.helpers)。"""
    from utils.helpers import save_inference_model as _save_impl
    _save_impl(methods, method_name, BASE, lexicon_extractor=lexicon_extractor)



# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TangutEncoder 预训练 + 下游分词实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_pretrain.py --pretrain --mask-mode mixed
  python train_pretrain.py --pretrain --mask-mode single
  python train_pretrain.py --phase2 --pretrained-model output/pretrain/tangut_encoder_mixed/best_model0721.pt
  python train_pretrain.py --phase2 --pretrained-model ... --phase2-max-steps 3000 --lambda-word 0.3
  python train_pretrain.py --phase2 --pretrained-model ... --seg --cv  # Phase 2 + 下游验证
  python train_pretrain.py --seg --pretrained-model output/pretrain/tangut_encoder_phase2/best_model.pt
  python train_pretrain.py --seg --max 500 --folds 3
        """,
    )
    parser.add_argument("--pretrain", action="store_true",
                        help="运行 MLM 预训练")
    parser.add_argument("--phase2", action="store_true",
                        help="运行 Phase 2 词感知预训练 (MLM + Word Ranking)")
    parser.add_argument("--seg", action="store_true",
                        help="运行下游分词实验")
    parser.add_argument("--mask-mode", choices=["mixed", "single"],
                        default="mixed",
                        help="MLM 遮盖方式 (default: mixed)")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="预训练最大步数 (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="预训练 batch size (default: 32)")
    parser.add_argument("--max", type=int, default=None,
                        help="下游分词语料截断 (default: 全部)")
    parser.add_argument("--cv", action="store_true",
                        help="使用 K 折交叉验证")
    parser.add_argument("--folds", type=int, default=5,
                        help="K 折数 (default: 5)")
    parser.add_argument("--pretrained-model", type=str, default=None,
                        help="指定预训练模型路径 (跳过预训练步骤)")
    parser.add_argument("--methods", type=str, default=None,
                        help="指定实验方法, 逗号分隔 (可选: TEnc-Random,TEnc-MLM,TEnc-Random+dict,TEnc-MLM+dict,TEnc-MLM+dict+gap,CRF-U). "
                             "默认全部运行")
    # Phase 2 参数
    parser.add_argument("--lambda-word", type=float, default=0.3,
                        help="Phase 2 词损失权重 (default: 0.3)")
    parser.add_argument("--lr-encoder", type=float, default=5e-5,
                        help="Phase 2 编码器学习率 (default: 5e-5)")
    parser.add_argument("--lr-span-head", type=float, default=3e-4,
                        help="Phase 2 Span head 学习率 (default: 3e-4)")
    parser.add_argument("--phase2-max-steps", type=int, default=3000,
                        help="Phase 2 最大步数 (default: 3000)")
    parser.add_argument("--word-warmup-steps", type=int, default=500,
                        help="Phase 2 词损失 warmup 步数 (default: 500)")
    parser.add_argument("--num-neg-per-pos", type=int, default=5,
                        help="Phase 2 每正样本负样本数 (default: 5)")
    parser.add_argument("--eval-interval", type=int, default=200,
                        help="Phase 2 评估间隔 (default: 200)")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Phase 2 早停耐心 (default: 5)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Phase 2 梯度裁剪 (default: 1.0)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Phase 2 权重衰减 (default: 0.01)")
    parser.add_argument("--phase2-output-dir", type=str, default=None,
                        help="Phase 2 输出目录 (default: output/pretrain/tangut_encoder_phase2)")
    parser.add_argument("--w2v-model", type=str, default=None,
                        help="Word2Vec 字符向量路径 (.pt), 启用 TEnc-Char2Vec 方法")
    parser.add_argument("--train-w2v", action="store_true",
                        help="在分词实验前先训练 Word2Vec 字符向量")
    parser.add_argument("--save-model", type=str, default=None,
                        help="保存指定方法的完整推理模型到 saved_models/ (e.g. TEnc-MLM+dict+gap). "
                             "包含编码器+特征投射+CRF+词表+特征提取器，可直接用于推理服务。")
    args = parser.parse_args()

    if not args.pretrain and not args.seg and not args.phase2 and not args.train_w2v:
        parser.print_help()
        return

    pretrained_model_path = args.pretrained_model
    w2v_model_path = args.w2v_model

    # ========================
    # Step -1: Word2Vec 字符向量训练
    # ========================
    if args.train_w2v:
        from pretrain.train_w2v import main as train_w2v_main
        import subprocess
        import sys as _sys
        
        w2v_output = w2v_model_path
        if w2v_output is None:
            w2v_output = str(PRETRAIN_DIR / "w2v_char_192.pt")
        
        # 使用 subprocess 调用 train_w2v 脚本
        w2v_cmd = [
            _sys.executable, "-m", "pretrain.train_w2v",
            "--output", w2v_output,
            "--vector-size", "192",
        ]
        w2v_result = subprocess.run(w2v_cmd, cwd=str(BASE))
        if w2v_result.returncode != 0:
            print("[ERROR] Word2Vec training failed")
            return
        w2v_model_path = w2v_output
        print(f"\n[W2V] Word2Vec model saved to {w2v_model_path}")

    # ========================
    # Step 0: Phase 2 词感知预训练
    # ========================
    if args.phase2:
        from pretrain.train_phase2 import train_phase2
        
        if pretrained_model_path is None:
            print("[ERROR] --phase2 requires --pretrained-model")
            return
        
        phase2_output_dir = args.phase2_output_dir
        if phase2_output_dir is None:
            phase2_output_dir = str(PRETRAIN_DIR / "tangut_encoder_phase2")
        
        print(f"\n{'=' * 60}")
        print(f"  Phase 2: Lexicon-aware Pre-training")
        print(f"  Output: {phase2_output_dir}")
        print(f"{'=' * 60}\n")
        
        train_phase2(
            pretrain_json_path=PRETRAIN_JSON_PATH,
            lexicon_path=LEXICON_PATH,
            tongyin_json_path=str(BASE / "corpus" / "同音.json"),
            pretrained_model_path=pretrained_model_path,
            output_dir=phase2_output_dir,
            config={
                "lambda_word": args.lambda_word,
                "lr_encoder": args.lr_encoder,
                "lr_span_head": args.lr_span_head,
                "max_steps": args.phase2_max_steps,
                "batch_size": args.batch_size,
                "word_warmup_steps": args.word_warmup_steps,
                "num_neg_per_pos": args.num_neg_per_pos,
                "eval_interval": args.eval_interval,
                "early_stop_patience": args.early_stop_patience,
                "grad_clip": args.grad_clip,
                "weight_decay": args.weight_decay,
            },
        )
        print(f"\n[Phase2] Saved best model to {phase2_output_dir}/best_model.pt")

        # 将 Phase 2 模型路径传给下游 --seg 使用
        pretrained_model_path = str(Path(phase2_output_dir) / "best_model.pt")

    # ========================
    # Step 1: MLM 预训练
    # ========================
    if args.pretrain:
        from pretrain.train_mlm import train_mlm

        mask_name = "mixed" if args.mask_mode == "mixed" else "single"
        output_dir = PRETRAIN_DIR / f"tangut_encoder_{mask_name}"

        print(f"\n{'=' * 60}")
        print(f"  Pre-training: TangutEncoder-S MLM ({args.mask_mode})")
        print(f"  Output: {output_dir}")
        print(f"{'=' * 60}\n")

        model, log = train_mlm(
            pretrain_json_path=PRETRAIN_JSON_PATH,
            lexicon_path=LEXICON_PATH,
            output_dir=str(output_dir),
            mask_mode=args.mask_mode,
            config={
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
            },
        )
        pretrained_model_path = str(output_dir / "best_model.pt")
        print(f"\n[PRETRAIN] Saved best model to {pretrained_model_path}")

    # ========================
    # Step 2: 下游分词
    # ========================
    if args.seg:
        print(f"\n{'=' * 60}")
        print(f"  Downstream: TangutEncoder + BIES-CRF segmentation")
        print(f"{'=' * 60}\n")

        # 加载词典特征提取器 (共享)
        from models.lexicon import LexiconFeatureExtractor
        shared_lexicon = LexiconFeatureExtractor()
        shared_lexicon.load(LEXICON_PATH)

        if args.cv:
            # K 折交叉验证
            corpus_path = str(BASE / "corpus" / "all_cleaned.txt")
            parser_corpus = CorpusParser(corpus_path)
            all_words, all_tags, all_cats = parser_corpus.parse_file_with_categories(
                max_sentences=args.max,
            )
            folds = make_kfolds(all_words, all_tags, k=args.folds, categories=all_cats)

            fold_results = []
            fold_cat_results = []
            for fold_idx, fold_dataset in enumerate(folds):
                print(f"\n{'=' * 50}")
                print(f"  Fold {fold_idx + 1}/{args.folds}")
                print(f"{'=' * 50}")

                methods = build_pretrain_methods(
                    pretrained_path=pretrained_model_path,
                    lexicon_extractor=shared_lexicon,
                    w2v_path=w2v_model_path,
                )
                methods = filter_methods(methods, args.methods)
                fold_res, fold_cat = train_eval_dataset(
                    methods, fold_dataset, verbose=True,
                    seed=config.RANDOM_SEED + fold_idx,
                )
                fold_results.append(fold_res)
                fold_cat_results.append(fold_cat)

                # 记录 fold 信息
                print(f"\n  Fold {fold_idx + 1} 结果:")
                print(format_metrics_table(fold_res))

            # 汇总 CV 结果
            print(f"\n{'=' * 60}")
            print(f"  Cross-Validation Results ({args.folds}-fold)")
            print(f"{'=' * 60}\n")

            agg_results = aggregate_cv_results(fold_results)
            print(format_cv_table(agg_results))

            # 分类别 CV 结果
            agg_cat = aggregate_cv_category_results(fold_cat_results)
            print(f"\n  Category Breakdown ({args.folds}-fold):")
            print(format_cv_category_table(agg_cat))

            save_results(agg_results, PRETRAIN_DIR / "cv_results.json")
            save_results(agg_cat, PRETRAIN_DIR / "cv_category_results.json")

            # ---- CV 模式: 用全量数据重新训练一版用于保存 ----
            if args.save_model:
                methods = build_pretrain_methods(
                    pretrained_path=pretrained_model_path,
                    lexicon_extractor=shared_lexicon,
                    w2v_path=w2v_model_path,
                )
                methods = filter_methods(methods, args.save_model)
                # 全量数据训练 (不做 train/dev/test 划分)
                full_dataset = split_dataset(all_words, all_tags, categories=all_cats)
                print(f"\n{'=' * 50}")
                print(f"  Retraining on full dataset for saving...")
                print(f"{'=' * 50}")
                train_eval_dataset(methods, full_dataset, verbose=True)
                _save_inference_model(
                    methods, args.save_model,
                    lexicon_extractor=shared_lexicon,
                )

        else:
            # 单次划分
            corpus_path = str(BASE / "corpus" / "all_cleaned.txt")
            parser_corpus = CorpusParser(corpus_path)
            all_words, all_tags, all_cats = parser_corpus.parse_file_with_categories(
                max_sentences=args.max,
            )
            dataset = split_dataset(all_words, all_tags, categories=all_cats)

            methods = build_pretrain_methods(
                pretrained_path=pretrained_model_path,
                lexicon_extractor=shared_lexicon,
                w2v_path=w2v_model_path,
            )
            methods = filter_methods(methods, args.methods)

            results, cat_results = train_eval_dataset(methods, dataset, verbose=True)

            print(f"\n{'=' * 60}")
            print(f"  Segmentation Results")
            print(f"{'=' * 60}\n")
            print(format_metrics_table(results))

            if cat_results:
                for name, cat_m in cat_results.items():
                    print(f"\n  {name} - Categories:")
                    print(format_category_table({name: cat_m}))

            save_results(results, PRETRAIN_DIR / "seg_results.json")

            # ---- 保存完整推理模型 (for deployment) ----
            if args.save_model:
                _save_inference_model(
                    methods, args.save_model,
                    lexicon_extractor=shared_lexicon,
                )

    print(f"\nDone. Outputs in {PRETRAIN_DIR}")


if __name__ == "__main__":
    main()
