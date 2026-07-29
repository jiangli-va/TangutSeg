"""训练西夏字 Word2Vec (Skip-gram) 字符向量。

用法:
    python -m pretrain.train_w2v \
        --output output/pretrain/w2v_char_192.pt

输出:
    w2v_char_192.pt: 包含:
        - vectors: np.ndarray (vocab_size, 192)
        - char_to_idx: Dict[str, int]
        - idx_to_char: Dict[int, str]
        - config: 训练参数
        - type_coverage: 字符型覆盖率
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from pretrain.data import is_tangut, split_tangut_segments


# ============================================================
# 训练数据准备
# ============================================================

def prepare_w2v_sentences(pretrain_json_path: str) -> List[List[str]]:
    """从四行对译语料中提取字符序列（与 MLM 预训练数据一致）。

    处理流程 (与 load_pretrain_segments 一致):
        1. 以非西夏字为分隔符
        2. 每个纯西夏文片段转为字符列表
        3. 不跨图版拼接
        4. 不使用 [CLS]/[SEP]/[MASK]

    Returns:
        sentences: 每个元素是一个字符列表，如 ["𗀀", "𗀁", "𗀂"]
    """
    with open(pretrain_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = []
    for item in data:
        text = item.get("originalText", "")
        segments = split_tangut_segments(text)
        for seg in segments:
            chars = list(seg)
            if len(chars) >= 2:  # Word2Vec 需要至少 2 个 token 的上下文
                sentences.append(chars)

    return sentences


# ============================================================
# Word2Vec 训练
# ============================================================

def train_word2vec(
    sentences: List[List[str]],
    vector_size: int = 192,
    window: int = 5,
    min_count: int = 1,
    sg: int = 1,
    negative: int = 10,
    epochs: int = 30,
    workers: int = 1,
    seed: int = 42,
) -> tuple:
    """训练 Gensim Word2Vec 模型。

    Returns:
        (model, char_to_idx, idx_to_char)
    """
    from gensim.models import Word2Vec

    print(f"[W2V] Training on {len(sentences):,} sentences, "
          f"vector_size={vector_size}, window={window}, sg={sg}, "
          f"negative={negative}, epochs={epochs}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        epochs=epochs,
        workers=workers,
        seed=seed,
    )

    # 构建词表映射
    char_to_idx = {}
    idx_to_char = {}
    for i, word in enumerate(model.wv.index_to_key):
        char_to_idx[word] = i
        idx_to_char[i] = word

    print(f"[W2V] Vocabulary size: {len(char_to_idx)}")
    return model, char_to_idx, idx_to_char


# ============================================================
# 覆盖分析
# ============================================================

def compute_coverage(
    w2v_char_to_idx: Dict[str, int],
    tangut_char2idx: Dict[str, int],
    downstream_corpus_path: str = None,
) -> dict:
    """计算 Word2Vec 向量对 TangutEncoder 词表和下游语料的覆盖率。

    Returns:
        dict with keys:
            type_coverage: float
            w2v_chars: int
            tangut_chars: int
            missing_chars: List[str] (前20个)
            token_coverage: float (如果有下游语料)
    """
    w2v_set = set(w2v_char_to_idx.keys())

    # 排除特殊 token
    special = {"[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]"}
    tangut_set = set(tangut_char2idx.keys()) - special

    intersection = w2v_set & tangut_set
    type_coverage = len(intersection) / max(len(tangut_set), 1)

    missing = sorted(tangut_set - w2v_set)
    missing_sample = missing[:20]

    result = {
        "type_coverage": type_coverage,
        "w2v_chars": len(w2v_set),
        "tangut_chars": len(tangut_set),
        "covered_chars": len(intersection),
        "missing_count": len(missing),
        "missing_sample": missing_sample,
    }

    # Token coverage (在下游语料上的覆盖率)
    if downstream_corpus_path is not None:
        total_tokens = 0
        covered_tokens = 0
        token_counts = {}
        try:
            with open(downstream_corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    for ch in line:
                        if is_tangut(ch):
                            total_tokens += 1
                            token_counts[ch] = token_counts.get(ch, 0) + 1
                            if ch in w2v_set:
                                covered_tokens += 1
            result["token_coverage"] = covered_tokens / max(total_tokens, 1)
            result["total_tokens"] = total_tokens
            result["covered_tokens"] = covered_tokens

            # 按频次排序，找出缺失的最高频字符
            missing_sorted = sorted(
                [(ch, token_counts.get(ch, 0)) for ch in missing],
                key=lambda x: -x[1],
            )
            result["top_missing_by_freq"] = missing_sorted[:10]
        except FileNotFoundError:
            result["token_coverage"] = None

    return result


# ============================================================
# 保存
# ============================================================

def save_w2v_model(
    model,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    coverage: dict,
    config: dict,
    output_path: str,
):
    """保存 Word2Vec 向量。兼容 TangutEncoder 加载。"""
    # 提取向量矩阵
    vectors = model.wv.vectors  # (vocab_size, vector_size)

    torch_save = {
        "vectors": vectors,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "config": config,
        "coverage": coverage,
    }
    import torch
    torch.save(torch_save, output_path)
    print(f"[W2V] Saved to {output_path}")
    print(f"[W2V] Type coverage: {coverage['type_coverage']:.2%}")
    if coverage.get("token_coverage") is not None:
        print(f"[W2V] Token coverage: {coverage['token_coverage']:.2%}")
    if coverage.get("missing_sample"):
        print(f"[W2V] Missing chars (sample): {coverage['missing_sample'][:10]}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Tangut character Word2Vec (Skip-gram)",
    )
    parser.add_argument("--output", type=str, required=True,
                        help="输出路径 (.pt)")
    parser.add_argument("--pretrain-json", type=str, default=None,
                        help="四行对译 JSON 路径 (default: corpus/提取四行对译中的西夏字...)")
    parser.add_argument("--vector-size", type=int, default=192,
                        help="向量维度 (default: 192)")
    parser.add_argument("--window", type=int, default=5,
                        help="窗口大小 (default: 5)")
    parser.add_argument("--negative", type=int, default=10,
                        help="负采样数 (default: 10)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="训练轮数 (default: 30)")
    parser.add_argument("--min-count", type=int, default=1,
                        help="最小出现次数 (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    parser.add_argument("--workers", type=int, default=1,
                        help="线程数 (default: 1)")
    parser.add_argument("--corpus", type=str, default=None,
                        help="下游分词语料路径 (计算 token coverage, optional)")

    args = parser.parse_args()

    # 默认路径
    pretrain_json = args.pretrain_json
    if pretrain_json is None:
        pretrain_json = str(BASE / "corpus" / "提取四行对译中的西夏字（以典籍的图片为单位）.json")

    # 1. 准备训练数据
    sentences = prepare_w2v_sentences(pretrain_json)
    print(f"[W2V] Extracted {len(sentences)} sentences from {pretrain_json}")

    # 2. 训练 Word2Vec
    w2v_config = {
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "sg": 1,  # Skip-gram
        "negative": args.negative,
        "epochs": args.epochs,
        "workers": args.workers,
        "seed": args.seed,
    }
    model, char_to_idx, idx_to_char = train_word2vec(
        sentences, **w2v_config,
    )

    # 3. 计算覆盖率 (需要 TangutEncoder 词表)
    tangut_char2idx = None
    # 尝试从 Phase 1 checkpoint 加载词表
    phase1_path = BASE / "output" / "pretrain" / "tangut_encoder_mixed" / "best_model0721.pt"
    if phase1_path.exists():
        import torch
        ckpt = torch.load(str(phase1_path), map_location="cpu", weights_only=False)
        tangut_char2idx = ckpt["char2idx"]
    else:
        # 回退：从预训练数据构建词表
        from pretrain.data import build_vocab
        lexicon_path = str(BASE / "corpus" / "西夏文词典.json")
        tangut_char2idx, _ = build_vocab(lexicon_path, pretrain_json)

    corpus_path = args.corpus
    if corpus_path is None:
        default_corpus = BASE / "corpus" / "all_cleaned.txt"
        if default_corpus.exists():
            corpus_path = str(default_corpus)

    coverage = compute_coverage(
        char_to_idx, tangut_char2idx,
        downstream_corpus_path=corpus_path,
    )

    # 4. 保存
    save_w2v_model(
        model, char_to_idx, idx_to_char, coverage,
        w2v_config, args.output,
    )


if __name__ == "__main__":
    main()
