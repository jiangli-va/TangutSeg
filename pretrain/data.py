"""预训练数据准备 —— UUID 划分、词表构建、MLM 数据集。"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


# ============================================================
# 西夏文字符判断 (复用 unlabeled_stats 中的 Unicode 范围)
# ============================================================

_TANGUT_RANGES = [
    (0x17000, 0x187FF),   # Tangut
    (0x18800, 0x18AFF),   # Tangut Components
    (0x18D00, 0x18D8F),   # Tangut Supplement
]

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]"]


def is_tangut(ch: str) -> bool:
    """判断是否是西夏文字符。"""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _TANGUT_RANGES)


def split_tangut_segments(text: str) -> List[str]:
    """以所有非西夏文字符为分隔符，得到纯西夏文片段列表。"""
    segments = []
    buf = []
    for ch in text:
        if is_tangut(ch):
            buf.append(ch)
        else:
            if buf:
                segments.append("".join(buf))
                buf = []
    if buf:
        segments.append("".join(buf))
    return segments


# ============================================================
# 词表构建
# ============================================================

def build_vocab(
    lexicon_path: str,
    pretrain_json_path: str,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """构建字符词表。

    Returns:
        char2idx: 字符 → 索引
        idx2char: 索引 → 字符
    """
    char_set: Set[str] = set()

    # 1. 特殊 token
    for tok in SPECIAL_TOKENS:
        char_set.add(tok)

    # 2. 词典头字
    with open(lexicon_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    for entry in entries:
        head = entry.get("xixia_character", "").strip()
        if head and is_tangut(head):
            char_set.add(head)

    # 3. 词典 term_character 中的字符
    for entry in entries:
        for term in entry.get("term_list", []):
            word = term.get("term_character", "").strip()
            for ch in word:
                if is_tangut(ch):
                    char_set.add(ch)

    # 4. 四行对译中出现的字符
    with open(pretrain_json_path, "r", encoding="utf-8") as f:
        pretrain_data = json.load(f)
    for item in pretrain_data:
        for ch in item.get("originalText", ""):
            if is_tangut(ch):
                char_set.add(ch)

    # 排序：特殊 token 在最前面，其他按 Unicode 排序
    special_set = set(SPECIAL_TOKENS)
    tangut_chars = sorted([c for c in char_set if c not in special_set])
    all_chars = SPECIAL_TOKENS + tangut_chars

    char2idx = {ch: i for i, ch in enumerate(all_chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char


# ============================================================
# UUID 划分
# ============================================================

def split_uuids(
    pretrain_json_path: str,
    num_valid_uuids: int = 5,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """按 UUID 固定划分训练集和验证集。

    同一个 UUID 的图版不会分到两边。

    Returns:
        train_uuids, valid_uuids
    """
    with open(pretrain_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_uuids = sorted(set(item["uuid"] for item in data))
    rng = random.Random(seed)
    rng.shuffle(all_uuids)

    valid_uuids = all_uuids[:num_valid_uuids]
    train_uuids = all_uuids[num_valid_uuids:]
    return train_uuids, valid_uuids


# ============================================================
# 预训练数据加载 & 切块
# ============================================================

def load_pretrain_segments(
    pretrain_json_path: str,
    uuids: List[str],
    max_length: int = 128,
    min_length: int = 4,
) -> List[str]:
    """加载指定 UUID 的原文，提取纯西夏文片段并切块。

    - 以非西夏文字符为分隔
    - 丢弃长度 < min_length 的片段
    - 超过 max_length 的顺序切块（不重叠，不跨图版）
    - 不跨图版拼接

    Returns:
        List[str]: 每个元素是一个不超过 max_length 的纯西夏文字符串
    """
    with open(pretrain_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    uuid_set = set(uuids)
    chunks = []

    for item in data:
        if item["uuid"] not in uuid_set:
            continue
        segments = split_tangut_segments(item["originalText"])

        for seg in segments:
            if len(seg) < min_length:
                continue
            # 超长片段顺序切块
            for start in range(0, len(seg), max_length):
                chunk = seg[start:start + max_length]
                if len(chunk) >= min_length:
                    chunks.append(chunk)

    return chunks


# ============================================================
# MLM 数据集
# ============================================================

class MLMDataset(TorchDataset):
    """MLM 预训练数据集。

    每个 epoch 动态重新生成遮盖（训练集），验证集使用固定随机种子。
    """

    def __init__(
        self,
        chunks: List[str],
        char2idx: Dict[str, int],
        max_length: int = 128,
        mask_ratio: float = 0.15,
        span_ratio: float = 0.5,
        random_seed: Optional[int] = None,  # None = each epoch different
        device: Optional[torch.device] = None,
    ):
        self.chunks = chunks
        self.char2idx = char2idx
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.span_ratio = span_ratio  # 50% span masking
        self.random_seed = random_seed
        self._device = device

        self.pad_idx = char2idx["[PAD]"]
        self.mask_idx = char2idx["[MASK]"]
        self.unk_idx = char2idx["[UNK]"]
        self.cls_idx = char2idx["[CLS]"]
        self.sep_idx = char2idx["[SEP]"]

        # 预先编码所有 chunk（不含遮盖）
        self._encoded_chunks: List[List[int]] = []
        self._lengths: List[int] = []
        for chunk in chunks:
            ids = [char2idx.get(c, self.unk_idx) for c in chunk]
            # 截断到 max_length (实际 max_length 已切分，这里做安全截断)
            ids = ids[:max_length]
            self._encoded_chunks.append(ids)
            self._lengths.append(len(ids))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (input_ids, labels)。

        每次调用动态生成遮盖（训练集随机，验证集按 self.random_seed 固定）。
        """
        token_ids = self._encoded_chunks[idx][:]
        seq_len = len(token_ids)

        if self.random_seed is not None:
            rng = random.Random(self.random_seed + idx)
        else:
            rng = random.Random()

        # 决定哪些位置被遮盖
        num_to_mask = max(1, int(seq_len * self.mask_ratio))
        masked_positions = self._select_mask_positions(seq_len, num_to_mask, rng)

        # 应用 80/10/10 替换策略
        input_ids = token_ids[:]
        labels = [self.pad_idx] * seq_len  # pad_idx 表示忽略

        for pos in masked_positions:
            labels[pos] = token_ids[pos]  # 记录原始 token
            rand = rng.random()
            if rand < 0.8:
                input_ids[pos] = self.mask_idx
            elif rand < 0.9:
                # 随机替换为随机西夏字
                rand_idx = rng.randint(len(SPECIAL_TOKENS), len(self.char2idx) - 1)
                input_ids[pos] = rand_idx
            else:
                input_ids[pos] = token_ids[pos]  # 保持不变

        # Padding
        pad_len = self.max_length - seq_len
        if pad_len > 0:
            input_ids += [self.pad_idx] * pad_len
            labels += [self.pad_idx] * pad_len  # padding 位置忽略

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def _select_mask_positions(
        self, seq_len: int, num_to_mask: int, rng: random.Random,
    ) -> Set[int]:
        """混合选择遮盖位置：~50% 单字 + ~50% span(2-4字)。"""
        masked: Set[int] = set()
        remaining = list(range(seq_len))

        while len(masked) < num_to_mask and remaining:
            if rng.random() < self.span_ratio:
                # Span 遮盖: 2-4 字
                span_len = rng.choice([2, 3, 4])
                # 找一个合法起点（连续在 remaining 中）
                candidates = []
                for start in remaining:
                    span = list(range(start, min(start + span_len, seq_len)))
                    if all(p in remaining for p in span) and len(span) >= 2:
                        candidates.append(start)
                if candidates:
                    start = rng.choice(candidates)
                    span = list(range(start, min(start + span_len, seq_len)))
                    for p in span:
                        if p in remaining and len(masked) < num_to_mask:
                            masked.add(p)
                            remaining.remove(p)
                else:
                    # 无法找 span，退化为单字
                    pos = rng.choice(remaining)
                    masked.add(pos)
                    remaining.remove(pos)
            else:
                # 单字遮盖
                pos = rng.choice(remaining)
                masked.add(pos)
                remaining.remove(pos)

        return masked


def collate_mlm(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """MLM batch collator —— 简单 stack。所有句子已 padding 到相同长度。"""
    input_ids, labels = zip(*batch)
    return torch.stack(input_ids), torch.stack(labels)


# ============================================================
# Phase 2: 词感知预训练数据
# ============================================================

def build_word_vocab(
    dict_json_path: str,
    tongyin_json_path: str,
    char2idx: Dict[str, int],
    min_len: int = 2,
    max_len: int = 4,
) -> Dict[str, List[str]]:
    """构建词表: 字序列 → 覆盖该序列的词典词。

    来源:
        1. 西夏文词典.json → term_list[].term_character (多字词)
        2. 同音.json → 同音重校本.词 (高置信多字词)

    过滤:
        - 只保留 len ∈ [min_len, max_len] 的词
        - 只保留所有字符都在 char2idx 中的词
        - 排除纯单字

    Returns:
        word_vocab: Dict[str, List[str]]
            key = 初始字 (西夏文单字)
            value = 以该字开头的词典词列表
    """
    all_words: Set[str] = set()

    # 1. 从词典加载
    with open(dict_json_path, "r", encoding="utf-8") as f:
        dict_entries = json.load(f)
    for entry in dict_entries:
        for term in entry.get("term_list", []):
            word = term.get("term_character", "").strip()
            if min_len <= len(word) <= max_len:
                all_words.add(word)

    # 2. 从同音加载高置信词
    with open(tongyin_json_path, "r", encoding="utf-8") as f:
        ty_entries = json.load(f)
    for entry in ty_entries:
        # 同音重校本.词 字段
        word = entry.get("同音重校本", {}).get("词", "").strip()
        if word and min_len <= len(word) <= max_len:
            # 验证全是西夏字
            if all(is_tangut(ch) for ch in word):
                all_words.add(word)

    # 3. 按首字分组索引，过滤不在 char2idx 中的词
    word_vocab: Dict[str, List[str]] = {}
    skipped = 0
    for word in sorted(all_words):
        if not all(ch in char2idx for ch in word):
            skipped += 1
            continue
        first_char = word[0]
        if first_char not in word_vocab:
            word_vocab[first_char] = []
        word_vocab[first_char].append(word)

    print(f"  [WordVocab] {len(all_words)} unique words, "
          f"{skipped} skipped (OOV chars), "
          f"{sum(len(v) for v in word_vocab.values())} indexed")
    return word_vocab


def find_word_spans(
    text: str,
    word_vocab: Dict[str, List[str]],
) -> List[Tuple[int, int]]:
    """在文本中匹配词典词，返回 (start, end_inclusive) 列表。

    使用首字索引加速：对每个位置，只看以该字开头的候选词。
    使用 startswith 做快匹配，不建立 Trie（数据量不大）。

    Args:
        text: 纯西夏文字符串
        word_vocab: word_vocab
            key = 首字, value = 以该字开头的词列表

    Returns:
        spans: [(start, end_inclusive), ...]，按 start 位置排序
    """
    spans = []
    for i, ch in enumerate(text):
        if ch not in word_vocab:
            continue
        candidates = word_vocab[ch]
        for cand in candidates:
            end = i + len(cand) - 1
            if end < len(text) and text[i:end + 1] == cand:
                spans.append((i, end))
    return spans


def sample_negative_spans(
    seq_len: int,
    pos_spans: List[Tuple[int, int]],
    neg_pool: List[str],
    char2idx: Dict[str, int],
    rng: random.Random,
    num_neg_per_pos: int = 5,
) -> List[Tuple[int, int]]:
    """为每个正样本 span 采样负样本。

    策略 (混合):
        1. 50% 从同序列采样同长度的随机 span（不与任何正样本完全重叠）
        2. 50% 从预采样池中取不在词典中的 span

    Args:
        seq_len: 序列长度
        pos_spans: 正样本 span 列表
        neg_pool: 预采样负样本池（不在词典中的多字串）
        char2idx: 字符词表
        rng: 随机数生成器
        num_neg_per_pos: 每个正样本的负样本数

    Returns:
        neg_spans: [(start, end), ...] num_pos * num_neg_per_pos 个
    """
    pos_set = set(pos_spans)
    neg_spans = []

    for pos_start, pos_end in pos_spans:
        pos_len = pos_end - pos_start + 1
        for _ in range(num_neg_per_pos):
            if rng.random() < 0.5 and neg_pool:
                # 从负样本池随机选一个
                neg_str = rng.choice(neg_pool)
                # 用随机偏移嵌入序列
                max_start = seq_len - len(neg_str)
                if max_start > 0:
                    start = rng.randint(0, max_start)
                    end = start + len(neg_str) - 1
                    if (start, end) not in pos_set:
                        neg_spans.append((start, end))
                        continue

            # 从同序列采样同长度随机 span
            max_start = seq_len - pos_len
            if max_start <= 0:
                neg_spans.append((0, pos_len - 1))
                continue
            for _ in range(20):  # 最多尝试 20 次
                start = rng.randint(0, max_start)
                end = start + pos_len - 1
                if (start, end) not in pos_set:
                    neg_spans.append((start, end))
                    break
            else:
                # fallback: 任意同长度 span
                start = rng.randint(0, max_start)
                neg_spans.append((start, min(start + pos_len - 1, seq_len - 1)))

    return neg_spans


def build_neg_pool(
    chunks: List[str],
    word_vocab: Dict[str, List[str]],
    pool_size: int = 5000,
    min_len: int = 2,
    max_len: int = 4,
    seed: int = 42,
) -> List[str]:
    """构建负样本预采样池：从语料中提取不在词典中的多字串。

    Args:
        chunks: 预训练文本片段
        word_vocab: 词表
        pool_size: 池大小
        min_len, max_len: 负样本长度范围
        seed: 随机种子

    Returns:
        neg_pool: 负样本字符串列表
    """
    rng = random.Random(seed)
    neg_candidates = []

    # 采样一些 chunk，提取所有 n-gram
    sample_chunks = rng.sample(chunks, min(len(chunks), 500))

    for chunk in sample_chunks:
        for start in range(len(chunk)):
            for length in range(min_len, max_len + 1):
                end = start + length
                if end > len(chunk):
                    break
                ngram = chunk[start:end]
                neg_candidates.append(ngram)

    # 去重并排除词典词
    rng.shuffle(neg_candidates)
    neg_pool = []
    seen = set()
    for ng in neg_candidates:
        if ng in seen:
            continue
        seen.add(ng)
        # 检查是否在词典中
        first_char = ng[0]
        if first_char in word_vocab and ng in word_vocab[first_char]:
            continue
        neg_pool.append(ng)
        if len(neg_pool) >= pool_size:
            break

    print(f"  [NegPool] Built negative pool: {len(neg_pool)} candidates")
    return neg_pool


class WordRankingDataset(TorchDataset):
    """词排序数据集——用于 Phase 2 联合训练。

    每个样本返回:
        - MLM 的 (input_ids, labels)
        - 正样本 span 列表
        - 负样本 span 列表
    """

    def __init__(
        self,
        chunks: List[str],
        char2idx: Dict[str, int],
        word_vocab: Dict[str, List[str]],
        neg_pool: List[str],
        max_length: int = 128,
        mask_ratio: float = 0.15,
        span_ratio: float = 0.5,
        num_neg_per_pos: int = 5,
        random_seed: Optional[int] = None,
    ):
        self.chunks = chunks
        self.char2idx = char2idx
        self.word_vocab = word_vocab
        self.neg_pool = neg_pool
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.span_ratio = span_ratio
        self.num_neg_per_pos = num_neg_per_pos
        self.random_seed = random_seed

        self.pad_idx = char2idx["[PAD]"]
        self.mask_idx = char2idx["[MASK]"]
        self.unk_idx = char2idx["[UNK]"]

        # 预编码所有 chunk
        self._encoded_chunks: List[List[int]] = []
        self._lengths: List[int] = []
        for chunk in chunks:
            ids = [char2idx.get(c, self.unk_idx) for c in chunk]
            ids = ids[:max_length]
            self._encoded_chunks.append(ids)
            self._lengths.append(len(ids))

        # 预计算每个 chunk 的正样本 span
        self._pos_spans: List[List[Tuple[int, int]]] = []
        for i, chunk in enumerate(chunks):
            spans = find_word_spans(chunk[:max_length], word_vocab)
            self._pos_spans.append(spans)

        self._chunks_with_words = sum(1 for s in self._pos_spans if s)
        total_words = sum(len(s) for s in self._pos_spans)
        print(f"  [WordRanking] {len(chunks)} chunks, "
              f"{self._chunks_with_words} with words, "
              f"{total_words} total word spans")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        """返回 (input_ids, labels, pos_spans, neg_spans)。"""
        token_ids = self._encoded_chunks[idx][:]
        seq_len = len(token_ids)
        pos_spans_raw = self._pos_spans[idx]

        if self.random_seed is not None:
            rng = random.Random(self.random_seed + idx)
        else:
            rng = random.Random()

        # ---- MLM 遮盖 ----
        num_to_mask = max(1, int(seq_len * self.mask_ratio))
        masked_positions = self._select_mask_positions(seq_len, num_to_mask, rng)

        input_ids = token_ids[:]
        labels = [self.pad_idx] * seq_len
        for pos in masked_positions:
            labels[pos] = token_ids[pos]
            rand = rng.random()
            if rand < 0.8:
                input_ids[pos] = self.mask_idx
            elif rand < 0.9:
                rand_idx = rng.randint(len(SPECIAL_TOKENS), len(self.char2idx) - 1)
                input_ids[pos] = rand_idx

        # ---- Padding ----
        pad_len = self.max_length - seq_len
        if pad_len > 0:
            input_ids += [self.pad_idx] * pad_len
            labels += [self.pad_idx] * pad_len

        # ---- 负采样 ----
        neg_spans = sample_negative_spans(
            seq_len, pos_spans_raw, self.neg_pool, self.char2idx,
            rng, self.num_neg_per_pos,
        )

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            pos_spans_raw,
            neg_spans,
        )

    def _select_mask_positions(
        self, seq_len: int, num_to_mask: int, rng: random.Random,
    ) -> Set[int]:
        """混合选择遮盖位置：~50% 单字 + ~50% span(2-4字)。复用 MLM 逻辑。"""
        masked: Set[int] = set()
        remaining = list(range(seq_len))

        while len(masked) < num_to_mask and remaining:
            if rng.random() < self.span_ratio:
                span_len = rng.choice([2, 3, 4])
                candidates = []
                for start in remaining:
                    span = list(range(start, min(start + span_len, seq_len)))
                    if all(p in remaining for p in span) and len(span) >= 2:
                        candidates.append(start)
                if candidates:
                    start = rng.choice(candidates)
                    span = list(range(start, min(start + span_len, seq_len)))
                    for p in span:
                        if p in remaining and len(masked) < num_to_mask:
                            masked.add(p)
                            remaining.remove(p)
                else:
                    pos = rng.choice(remaining)
                    masked.add(pos)
                    remaining.remove(pos)
            else:
                pos = rng.choice(remaining)
                masked.add(pos)
                remaining.remove(pos)

        return masked


def collate_joint(batch):
    """联合训练 batch collator。

    返回:
        input_ids:    (B, max_len) Tensor
        labels:       (B, max_len) Tensor
        all_pos_spans: [(batch_idx, start, end), ...]
        all_neg_spans: [(batch_idx, start, end), ...]
        lengths:      (B,) Tensor (有效长度，不含 padding)
    """
    input_ids, labels, pos_spans_list, neg_spans_list = zip(*batch)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)

    # 计算有效长度 (不含 padding)
    lengths = torch.tensor([
        (ids != 0).sum().item() for ids in input_ids
    ], dtype=torch.long)

    # 展平 span 列表
    all_pos_spans = []
    all_neg_spans = []
    for batch_idx, (pos_spans, neg_spans) in enumerate(zip(pos_spans_list, neg_spans_list)):
        for start, end in pos_spans:
            all_pos_spans.append((batch_idx, start, end))
        for start, end in neg_spans:
            all_neg_spans.append((batch_idx, start, end))

    return input_ids, labels, all_pos_spans, all_neg_spans, lengths
