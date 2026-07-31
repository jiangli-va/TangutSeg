"""
无标注语料统计特征提取器 —— 从四行对译 JSON 中提取 unigram/bigram 分布,
计算相邻字对的 log-freq、关联度量、左右邻接熵。

关联度量 (通过 bigram_metric 参数选择):
    - "dpmi":    折扣 PMI = max(0, log(cnt_ab - d) - log(cnt_a) - log(cnt_b) + log(N))
                  其中 d = 0.5，抑制低频噪声
    - "dice":     Dice = 2 * cnt_ab / (cnt_a + cnt_b)，值域 [0, 1]
    - "t_score":  t-score ≈ (cnt_ab - cnt_a*cnt_b/N) / sqrt(cnt_ab)，统计显著性

用法:
    extractor = UnlabeledStatsExtractor(bigram_metric="dpmi")
    extractor.load("corpus/提取四行对译中的西夏字（以典籍的图片为单位）.json")
    gap_vec = extractor.extract(sentence)  # (seq_len, 8) float32

8 维特征 (每位置):
    [0] 左间隙 log(1+bigram_count)
    [1] 左间隙 关联度量 (dPMI / Dice / t-score)
    [2] 左间隙 左邻接熵 (前字 H_right)
    [3] 左间隙 右邻接熵 (当前字 H_left)
    [4] 右间隙 log(1+bigram_count)
    [5] 右间隙 关联度量 (dPMI / Dice / t-score)
    [6] 右间隙 左邻接熵 (当前字 H_right)
    [7] 右间隙 右邻接熵 (后字 H_left)

gap_level 控制消融:
    0 = 不使用
    1 = 仅 freq (dim 0, 4)
    2 = freq + 关联度量 (dim 0,1,4,5)
    3 = freq + 关联度量 + entropy (全部 8 维)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

# 西夏文 Unicode 范围
_TANGUT_RANGES = [
    (0x17000, 0x187F7),   # Tangut
    (0x18800, 0x18AFF),   # Tangut Components
    (0x18D00, 0x18D8F),   # Tangut Supplement
]

# dPMI 折扣常数 — 从 bigram 计数中减去, 压制低频虚假高关联
_DPMI_DISCOUNT = 0.5

# 支持的关联度量
_SUPPORTED_METRICS = {"dpmi", "dice", "t_score"}

# 通用特征名 (关联度量列名统一为 "assoc", 由 bigram_metric 决定实际算法)
GAP_FEATURE_NAMES = [
    "L_freq", "L_assoc", "L_ent_prev", "L_ent_cur",
    "R_freq", "R_assoc", "R_ent_cur", "R_ent_next",
]

# 各消融级别使用的列索引
_GAP_LEVEL_INDICES = {
    0: [],
    1: [0, 4],              # freq only
    2: [0, 1, 4, 5],        # freq + 关联度量 (dPMI/Dice/t-score)
    3: list(range(8)),       # all 8
    4: [0, 2, 3, 4, 6, 7],  # freq + entropy (no assoc)
}


def _is_tangut(ch: str) -> bool:
    """判断是否是西夏文字符。"""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _TANGUT_RANGES)


def _split_tangut_segments(text: str) -> List[str]:
    """以所有非西夏文字符为分隔符，得到纯西夏文片段列表。"""
    segments = []
    buf = []
    for ch in text:
        if _is_tangut(ch):
            buf.append(ch)
        else:
            if buf:
                segments.append("".join(buf))
                buf = []
    if buf:
        segments.append("".join(buf))
    return segments


class UnlabeledStatsExtractor:
    """从无标注四行对译语料中提取 bigram 统计特征。

    Attributes:
        _bigram_freq: Dict[Tuple[str,str], int] — bigram 出现次数
        _unigram_freq: Counter — 单字出现次数
        _left_adj: Dict[str, Counter] — 每个字的右邻字分布 (left char → right char counts)
        _right_adj: Dict[str, Counter] — 每个字的左邻字分布 (right char → left char counts)
        _total_bigrams: int — bigram 总数
        _bigram_logfreq: Dict[Tuple[str,str], float] — log(1+count)
        _bigram_assoc: Dict[Tuple[str,str], float] — 当前激活的关联度量
        _char_ent_left: Dict[str, float] — 每个字的左邻接熵
        _char_ent_right: Dict[str, float] — 每个字的右邻接熵
    """

    def __init__(self, bigram_metric: str = "dpmi"):
        if bigram_metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"不支持的关联度量 '{bigram_metric}'，可选: {sorted(_SUPPORTED_METRICS)}"
            )
        self._bigram_metric = bigram_metric

        self._bigram_freq: Dict[Tuple[str, str], int] = {}
        self._unigram_freq: Counter = Counter()
        self._left_adj: Dict[str, Counter] = {}   # char → chars that follow it
        self._right_adj: Dict[str, Counter] = {}   # char → chars that precede it
        self._total_bigrams: int = 0

        # 缓存
        self._bigram_logfreq: Dict[Tuple[str, str], float] = {}
        self._bigram_assoc: Dict[Tuple[str, str], float] = {}
        self._char_ent_left: Dict[str, float] = {}   # H_left: 前面可以有什么字
        self._char_ent_right: Dict[str, float] = {}  # H_right: 后面可以有什么字

        self._loaded = False

    # ---------- 加载 ----------
    def load(self, json_path: str) -> None:
        """从四行对译 JSON 加载并计算所有统计量。

        Args:
            json_path: JSON 文件路径
        """
        with open(json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        # 第一遍: 收集所有纯西夏文片段
        all_segments: List[str] = []
        for entry in entries:
            text = entry.get("originalText", "")
            if not text:
                continue
            segments = _split_tangut_segments(text)
            all_segments.extend(segments)

        # 第二遍: 统计 unigram / bigram
        self._bigram_freq.clear()
        self._unigram_freq.clear()
        self._left_adj.clear()
        self._right_adj.clear()
        self._total_bigrams = 0

        for seg in all_segments:
            chars = list(seg)
            for ch in chars:
                self._unigram_freq[ch] += 1

            for i in range(len(chars) - 1):
                a, b = chars[i], chars[i + 1]
                bigram = (a, b)
                self._bigram_freq[bigram] = self._bigram_freq.get(bigram, 0) + 1
                self._total_bigrams += 1

                # 左邻接: a → b (a 后面跟什么)
                if a not in self._left_adj:
                    self._left_adj[a] = Counter()
                self._left_adj[a][b] += 1

                # 右邻接: b ← a (b 前面可以是什么)
                if b not in self._right_adj:
                    self._right_adj[b] = Counter()
                self._right_adj[b][a] += 1

        # 第三遍: 计算派生统计量
        self._compute_derived()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ---------- 序列化 ----------
    def save(self, path: str) -> None:
        """保存全部 bigram 统计到文件，可在推理服务中恢复。"""
        import pickle
        # tuple key → string key (JSON 不支持 tuple key)
        state = {
            "bigram_metric": self._bigram_metric,
            "bigram_freq": {"|".join(k): v for k, v in self._bigram_freq.items()},
            "unigram_freq": dict(self._unigram_freq),
            "left_adj": {k: dict(v) for k, v in self._left_adj.items()},
            "right_adj": {k: dict(v) for k, v in self._right_adj.items()},
            "total_bigrams": self._total_bigrams,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path: str) -> None:
        """从保存的文件恢复 bigram 统计状态并重新计算派生度量。"""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)

        self._bigram_metric = state["bigram_metric"]
        self._bigram_freq = {tuple(k.split("|")): v for k, v in state["bigram_freq"].items()}
        self._unigram_freq = Counter(state["unigram_freq"])
        self._left_adj = {k: Counter(v) for k, v in state["left_adj"].items()}
        self._right_adj = {k: Counter(v) for k, v in state["right_adj"].items()}
        self._total_bigrams = state["total_bigrams"]

        self._bigram_logfreq = {}
        self._bigram_assoc = {}
        self._char_ent_left = {}
        self._char_ent_right = {}
        self._compute_derived()
        self._loaded = True

    # ---------- 内部计算 ----------
    def _compute_derived(self) -> None:
        """预计算 log-freq, 关联度量 (三选一), 邻接熵。"""

        N = float(self._total_bigrams)

        # 1) log(1 + bigram_count)
        self._bigram_logfreq.clear()
        for bigram, cnt in self._bigram_freq.items():
            self._bigram_logfreq[bigram] = math.log(1 + cnt)

        # 2) 关联度量: 根据 self._bigram_metric 选择 dPMI / Dice / t-score
        self._bigram_assoc.clear()
        metric = self._bigram_metric

        if metric == "dpmi":
            # dPMI(a,b) = max(0, log(cnt_ab - d) - log(cnt_a) - log(cnt_b) + log(N))
            d = _DPMI_DISCOUNT
            log_N = math.log(N) if N > 0 else 0.0
            for (a, b), cnt_ab in self._bigram_freq.items():
                if cnt_ab <= d:
                    self._bigram_assoc[(a, b)] = 0.0
                else:
                    cnt_a = max(self._unigram_freq.get(a, 1), 1)
                    cnt_b = max(self._unigram_freq.get(b, 1), 1)
                    dpmi = math.log(cnt_ab - d) - math.log(cnt_a) - math.log(cnt_b) + log_N
                    self._bigram_assoc[(a, b)] = max(dpmi, 0.0)

        elif metric == "dice":
            # Dice(a,b) = 2 * cnt_ab / (cnt_a + cnt_b)
            for (a, b), cnt_ab in self._bigram_freq.items():
                cnt_a = max(self._unigram_freq.get(a, 1), 1)
                cnt_b = max(self._unigram_freq.get(b, 1), 1)
                dice = 2.0 * cnt_ab / (cnt_a + cnt_b)
                self._bigram_assoc[(a, b)] = dice

        elif metric == "t_score":
            # t-score ≈ (cnt_ab - E) / sqrt(cnt_ab)
            # E = cnt_a * cnt_b / N  (期望共现频次)
            if N == 0:
                for bigram in self._bigram_freq:
                    self._bigram_assoc[bigram] = 0.0
            else:
                for (a, b), cnt_ab in self._bigram_freq.items():
                    cnt_a = max(self._unigram_freq.get(a, 1), 1)
                    cnt_b = max(self._unigram_freq.get(b, 1), 1)
                    expected = cnt_a * cnt_b / N
                    if cnt_ab <= 0:
                        self._bigram_assoc[(a, b)] = 0.0
                    else:
                        t = (cnt_ab - expected) / math.sqrt(cnt_ab)
                        self._bigram_assoc[(a, b)] = t

        # 3) 邻接熵
        # H_left(c) = 一个字前面可以是什么字的熵 (基于 right_adj)
        self._char_ent_left.clear()
        for ch, neighbor_counts in self._right_adj.items():
            total = sum(neighbor_counts.values())
            ent = 0.0
            for cnt in neighbor_counts.values():
                p = cnt / total
                ent -= p * math.log(p)
            self._char_ent_left[ch] = ent

        # H_right(c) = 一个字后面可以是什么字的熵 (基于 left_adj)
        self._char_ent_right.clear()
        for ch, neighbor_counts in self._left_adj.items():
            total = sum(neighbor_counts.values())
            ent = 0.0
            for cnt in neighbor_counts.values():
                p = cnt / total
                ent -= p * math.log(p)
            self._char_ent_right[ch] = ent

    # ---------- 特征提取 ----------
    def extract(self, sentence: str, gap_level: int = 3) -> np.ndarray:
        """对单句提取 gap 特征。

        Args:
            sentence: 未分词文本
            gap_level: 0=不使用, 1=freq, 2=freq+Dice, 3=全部

        Returns:
            np.ndarray shape=(len(sentence), 8), dtype=float32
        """
        if not self._loaded:
            raise RuntimeError("请先调用 load() 加载语料")

        chars = list(sentence)
        n = len(chars)
        vec = np.zeros((n, 8), dtype=np.float32)
        if n == 0:
            return vec

        for i in range(n):
            # 左间隙: bigram (chars[i-1], chars[i])
            if i > 0:
                bigram_l = (chars[i - 1], chars[i])
                vec[i, 0] = self._bigram_logfreq.get(bigram_l, 0.0)
                vec[i, 1] = self._bigram_assoc.get(bigram_l, 0.0)
                vec[i, 2] = self._char_ent_right.get(chars[i - 1], 0.0)
                vec[i, 3] = self._char_ent_left.get(chars[i], 0.0)

            # 右间隙: bigram (chars[i], chars[i+1])
            if i < n - 1:
                bigram_r = (chars[i], chars[i + 1])
                vec[i, 4] = self._bigram_logfreq.get(bigram_r, 0.0)
                vec[i, 5] = self._bigram_assoc.get(bigram_r, 0.0)
                vec[i, 6] = self._char_ent_right.get(chars[i], 0.0)
                vec[i, 7] = self._char_ent_left.get(chars[i + 1], 0.0)

        return vec

    # ---------- 诊断 ----------
    def describe(self) -> str:
        """返回统计摘要。"""
        metric_names = {"dpmi": "dPMI (折扣PMI)", "dice": "Dice", "t_score": "t-score"}
        metric_label = metric_names.get(self._bigram_metric, self._bigram_metric)
        lines = [
            f"无标注语料统计 (关联度量: {metric_label}):",
            f"  unigram 种类: {len(self._unigram_freq):,}",
            f"  unigram 总数:  {sum(self._unigram_freq.values()):,}",
            f"  bigram 种类:  {len(self._bigram_freq):,}",
            f"  bigram 总数:  {self._total_bigrams:,}",
        ]
        # log-freq 范围
        if self._bigram_logfreq:
            vals = list(self._bigram_logfreq.values())
            lines.append(f"  log(1+freq):   min={min(vals):.3f} max={max(vals):.3f} mean={np.mean(vals):.3f}")
        # 关联度量范围
        if self._bigram_assoc:
            vals = list(self._bigram_assoc.values())
            if vals:
                pos_vals = [v for v in vals if v > 0]
                zero_ratio = (len(vals) - len(pos_vals)) / len(vals) if vals else 0
                lines.append(
                    f"  {metric_label}:         "
                    f"min={min(vals):.3f} max={max(vals):.3f} mean={np.mean(vals):.3f}  "
                    f"(zero: {zero_ratio:.1%})"
                )
        # 邻接熵范围
        if self._char_ent_left:
            vals = list(self._char_ent_left.values())
            lines.append(f"  H_left:        min={min(vals):.3f} max={max(vals):.3f} mean={np.mean(vals):.3f}")
        if self._char_ent_right:
            vals = list(self._char_ent_right.values())
            lines.append(f"  H_right:       min={min(vals):.3f} max={max(vals):.3f} mean={np.mean(vals):.3f}")

        return "\n".join(lines)
