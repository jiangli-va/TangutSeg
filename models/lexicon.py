from __future__ import annotations

"""
统一的词典格网特征提取器 (v2)。

组件:
    Trie 树:   加载西夏文词典, 支持 find_all(chars) → List[Match]
    LexiconFeatureExtractor:  从 Trie 匹配结果生成 20 维字符级特征向量

特征维度 (20 维):
    B2 B3 B4 B5P     (4)  该位置是否为某个候选词的 B, 按词长分桶
    I3 I4 I5P         (3)  该位置是否为某个候选词的 I, 按词长分桶 (I2 永不出现)
    E2 E3 E4 E5P     (4)  该位置是否为某个候选词的 E, 按词长分桶
    rel_seen_B        (1)  以该位置为起点的所有候选词中最大可靠度 (occ>0)
    rel_seen_I        (1)  覆盖该位置的所有候选词中最大可靠度 (occ>0)
    rel_seen_E        (1)  以该位置为终点的所有候选词中最大可靠度 (occ>0)
    rel_unseen_B      (1)  以该位置为起点的所有候选词中最大 p_g (occ=0)
    rel_unseen_I      (1)  覆盖该位置的所有候选词中最大 p_g (occ=0)
    rel_unseen_E      (1)  以该位置为终点的所有候选词中最大 p_g (occ=0)
    has_yi            (1)  该位置所在候选词有「義」项
    has_yin           (1)  该位置所在候选词有「音」项
    has_book_title    (1)  该位置所在候选词为书名

可靠度公式 (贝叶斯平滑):
    r(w) = (hit(w) + κ · p_g) / (occ(w) + κ)
    p_g: 词条所属词长桶 (2/3/4/5P) 的平均成词概率
    κ: 先验强度 (默认 5)

消融级别 (level → 列索引):
    0: []                          baseline
    1: [0..10]                     BIE × 词长 (11 维)
    2: [0..13]                     BIE + rel_seen (14 维)
    3: [0..16]                     BIE + rel_all = dict_core (17 维, 不含 meta)
    4: [0..10] + [17..19]          BIE + meta (14 维)
    5: [0..19]                     BIE + rel_all + meta = dict_full (20 维)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import Counter
import json
import numpy as np


# ======================== 常量定义 ========================

DICT_FEATURE_NAMES = [
    # BIE × 词长 (11 维, I2 已删除)
    "B2",  "B3",  "B4",  "B5P",
    "I3",  "I4",  "I5P",
    "E2",  "E3",  "E4",  "E5P",
    # 已观察可靠度 (3 维)
    "rel_seen_B", "rel_seen_I", "rel_seen_E",
    # 未观察类别先验 (3 维)
    "rel_unseen_B", "rel_unseen_I", "rel_unseen_E",
    # 元数据 (3 维)
    "has_yi", "has_yin", "has_book_title",
]

DICT_FEATURE_DIM = len(DICT_FEATURE_NAMES)  # 20

# B/I/E 各词长桶的列索引
_B_IDX = {"2": 0, "3": 1, "4": 2, "5P": 3}
_I_IDX = {"3": 4, "4": 5, "5P": 6}
_E_IDX = {"2": 7, "3": 8, "4": 9, "5P": 10}

# 可靠度列索引
_COL_REL_SEEN_B = 11
_COL_REL_SEEN_I = 12
_COL_REL_SEEN_E = 13
_COL_REL_UNSEEN_B = 14
_COL_REL_UNSEEN_I = 15
_COL_REL_UNSEEN_E = 16

# 元数据列索引
_COL_HAS_YI   = 17
_COL_HAS_YIN  = 18
_COL_HAS_BOOK_TITLE = 19

# 消融级别 → 使用的列索引列表
# 注: 外部代码 (crf.py) 使用 LEVEL_INDICES[level] 获取要注入的特征列
LEVEL_INDICES = {
    0: [],
    1: list(range(11)),                      # BIE (11 维)
    2: list(range(14)),                      # BIE + rel_seen (14 维)
    3: list(range(17)),                      # BIE + rel_all = dict_core (17 维)
    4: list(range(11)) + list(range(17, 20)),  # BIE + meta (14 维)
    5: list(range(20)),                      # BIE + rel_all + meta = dict_full (20 维)
}

# 归一化参数
_MAX_WORD_LEN = 5
_KAPPA = 5          # 贝叶斯平滑先验强度


def _length_bin(length: int) -> str:
    """将词长映射到分桶标签 2/3/4/5P。"""
    if length <= 2:
        return "2"
    elif length == 3:
        return "3"
    elif length == 4:
        return "4"
    else:
        return "5P"


# ======================== 可靠度数据结构 ========================

@dataclass
class ReliabilityInfo:
    """词条的可靠度信息。"""
    value: float      # 贝叶斯平滑后的 r(w)
    observed: bool    # occ(w) > 0
    occ: int          # 原始出现次数
    hit: int          # 原始命中次数
    p_g: float        # 类别先验


# ======================== 全局函数 ========================

def compute_class_priors(
    trie: "Trie",
    train_words: List[List[str]],
) -> Dict[str, float]:
    """按词长桶统计类别先验 p_g。

    p_g(桶) = sum(hit + 1) / sum(occ + 2)  for 该桶所有有匹配的词条。

    Args:
        trie: 词典 Trie
        train_words: 训练集 (句 → 词列表)

    Returns:
        {"2": p_g, "3": p_g, "4": p_g, "5P": p_g}
    """
    occ: Counter = Counter()
    hit: Counter = Counter()

    for words in train_words:
        sent = "".join(words)
        gold_spans: Dict[Tuple[int, int], str] = {}
        pos = 0
        for w in words:
            gold_spans[(pos, pos + len(w))] = w
            pos += len(w)

        matches = trie.find_all(list(sent))
        for m in matches:
            occ[m.word] += 1
            span = (m.start, m.end)
            if span in gold_spans and gold_spans[span] == m.word:
                hit[m.word] += 1

    # 按词长桶聚合 hit/occ
    bin_hit: Counter = Counter()
    bin_occ: Counter = Counter()
    for word in occ:
        lbin = _length_bin(len(word))
        bin_hit[lbin] += hit.get(word, 0)
        bin_occ[lbin] += occ[word]

    # 计算每个桶的先验 (平滑)
    priors = {}
    for lbin in ["2", "3", "4", "5P"]:
        h = bin_hit.get(lbin, 0)
        o = bin_occ.get(lbin, 0)
        priors[lbin] = (h + 1) / (o + 2)

    return priors


def compute_lexicon_reliability(
    train_words: List[List[str]],
    trie: "Trie",
    kappa: int = _KAPPA,
) -> Dict[str, ReliabilityInfo]:
    """统计训练集中每个词典词条的 occ/hit, 用贝叶斯平滑计算 r(w)。

    r(w) = (hit(w) + κ · p_g) / (occ(w) + κ)

    Args:
        train_words: 训练集 (句 → 词列表)
        trie: 词典 Trie
        kappa: 贝叶斯平滑先验强度

    Returns:
        {词条: ReliabilityInfo}
    """
    # 1) 统计 occ/hit
    occ: Counter = Counter()
    hit: Counter = Counter()

    for words in train_words:
        sent = "".join(words)
        gold_spans: Dict[Tuple[int, int], str] = {}
        pos = 0
        for w in words:
            gold_spans[(pos, pos + len(w))] = w
            pos += len(w)

        matches = trie.find_all(list(sent))
        for m in matches:
            occ[m.word] += 1
            span = (m.start, m.end)
            if span in gold_spans and gold_spans[span] == m.word:
                hit[m.word] += 1

    # 2) 计算类别先验
    priors = compute_class_priors(trie, train_words)

    # 3) 贝叶斯平滑
    result: Dict[str, ReliabilityInfo] = {}
    for word, o in occ.items():
        h = hit.get(word, 0)
        p_g = priors[_length_bin(len(word))]
        value = (h + kappa * p_g) / (o + kappa)
        result[word] = ReliabilityInfo(
            value=value,
            observed=(o > 0),
            occ=o,
            hit=h,
            p_g=p_g,
        )

    return result


def _reliability_for_word(
    word: str,
    rel_info: Dict[str, ReliabilityInfo],
    priors: Dict[str, float],
) -> ReliabilityInfo:
    """为某词查找可靠度信息，若未录入则返回纯先验 fallback。"""
    if word in rel_info:
        return rel_info[word]
    lbin = _length_bin(len(word))
    p_g = priors.get(lbin, 0.5)
    return ReliabilityInfo(value=p_g, observed=False, occ=0, hit=0, p_g=p_g)


def compute_oof_dict_vectors(
    train_words: List[List[str]],
    trie: "Trie",
    extractor: "LexiconFeatureExtractor",
    inner_k: int = 5,
    seed: int = 42,
) -> List[np.ndarray]:
    """用 inner k-fold OOF 方式为训练集每句生成 20 维词典特征向量。

    Args:
        train_words: 训练集
        trie: 词典 Trie
        extractor: LexiconFeatureExtractor
        inner_k: 内部折数
        seed: 随机种子

    Returns:
        List[np.ndarray], len = len(train_words), 每个 shape=(句长, 20)
    """
    import random as _random
    n = len(train_words)
    indices = list(range(n))
    rng = _random.Random(seed)
    rng.shuffle(indices)

    fold_of = [0] * n
    for pos, idx in enumerate(indices):
        fold_of[idx] = pos % inner_k

    train_sents = ["".join(w) for w in train_words]
    oof_vecs: List[Optional[np.ndarray]] = [None] * n

    for f in range(inner_k):
        other_words = [train_words[i] for i in range(n) if fold_of[i] != f]
        rel = compute_lexicon_reliability(other_words, trie)
        priors = compute_class_priors(trie, other_words)
        extractor.set_class_priors(priors)
        for i in range(n):
            if fold_of[i] == f:
                oof_vecs[i] = extractor.extract(train_sents[i], reliability=rel)

    return oof_vecs  # type: ignore[return-value]


# ======================== Match / Trie ========================

@dataclass
class Match:
    """Trie 匹配结果。"""
    start: int       # 左闭
    end: int         # 右开
    word: str
    info: Dict       # 词条元信息


class _TrieNode:
    __slots__ = ("children", "word_info")

    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.word_info: Optional[Dict] = None


class Trie:
    """Trie 树: 插入多字词条, 在未分词文本中查找所有匹配。"""

    def __init__(self):
        self.root = _TrieNode()
        self._size = 0

    def insert(self, word: str, info: Dict) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        node.word_info = info
        self._size += 1

    def find_all(self, chars: List[str]) -> List[Match]:
        matches: List[Match] = []
        n = len(chars)
        for start in range(n):
            node = self.root
            for end in range(start, n):
                ch = chars[end]
                if ch not in node.children:
                    break
                node = node.children[ch]
                if node.word_info is not None:
                    matches.append(Match(
                        start=start,
                        end=end + 1,
                        word="".join(chars[start:end + 1]),
                        info=node.word_info,
                    ))
        return matches

    def __len__(self) -> int:
        return self._size

    def __contains__(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.word_info is not None


# ======================== 特征提取器 ========================

class LexiconFeatureExtractor:
    """从外部词典的 Trie 匹配结果提取 20 维字符级词典特征。

    用法:
        extractor = LexiconFeatureExtractor()
        extractor.load("corpus/西夏文词典.json")
        vectors = extractor.extract(sentence)           # (len(sentence), 20)
        extractor.set_reliability({...})                # 设置 OOF 可靠度
    """

    def __init__(self):
        self._trie: Optional[Trie] = None
        self._reliability: Dict[str, ReliabilityInfo] = {}
        self._class_priors: Dict[str, float] = {}
        self._loaded = False

    # ---------- 加载 ----------
    def load(self, dict_path: str) -> None:
        with open(dict_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        self._trie = Trie()
        for entry in entries:
            for term in entry.get("term_list", []):
                word = term.get("term_character", "").strip()
                if len(word) < 2:
                    continue
                info = self._parse_term_info(term)
                self._trie.insert(word, info)
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def trie(self) -> Trie:
        if self._trie is None:
            raise RuntimeError("请先调用 load() 加载词典")
        return self._trie

    # ---------- 可靠度 ----------
    def set_reliability(self, reliability: Dict[str, ReliabilityInfo]) -> None:
        """设置词条可靠度映射。"""
        self._reliability = reliability

    def set_class_priors(self, priors: Dict[str, float]) -> None:
        """设置按词长桶的类别先验 p_g。"""
        self._class_priors = priors

    @property
    def has_reliability(self) -> bool:
        return len(self._reliability) > 0

    # ---------- 特征提取 ----------
    def extract(
        self,
        sentence: str,
        reliability: Optional[Dict[str, ReliabilityInfo]] = None,
    ) -> np.ndarray:
        """对单句提取 20 维词典特征。

        Args:
            sentence: 未分词文本
            reliability: 可选, 临时可靠度映射 (用于 OOF 生成)

        Returns:
            np.ndarray shape=(len(sentence), 20), dtype=float32
        """
        if not self._loaded:
            raise RuntimeError("请先调用 load() 加载词典")

        rel_map = reliability if reliability is not None else self._reliability
        chars = list(sentence)
        n = len(chars)
        vectors = np.zeros((n, DICT_FEATURE_DIM), dtype=np.float32)
        if n == 0:
            return vectors

        # ---- 追踪每位置的最大可靠度 (区分 seen/unseen) ----
        # store raw (rel_value, length) for each position
        max_begin_s: List[float] = [-1.0] * n
        max_begin_u: List[float] = [-1.0] * n
        max_begin_u_len: List[int] = [2] * n  # fallback length for p_g lookup
        max_end_s: List[float] = [-1.0] * n
        max_end_u: List[float] = [-1.0] * n
        max_end_u_len: List[int] = [2] * n
        max_cover_s: List[float] = [-1.0] * n
        max_cover_u: List[float] = [-1.0] * n
        max_cover_u_len: List[int] = [2] * n

        matches = self._trie.find_all(chars)

        for m in matches:
            start, end = m.start, m.end
            length = end - start
            lbin = _length_bin(length)
            info = m.info

            # B/I/E × 词长 (multi-hot)
            vectors[start, _B_IDX[lbin]] = 1.0
            vectors[end - 1, _E_IDX[lbin]] = 1.0
            # I2 永不出现 (二字词没有内部位置)
            if lbin != "2":
                for i in range(start + 1, end - 1):
                    vectors[i, _I_IDX[lbin]] = 1.0

            # 元数据 (OR)
            for i in range(start, end):
                if info["has_yi"]:
                    vectors[i, _COL_HAS_YI] = 1.0
                if info["has_yin"]:
                    vectors[i, _COL_HAS_YIN] = 1.0
                if info["has_book_title"]:
                    vectors[i, _COL_HAS_BOOK_TITLE] = 1.0

            # 区分 seen/unseen, 取可靠度值
            occ_val = 0
            rel_value: float = 0.0
            if rel_map and m.word in rel_map:
                ri = rel_map[m.word]
                occ_val = ri.occ
                rel_value = ri.value
            else:
                # unseen: fallback to p_g by length bin
                rel_value = self._class_priors.get(lbin, 0.5)
            is_seen = (occ_val > 0)

            # begin
            if is_seen:
                if rel_value > max_begin_s[start]:
                    max_begin_s[start] = rel_value
            else:
                if rel_value > max_begin_u[start]:
                    max_begin_u[start] = rel_value
                    max_begin_u_len[start] = length

            # end
            ei = end - 1
            if is_seen:
                if rel_value > max_end_s[ei]:
                    max_end_s[ei] = rel_value
            else:
                if rel_value > max_end_u[ei]:
                    max_end_u[ei] = rel_value
                    max_end_u_len[ei] = length

            # cover (内部)
            for i in range(start, end):
                if is_seen:
                    if rel_value > max_cover_s[i]:
                        max_cover_s[i] = rel_value
                else:
                    if rel_value > max_cover_u[i]:
                        max_cover_u[i] = rel_value
                        max_cover_u_len[i] = length

        # ---- 第二遍: 写入可靠度 ----
        if rel_map:
            for i in range(n):
                if max_begin_s[i] >= 0:
                    vectors[i, _COL_REL_SEEN_B] = max_begin_s[i]
                if max_end_s[i] >= 0:
                    vectors[i, _COL_REL_SEEN_E] = max_end_s[i]
                if max_cover_s[i] >= 0:
                    vectors[i, _COL_REL_SEEN_I] = max_cover_s[i]

                if max_begin_u[i] >= 0:
                    vectors[i, _COL_REL_UNSEEN_B] = max_begin_u[i]
                if max_end_u[i] >= 0:
                    vectors[i, _COL_REL_UNSEEN_E] = max_end_u[i]
                if max_cover_u[i] >= 0:
                    vectors[i, _COL_REL_UNSEEN_I] = max_cover_u[i]

        return vectors

    def extract_batch(
        self,
        sentences: List[str],
        reliability: Optional[Dict[str, ReliabilityInfo]] = None,
    ) -> List[np.ndarray]:
        """批量提取 20 维词典特征。"""
        return [self.extract(s, reliability=reliability) for s in sentences]

    # ---------- 内部 ----------
    @staticmethod
    def _parse_term_info(term: Dict) -> Dict:
        """从词条 JSON 对象提取元信息。

        Returns:
            {"has_yi": bool, "has_yin": bool, "has_book_title": bool}
        """
        info = {
            "has_yi": False,
            "has_yin": False,
            "has_book_title": False,
        }

        for shiyi_entry in term.get("shiyi", []):
            t = shiyi_entry.get("type", "")
            text = shiyi_entry.get("shiyi", "")

            if t == "義":
                info["has_yi"] = True
            elif t == "音":
                info["has_yin"] = True

            if "书名" in text:
                info["has_book_title"] = True

        return info


# ======================== Internal-Only 词典格网 ========================

INTERNAL_BIE_DIM = 11  # B2 B3 B4 B5P  I3 I4 I5P  E2 E3 E4 E5P

# 与外部词典 BIE 桶索引完全一致
_INT_B_IDX = {"2": 0, "3": 1, "4": 2, "5P": 3}
_INT_I_IDX = {"3": 4, "4": 5, "5P": 6}
_INT_E_IDX = {"2": 7, "3": 8, "4": 9, "5P": 10}


def build_internal_only_trie(
    train_words: List[List[str]],
    external_trie: "Trie",
) -> "Trie":
    """构造 internal-only Trie: 训练集出现但外部词典没有的多字词。

    Args:
        train_words: 训练集 (句 → 词列表)
        external_trie: 外部词典 Trie

    Returns:
        包含 internal-only 词的 Trie
    """
    internal_trie = Trie()
    seen = set()
    for words in train_words:
        for w in words:
            if len(w) < 2:
                continue
            if w in seen:
                continue
            seen.add(w)
            if w not in external_trie:
                internal_trie.insert(w, {})
    return internal_trie


def extract_internal_bie(
    chars: List[str],
    trie: "Trie",
    vectors: np.ndarray,
    col_offset: int,
) -> None:
    """从 internal-only Trie 匹配结果提取 B/I/E 格网，写入 vectors[col_offset:col_offset+11]。

    仅提取 BIE × 词长 (multi-hot)，不涉及可靠度。
    """
    matches = trie.find_all(chars)
    for m in matches:
        start, end = m.start, m.end
        length = end - start
        lbin = _length_bin(length)

        vectors[start, col_offset + _INT_B_IDX[lbin]] = 1.0
        vectors[end - 1, col_offset + _INT_E_IDX[lbin]] = 1.0
        if lbin != "2":
            for i in range(start + 1, end - 1):
                vectors[i, col_offset + _INT_I_IDX[lbin]] = 1.0


def compute_oof_internal_vectors(
    train_words: List[List[str]],
    external_trie: "Trie",
    inner_k: int = 5,
    seed: int = 42,
) -> List[np.ndarray]:
    """用 inner k-fold OOF 方式为训练集每句生成 11 维 internal-only BIE 特征。

    inner fold 划分与 compute_oof_dict_vectors 一致。
    """
    import random as _random
    n = len(train_words)
    indices = list(range(n))
    rng = _random.Random(seed)
    rng.shuffle(indices)

    fold_of = [0] * n
    for pos, idx in enumerate(indices):
        fold_of[idx] = pos % inner_k

    oof_vecs: List[Optional[np.ndarray]] = [None] * n
    train_sents = ["".join(w) for w in train_words]

    for f in range(inner_k):
        other_words = [train_words[i] for i in range(n) if fold_of[i] != f]
        internal_trie = build_internal_only_trie(other_words, external_trie)
        for i in range(n):
            if fold_of[i] == f:
                chars = list(train_sents[i])
                vec = np.zeros((len(chars), INTERNAL_BIE_DIM), dtype=np.float32)
                extract_internal_bie(chars, internal_trie, vec, 0)
                oof_vecs[i] = vec

    return oof_vecs  # type: ignore[return-value]


# ======================== 领域分布向量 ========================

GENRE_DOMAIN_DIM = 2  # [经书比例, 世俗文献比例]


def compute_word_domain_distribution(
    train_words: List[List[str]],
    train_categories: List[str],
) -> Dict[str, Tuple[float, float]]:
    """统计训练语料中每个多字词的领域分布比例。

    对每个词条，统计它在"经书"和"世俗文献"中分别出现多少次，
    返回归一化比例。

    Args:
        train_words: 训练集 (句 → 词列表)
        train_categories: 每句的类别标签 ("经书" / "世俗文献")

    Returns:
        {word: (jingshu_ratio, shisu_ratio)}
        不在训练语料中的词不会出现在返回字典中。
    """
    jingshu: Counter = Counter()
    shisu: Counter = Counter()

    for words, cat in zip(train_words, train_categories):
        for w in words:
            if len(w) < 2:
                continue
            if cat == "经书":
                jingshu[w] += 1
            else:
                shisu[w] += 1

    result: Dict[str, Tuple[float, float]] = {}
    all_words = set(jingshu.keys()) | set(shisu.keys())
    for w in all_words:
        jc = jingshu.get(w, 0)
        sc = shisu.get(w, 0)
        total = jc + sc
        if total > 0:
            result[w] = (jc / total, sc / total)

    return result


def _global_domain_fallback(
    train_categories: List[str],
) -> Tuple[float, float]:
    """未登录词的 fallback: 全局经书/世俗文献句子比例。"""
    jc = sum(1 for c in train_categories if c == "经书")
    sc = len(train_categories) - jc
    total = max(jc + sc, 1)
    return (jc / total, sc / total)


def _sentence_domain_fallback(category: str) -> Tuple[float, float]:
    """按句子类别返回确定性 fallback。

    经书句无匹配位置 → [1.0, 0.0], 世俗文献句 → [0.0, 1.0]。
    相比全局 fallback（所有句都一样），这给了模型区分领域的信号。
    """
    if category == "经书":
        return (1.0, 0.0)
    else:
        return (0.0, 1.0)


def extract_domain_vec(
    chars: List[str],
    ext_matches: List[Match],
    domain_dist: Dict[str, Tuple[float, float]],
    fallback: Tuple[float, float],
    internal_trie: Optional["Trie"] = None,
) -> np.ndarray:
    """对每个字符位置，汇总所有匹配词的领域分布，返回 (n, 2) 向量。

    对每个位置，取所有覆盖该位置的匹配词的领域分布的平均值。
    如果没有匹配词覆盖该位置，使用 fallback。

    Args:
        chars: 字符列表
        ext_matches: 外部词典 Trie 的 find_all 结果
        domain_dist: 词 → (经书比例, 世俗文献比例)
        fallback: 无匹配时的默认值
        internal_trie: 可选, internal-only Trie (额外匹配)

    Returns:
        np.ndarray shape=(n, 2), dtype=float32
    """
    n = len(chars)
    vec = np.zeros((n, GENRE_DOMAIN_DIM), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)

    # 外部词典匹配
    for m in ext_matches:
        dist = domain_dist.get(m.word, fallback)
        for i in range(m.start, m.end):
            vec[i, 0] += dist[0]
            vec[i, 1] += dist[1]
            counts[i] += 1

    # internal-only 匹配
    if internal_trie is not None:
        int_matches = internal_trie.find_all(chars)
        for m in int_matches:
            dist = domain_dist.get(m.word, fallback)
            for i in range(m.start, m.end):
                vec[i, 0] += dist[0]
                vec[i, 1] += dist[1]
                counts[i] += 1

    # 取平均, 无匹配用 fallback
    for i in range(n):
        if counts[i] > 0:
            vec[i] /= counts[i]
        else:
            vec[i] = fallback

    return vec


def compute_oof_domain_vectors(
    train_words: List[List[str]],
    train_categories: List[str],
    ext_trie: "Trie",
    inner_k: int = 5,
    seed: int = 42,
) -> List[np.ndarray]:
    """OOF (out-of-fold) 方式生成领域分布向量, 防止信息泄露。

    与 compute_oof_internal_vectors 相同的 inner-k-fold 模式:
    把训练集分成 inner_k 折, 对每折用其余折统计词领域分布,
    然后用该分布生成本折句子的领域向量。

    Args:
        train_words: 训练集 (句 → 词列表)
        train_categories: 每句的类别标签
        ext_trie: 外部词典 Trie
        inner_k: 内部折数
        seed: 随机种子

    Returns:
        List[np.ndarray], len = len(train_words), 每个 shape=(句长, 2)
    """
    import random as _random
    n = len(train_words)
    indices = list(range(n))
    rng = _random.Random(seed)
    rng.shuffle(indices)

    fold_of = [0] * n
    for pos, idx in enumerate(indices):
        fold_of[idx] = pos % inner_k

    oof_vecs: List[Optional[np.ndarray]] = [None] * n
    train_sents = ["".join(w) for w in train_words]

    for f in range(inner_k):
        other_idx = [i for i in range(n) if fold_of[i] != f]
        other_words_list = [train_words[i] for i in other_idx]
        other_cats = [train_categories[i] for i in other_idx]

        domain_dist = compute_word_domain_distribution(other_words_list, other_cats)
        int_trie = build_internal_only_trie(other_words_list, ext_trie)

        for i in range(n):
            if fold_of[i] == f:
                chars = list(train_sents[i])
                ext_matches = ext_trie.find_all(chars)
                # 逐句 fallback: 经书句无匹配位置用 [1,0], 世俗用 [0,1]
                sent_fallback = _sentence_domain_fallback(train_categories[i])
                oof_vecs[i] = extract_domain_vec(
                    chars, ext_matches, domain_dist, sent_fallback,
                    internal_trie=int_trie,
                )

    return oof_vecs  # type: ignore[return-value]
