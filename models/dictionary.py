"""基于词典的分词器 —— 最大匹配法 (正向/逆向/双向) + 高频词性标注。

支持三种词典来源：
    - corpus:  从训练语料中提取词表（原方法）
    - predefined: 使用外部预定义词典（如西夏文词典.json）
    - combined: 训练语料 + 预定义词典取并集
"""

from typing import List, Set, Dict, Tuple, Optional
from collections import Counter
import json
from pathlib import Path
from models.base import Segmenter


# ======================== 外部词典加载 ========================

def load_predefined_dict(path: str) -> Set[str]:
    """从西夏文词典 JSON 中提取所有词条（单字头 + 多字词 term_list）。"""
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    vocab: Set[str] = set()
    for e in entries:
        ch = e.get("xixia_character", "").strip()
        if ch:
            vocab.add(ch)
        for t in e.get("term_list", []):
            tc = t.get("term_character", "").strip()
            if len(tc) > 1:
                vocab.add(tc)
    return vocab


# ======================== 分词器 ========================

class DictionarySegmenter(Segmenter):
    """基于词典的最大匹配分词器 + 词频最高词性标注。

    方法:
        - 支持 fmm / bmm / bidirectional
        - dict_source: "corpus" | "predefined" | "combined"
        - 词性标注策略：查词典取该词在训练集中出现次数最多的词性；OOV 标为 "x"
    """

    name = "Dictionary"

    def __init__(self, mode: str = "bidirectional",
                 dict_source: str = "corpus",
                 predefined_path: Optional[str] = None):
        """
        Args:
            mode: "fmm" | "bmm" | "bidirectional"
            dict_source: "corpus" | "predefined" | "combined"
            predefined_path: predefined 模式下外部词典 JSON 路径
        """
        super().__init__()
        if mode not in ("fmm", "bmm", "bidirectional"):
            raise ValueError(f"Unknown mode: {mode}")
        if dict_source not in ("corpus", "predefined", "combined"):
            raise ValueError(f"Unknown dict_source: {dict_source}")
        self._mode = mode
        self._dict_source = dict_source
        self._predefined_path = predefined_path

        # 预定义词典（predefined / combined 模式下使用），fit 时延迟加载
        self._predefined_vocab: Optional[Set[str]] = None
        if dict_source in ("predefined", "combined"):
            if predefined_path is None:
                raise ValueError(f"dict_source={dict_source} 需要提供 predefined_path")
            self._predefined_vocab = load_predefined_dict(predefined_path)

        self._dict: Set[str] = set()
        self._max_word_len: int = 0
        self._word_pos: Dict[str, str] = {}   # 词 -> 最常见词性

    # ---------- 训练 ----------
    def fit(self, train_words: List[List[str]],
            train_tags: List[List[str]] = None) -> None:
        """从训练集提取词表作为词典，并记录每个词的高频词性。

        词典来源 (self._dict_source):
            - "corpus":      仅用训练语料中出现的词
            - "predefined":  仅用预定义外部词典
            - "combined":    两者取并集
        """
        # 训练语料词表 + 词性
        corpus_dict: Set[str] = set()
        corpus_max_len = 0
        pos_counter: Dict[str, Counter] = {}

        for words, tags in zip(train_words, train_tags or []):
            for w, t in zip(words, tags):
                corpus_dict.add(w)
                if len(w) > corpus_max_len:
                    corpus_max_len = len(w)
                if w not in pos_counter:
                    pos_counter[w] = Counter()
                pos_counter[w][t] += 1

        # 根据 dict_source 确定最终词典
        if self._dict_source == "corpus":
            self._dict = corpus_dict
        elif self._dict_source == "predefined":
            self._dict = set(self._predefined_vocab) if self._predefined_vocab else set()
        else:  # combined
            self._dict = corpus_dict | (self._predefined_vocab or set())

        # 词性标注：预定义词典中语料未见的词标 "x"
        if train_tags is None:
            for w in self._dict:
                self._word_pos[w] = "x"
        else:
            for w, counter in pos_counter.items():
                self._word_pos[w] = counter.most_common(1)[0][0]
            for w in self._dict:
                if w not in self._word_pos:
                    self._word_pos[w] = "x"

        self._max_word_len = max(
            (len(w) for w in self._dict),
            default=corpus_max_len,
        )

        # 添加单字符保底
        for ch in "0123456789abcdefghijklmnopqrstuvwxyz":
            self._dict.add(ch)
            if ch not in self._word_pos:
                self._word_pos[ch] = "x"

    # ---------- 预测 ----------
    def predict(self, sentence: str) -> Tuple[List[str], List[str]]:
        """对句子分词 + 词性标注。"""
        words = self._segment(sentence)
        pos = [self._word_pos.get(w, "x") for w in words]
        return words, pos

    def _segment(self, sentence: str) -> List[str]:
        """仅分词（内部用）。"""
        if self._mode == "fmm":
            return self._fmm(sentence)
        elif self._mode == "bmm":
            return self._bmm(sentence)
        else:
            return self._bidirectional(sentence)

    # ---------- 内部算法 ----------
    def _fmm(self, text: str) -> List[str]:
        """正向最大匹配。"""
        words = []
        i = 0
        n = len(text)
        while i < n:
            max_len = min(self._max_word_len, n - i)
            found = False
            for length in range(max_len, 0, -1):
                candidate = text[i:i + length]
                if candidate in self._dict:
                    words.append(candidate)
                    i += length
                    found = True
                    break
            if not found:
                words.append(text[i])
                i += 1
        return words

    def _bmm(self, text: str) -> List[str]:
        """逆向最大匹配。"""
        words = []
        n = len(text)
        i = n
        while i > 0:
            max_len = min(self._max_word_len, i)
            found = False
            for length in range(max_len, 0, -1):
                candidate = text[i - length:i]
                if candidate in self._dict:
                    words.append(candidate)
                    i -= length
                    found = True
                    break
            if not found:
                words.append(text[i - 1])
                i -= 1
        words.reverse()
        return words

    def _bidirectional(self, text: str) -> List[str]:
        """双向最大匹配 —— 正向与逆向结果不同时，选词数更少/单字词更少的。"""
        f_words = self._fmm(text)
        b_words = self._bmm(text)
        if f_words == b_words:
            return f_words
        def _score(ws):
            return sum(1 for w in ws if w not in self._dict), -len(ws)
        return f_words if _score(f_words) <= _score(b_words) else b_words
