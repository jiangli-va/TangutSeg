"""词汇表管理 —— 从语料构建词表，支持查询与扩展。"""

from collections import Counter
from typing import List, Tuple, Optional


class Vocabulary:
    """两层词表：char_vocab（字级） + word_vocab（词级）。

    字级供 CRF 特征提取使用；词级供词典法使用。
    """

    def __init__(self):
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._word_freq: Counter = Counter()

    # ---------- 构建 ----------
    def add_sentence(self, words: List[str]):
        """从一条已分词句子增量构建词表。"""
        for w in words:
            self._word_freq[w] += 1
            for ch in w:
                if ch not in self._char_to_id:
                    idx = len(self._char_to_id)
                    self._char_to_id[ch] = idx
                    self._id_to_char[idx] = ch

    def build_from_sentences(self, sentences: List[List[str]]):
        """从多句子批量构建。"""
        for words in sentences:
            self.add_sentence(words)

    # ---------- 查询 ----------
    @property
    def char_size(self) -> int:
        return len(self._char_to_id)

    @property
    def word_size(self) -> int:
        return len(self._word_freq)

    def char_id(self, ch: str) -> Optional[int]:
        return self._char_to_id.get(ch)

    def word_count(self, w: str) -> int:
        return self._word_freq.get(w, 0)

    def has_word(self, w: str) -> bool:
        return w in self._word_freq

    @property
    def word_set(self):
        return set(self._word_freq.keys())

    def top_words(self, n: int = 50) -> List[Tuple[str, int]]:
        return self._word_freq.most_common(n)
