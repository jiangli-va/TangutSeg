"""分词+词性标注器抽象基类 —— 所有方法统一定义 fit / predict 接口。

子类只需实现:
    fit(train_words, train_tags)  → None
    predict(sentence: str)         → Tuple[List[str], List[str]]   (word_list, pos_list)

评估与对比完全依赖这两个接口，保证可替换性。
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class Segmenter(ABC):
    """分词+词性标注器抽象基类。"""

    name: str = "BaseSegmenter"

    @abstractmethod
    def fit(self, train_words: List[List[str]], train_tags: List[List[str]]) -> None:
        """用已标注数据训练。train_tags 为词性标注（如 "n", "v", "rr" 等）。"""
        ...

    @abstractmethod
    def predict(self, sentence: str) -> Tuple[List[str], List[str]]:
        """对单句进行分词+词性标注，返回 (词列表, 词性列表)。"""
        ...

    def predict_batch(self, sentences: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
        """批量分词+词性标注（默认逐句调用，子类可覆盖优化）。"""
        all_words, all_pos = [], []
        for s in sentences:
            w, p = self.predict(s)
            all_words.append(w)
            all_pos.append(p)
        return all_words, all_pos
