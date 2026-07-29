"""基于 CRF 的分词+词性标注器 —— 两阶段：CRF 分词 + 词典高频词性标注。

支持统一的词典格网特征 (LexiconFeatureExtractor v2):
    通过 dict_feature_level 控制消融:
        0 = 不使用词典特征 (baseline)
        1 = + BIE × 词长   (11 维, 无 I2)
        2 = + rel_seen      (14 维)
        3 = + rel_all       (17 维, dict_core, 不含 meta)
        4 = + meta          (BIE + meta, 14 维)
        5 = dict_full       (BIE + rel_all + meta, 20 维)

支持无标注语料 gap 特征 (UnlabeledStatsExtractor):
    通过 gap_feature_level 控制消融:
        0 = 不使用
        6 = + freq (2 维)
        7 = + freq + assoc (4 维)
        8 = + freq + assoc + entropy (8 维)
        9 = + freq + entropy (6 维, 无关联度量)
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np
from models.base import Segmenter
from data.dataset import words_to_bies, bies_to_words
import config
from models.lexicon import (
    LexiconFeatureExtractor, DICT_FEATURE_NAMES, DICT_FEATURE_DIM,
    LEVEL_INDICES, ReliabilityInfo,
    compute_lexicon_reliability, compute_class_priors, compute_oof_dict_vectors,
)
from models.unlabeled_stats import (
    UnlabeledStatsExtractor, GAP_FEATURE_NAMES, _GAP_LEVEL_INDICES,
)

# 各消融级别使用的列索引 (从 lexicon.LEVEL_INDICES 导入)
# LEVEL_INDICES = {0: [], 1: range(11), 2: range(14), 3: range(17),
#                  4: range(11) + range(17, 20), 5: range(20)}


class CRFSegmenter(Segmenter):
    """基于 CRF 的两阶段分词+词性标注器。

    阶段 1: CRF 序列标注做 BIES 分词（4 标签，快速）
    阶段 2: 训练集高频词性查表（与词典法一致）

    Args:
        lexicon_extractor: 共享的 LexiconFeatureExtractor 实例（可为 None）
        dict_feature_level: 0-5, 控制使用多少词典特征维度
        unlabeled_extractor: 共享的 UnlabeledStatsExtractor 实例（可为 None）
        gap_feature_level: 0-3, 控制使用多少 gap 特征维度
    """

    name = "CRF"

    def __init__(self, c1: float = 1.0, c2: float = 1e-3,
                 max_iterations: int = 200,
                 all_possible_transitions: bool = True,
                 lexicon_extractor: Optional[LexiconFeatureExtractor] = None,
                 dict_feature_level: int = 0,
                 unlabeled_extractor: Optional[UnlabeledStatsExtractor] = None,
                 gap_feature_level: int = 0):
        self._c1 = c1
        self._c2 = c2
        self._max_iterations = max_iterations
        self._all_possible = all_possible_transitions
        self._model = None
        self._word_set: set = set()
        self._word_pos: Dict[str, str] = {}  # 词 -> 最高频词性

        # 词典格网特征
        self._lexicon_extractor = lexicon_extractor
        self._dict_feature_level = dict_feature_level
        self._dict_feature_indices: List[int] = LEVEL_INDICES.get(dict_feature_level, [])

        # 无标注语料 gap 特征
        self._unlabeled_extractor = unlabeled_extractor
        self._gap_feature_level = gap_feature_level
        self._gap_feature_indices: List[int] = _GAP_LEVEL_INDICES.get(gap_feature_level, [])

    # ---------- 训练 ----------
    def fit(self, train_words: List[List[str]],
            train_tags: List[List[str]] = None,
            dev_words: List[List[str]] = None,
            dev_tags: List[List[str]] = None) -> None:
        """训练 CRF 模型（分词）+ 构建词性映射。

        支持早停: 若有 dev_words/dev_tags，则用验证集损失进行早停。
        """
        # -- 构建词表（特征用）--
        self._word_set = set()
        for words in train_words:
            self._word_set.update(words)

        # -- 构建词→词性映射（高频）--
        if train_tags is not None:
            pos_counter: Dict[str, Counter] = {}
            for words, tags in zip(train_words, train_tags):
                for w, t in zip(words, tags):
                    if w not in pos_counter:
                        pos_counter[w] = Counter()
                    pos_counter[w][t] += 1
            for w, counter in pos_counter.items():
                self._word_pos[w] = counter.most_common(1)[0][0]

        # -- 预计算词典特征 (level >= 2 时使用 OOF 可靠度) --
        train_dict_vecs: List[Optional[np.ndarray]] = [None] * len(train_words)
        if self._lexicon_extractor is not None and self._dict_feature_level > 0:
            if self._dict_feature_level >= 2:
                # OOF 可靠度: 训练集内部 k 折, 保证每句的可靠度不含自身边界
                train_dict_vecs = compute_oof_dict_vectors(
                    train_words, self._lexicon_extractor.trie, self._lexicon_extractor,
                    seed=config.RANDOM_SEED,
                )
                # 为 predict() 准备全训练集可靠度
                full_rel = compute_lexicon_reliability(
                    train_words, self._lexicon_extractor.trie,
                )
                self._lexicon_extractor.set_reliability(full_rel)
                # 为 predict() 准备类别先验 (rel_unseen 用)
                full_priors = compute_class_priors(
                    self._lexicon_extractor.trie, train_words,
                )
                self._lexicon_extractor.set_class_priors(full_priors)
            else:
                train_sents = ["".join(w) for w in train_words]
                train_dict_vecs = self._lexicon_extractor.extract_batch(train_sents)

        # -- 构造 CRF 特征 --
        X, y = [], []
        for i, words in enumerate(train_words):
            chars = list("".join(words))
            gold_bies = words_to_bies(words)
            dv = train_dict_vecs[i]
            gv = None
            if self._unlabeled_extractor is not None and self._gap_feature_level > 0:
                sent = "".join(words)
                gv = self._unlabeled_extractor.extract(sent, gap_level=self._gap_feature_level)
            feats = [self._extract_features(chars, j, dv, gv) for j in range(len(chars))]
            X.append(feats)
            y.append(gold_bies)

        # -- 训练 CRF --
        print(f"  [CRF] Training (max_iter={self._max_iterations}):")
        from sklearn_crfsuite import CRF
        self._model = CRF(
            algorithm="lbfgs",
            c1=self._c1,
            c2=self._c2,
            max_iterations=self._max_iterations,
            all_possible_transitions=self._all_possible,
        )
        self._model.fit(X, y)

    # ---------- 预测 ----------
    def predict(self, sentence: str) -> Tuple[List[str], List[str]]:
        """对单句进行分词+词性标注。"""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        chars = list(sentence)

        # 词典特征
        dv: Optional[np.ndarray] = None
        if self._lexicon_extractor is not None and self._dict_feature_level > 0:
            dv = self._lexicon_extractor.extract(sentence)

        # gap 特征
        gv: Optional[np.ndarray] = None
        if self._unlabeled_extractor is not None and self._gap_feature_level > 0:
            gv = self._unlabeled_extractor.extract(sentence, gap_level=self._gap_feature_level)

        feats = [self._extract_features(chars, i, dv, gv) for i in range(len(chars))]
        bies_tags = self._model.predict_single(feats)
        words = bies_to_words(chars, bies_tags)
        pos = [self._word_pos.get(w, "x") for w in words]
        return words, pos

    # ---------- 持久化 ----------
    def save(self, path: str) -> None:
        import joblib
        data = {"model": self._model, "word_set": self._word_set, "word_pos": self._word_pos}
        joblib.dump(data, path)

    def load(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self._model = data["model"]
        self._word_set = data["word_set"]
        self._word_pos = data.get("word_pos", {})

    # ------------------------------------------------------
    # 特征模板
    # ------------------------------------------------------
    def _extract_features(self, chars: List[str], i: int,
                          dict_vector: Optional[np.ndarray] = None,
                          gap_vector: Optional[np.ndarray] = None) -> Dict:
        """提取位置 i 处字的特征。

        Args:
            chars: 字符列表
            i: 位置索引
            dict_vector: (N, D) 词典格网向量, 或 None
            gap_vector: (N, 8) gap 特征向量, 或 None

        Returns:
            特征字典 (sklearn-crfsuite 支持 float 值)
        """
        ch = chars[i]
        feats = {}

        # --- 字符 n-gram ---
        feats["c0"] = ch
        feats["c-1"] = chars[i - 1] if i > 0 else "<BOS>"
        feats["c+1"] = chars[i + 1] if i < len(chars) - 1 else "<EOS>"
        feats["c-2"] = chars[i - 2] if i > 1 else "<BOS2>"
        feats["c+2"] = chars[i + 2] if i < len(chars) - 2 else "<EOS2>"

        # --- 二元组合 ---
        if i > 0:
            feats["c-1_c0"] = chars[i - 1] + ch
        if i < len(chars) - 1:
            feats["c0_c+1"] = ch + chars[i + 1]

        # --- 字形特征 ---
        feats["is_digit"] = str(ch.isdigit())
        feats["is_alpha"] = str(ch.isalpha())
        feats["is_punct"] = str(not ch.isalnum() and not ch.isspace())

        # --- 旧词典特征（保留做消融）---
        feats["in_dict"] = str(ch in self._word_set)

        # --- 统一的词典格网特征 ---
        if dict_vector is not None and self._dict_feature_indices:
            for j in self._dict_feature_indices:
                if j >= dict_vector.shape[1]:
                    break
                val = float(dict_vector[i, j])
                if val != 0.0:
                    feats[f"dict:{DICT_FEATURE_NAMES[j]}"] = val

        # --- 无标注语料 gap 特征 ---
        if gap_vector is not None and self._gap_feature_indices:
            for j in self._gap_feature_indices:
                if j >= gap_vector.shape[1]:
                    break
                val = float(gap_vector[i, j])
                if val != 0.0:
                    feats[f"gap:{GAP_FEATURE_NAMES[j]}"] = val

        return feats
