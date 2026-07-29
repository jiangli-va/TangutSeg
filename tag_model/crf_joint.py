"""基于 CRF 的联合分词+词性标注器 —— 使用 BI+POS 大标签集。

字符级特征模板与 models/crf.py 中的 CRFSegmenter 完全相同，
区别在于标签集从 {B,I,E,S} 变为 {B-pos, I-pos, E-pos, S-pos}。
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np
from models.base import Segmenter
from tag_model.tag_utils import (
    normalize_tags, build_joint_label_map,
    words_tags_to_bies_pos, bies_pos_to_words_tags,
)
import config
from models.lexicon import (
    LexiconFeatureExtractor, DICT_FEATURE_NAMES, DICT_FEATURE_DIM,
    LEVEL_INDICES, compute_lexicon_reliability, compute_class_priors,
    compute_oof_dict_vectors,
)
from models.unlabeled_stats import (
    UnlabeledStatsExtractor, GAP_FEATURE_NAMES, _GAP_LEVEL_INDICES,
)


class CRFJointTagger(Segmenter):
    """CRF 联合分词+词性标注器。

    使用 BI+POS 大标签集，一次性完成分词和词性标注。

    Args:
        lexicon_extractor: 共享的 LexiconFeatureExtractor 实例
        dict_feature_level: 0-8, 控制词典特征维度 (6-8 对应 gap 消融)
        unlabeled_extractor: 共享的 UnlabeledStatsExtractor 实例 (level 6-8 需要)
        gap_feature_level: 1-3, 控制 gap 特征维度 (level 6-8 自动推导)
    """

    name = "CRF-Joint"

    def __init__(self, c1: float = 1.0, c2: float = 1e-3,
                 max_iterations: int = 200,
                 all_possible_transitions: bool = True,
                 lexicon_extractor: Optional[LexiconFeatureExtractor] = None,
                 dict_feature_level: int = 5,
                 unlabeled_extractor: Optional[UnlabeledStatsExtractor] = None,
                 gap_feature_level: int = 0):
        self._c1 = c1
        self._c2 = c2
        self._max_iterations = max_iterations
        self._all_possible = all_possible_transitions
        self._model = None
        self._word_set: set = set()

        # 词典格网特征
        self._lexicon_extractor = lexicon_extractor
        self._dict_feature_level = dict_feature_level
        # 实际词典特征级别: level 6-8 都使用 level 5 的词典特征
        _actual_dict_level = min(dict_feature_level, 5)
        self._dict_feature_indices: List[int] = LEVEL_INDICES.get(_actual_dict_level, [])

        # gap 特征
        self._unlabeled_extractor = unlabeled_extractor
        self._gap_feature_level = gap_feature_level
        self._gap_feature_indices: List[int] = _GAP_LEVEL_INDICES.get(gap_feature_level, [])

        # 联合标签映射 (fit 时构建)
        self._tag2idx: Dict[str, int] = {}
        self._idx2tag: Dict[int, str] = {}

    # ---------- 训练 ----------
    def fit(self, train_words: List[List[str]],
            train_tags: List[List[str]] = None) -> None:
        """训练 CRF 联合模型。"""
        from sklearn_crfsuite import CRF

        if train_tags is None:
            raise ValueError("CRFJointTagger requires train_tags (POS labels) for training.")

        # 规范化词性标签
        norm_tags = [normalize_tags(ts) for ts in train_tags]

        # 构建联合标签映射
        self._tag2idx, self._idx2tag, _ = build_joint_label_map(norm_tags)

        # -- 构建词表（特征用）--
        self._word_set = set()
        for words in train_words:
            self._word_set.update(words)

        # -- 预计算词典特征 (level >= 2 时使用 OOF 可靠度) --
        train_dict_vecs: List[Optional[np.ndarray]] = [None] * len(train_words)
        if self._lexicon_extractor is not None and self._dict_feature_level > 0:
            if self._dict_feature_level >= 2:
                # OOF 可靠度: 训练集内部 k 折
                train_dict_vecs = compute_oof_dict_vectors(
                    train_words, self._lexicon_extractor.trie, self._lexicon_extractor,
                    seed=config.RANDOM_SEED,
                )
                # 为 predict() 准备全训练集可靠度
                full_rel = compute_lexicon_reliability(
                    train_words, self._lexicon_extractor.trie,
                )
                self._lexicon_extractor.set_reliability(full_rel)
                full_priors = compute_class_priors(
                    self._lexicon_extractor.trie, train_words,
                )
                self._lexicon_extractor.set_class_priors(full_priors)
            else:
                train_sents = ["".join(w) for w in train_words]
                train_dict_vecs = self._lexicon_extractor.extract_batch(train_sents)

        # -- 预计算 gap 特征 (dict_feature_level >= 6) --
        train_gap_vecs: List[Optional[np.ndarray]] = [None] * len(train_words)
        if self._unlabeled_extractor is not None and self._gap_feature_level > 0:
            for i, words in enumerate(train_words):
                sent = "".join(words)
                train_gap_vecs[i] = self._unlabeled_extractor.extract(
                    sent, self._gap_feature_level,
                )

        # -- 训练 CRF（联合 BI+POS 标签，保留为字符串）--
        # sklearn_crfsuite 要求标签为字符串，不能是整数索引
        X, y = [], []
        for i, words in enumerate(train_words):
            chars = list("".join(words))
            gold_joint = words_tags_to_bies_pos(words, norm_tags[i])
            dv = train_dict_vecs[i]
            gv = train_gap_vecs[i]
            feats = [self._extract_features(chars, j, dv, gv) for j in range(len(chars))]
            X.append(feats)
            y.append(gold_joint)

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
        """对单句进行联合分词+词性标注。"""
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
            gv = self._unlabeled_extractor.extract(sentence, self._gap_feature_level)

        feats = [self._extract_features(chars, i, dv, gv) for i in range(len(chars))]
        joint_tags = self._model.predict_single(feats)

        return bies_pos_to_words_tags(chars, joint_tags)

    # ---------- 持久化 ----------
    def save(self, path: str) -> None:
        import joblib
        data = {
            "model": self._model,
            "word_set": self._word_set,
            "tag2idx": self._tag2idx,
            "idx2tag": self._idx2tag,
        }
        joblib.dump(data, path)

    def load(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self._model = data["model"]
        self._word_set = data["word_set"]
        self._tag2idx = data["tag2idx"]
        self._idx2tag = data["idx2tag"]

    # ------------------------------------------------------
    # 特征模板 (与 CRFSegmenter 完全相同)
    # ------------------------------------------------------
    def _extract_features(self, chars: List[str], i: int,
                          dict_vector: Optional[np.ndarray] = None,
                          gap_vector: Optional[np.ndarray] = None) -> Dict:
        """提取位置 i 处字的特征。"""
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

        # --- 旧词典特征 ---
        feats["in_dict"] = str(ch in self._word_set)

        # --- 统一的词典格网特征 ---
        if dict_vector is not None and self._dict_feature_indices:
            for j in self._dict_feature_indices:
                if j >= dict_vector.shape[1]:
                    break
                val = float(dict_vector[i, j])
                if val != 0.0:
                    feats[f"dict:{DICT_FEATURE_NAMES[j]}"] = val

        # --- gap 特征 (无标注语料统计特征) ---
        if gap_vector is not None and self._gap_feature_indices:
            for j in self._gap_feature_indices:
                if j >= gap_vector.shape[1]:
                    break
                val = float(gap_vector[i, j])
                if val != 0.0:
                    feats[f"gap:{GAP_FEATURE_NAMES[j]}"] = val

        return feats
