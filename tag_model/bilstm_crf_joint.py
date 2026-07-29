"""基于 BiLSTM-CRF 的联合分词+词性标注器 —— 使用 BI+POS 大标签集。

完全复用现有特征和向量（外部词典、internal-only、领域分布），
区别仅在于标签集从 {B,I,E,S} 变为 {B-pos, I-pos, E-pos, S-pos}。

模型架构、Dataset、collate_fn、训练循环均与 models/bilstm_crf.py 中的
BiLSTMCRFSegmenter 结构相同，只是标签集更大。
"""

import math
import os
import subprocess
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset as TorchDataset
from models.base import Segmenter
import config
from tag_model.tag_utils import (
    normalize_tags, build_joint_label_map,
    words_tags_to_bies_pos, bies_pos_to_words_tags,
)
from models.lexicon import (
    LexiconFeatureExtractor,
    LEVEL_INDICES,
    compute_lexicon_reliability,
    compute_class_priors,
    compute_oof_dict_vectors,
    DICT_FEATURE_DIM,
    INTERNAL_BIE_DIM,
    build_internal_only_trie,
    extract_internal_bie,
    compute_oof_internal_vectors,
    compute_word_domain_distribution,
    _sentence_domain_fallback,
    extract_domain_vec,
    compute_oof_domain_vectors,
)
from models.unlabeled_stats import UnlabeledStatsExtractor


# ============================================================
# 辅助函数
# ============================================================

def _slice_dict_vec(vec: np.ndarray, level: int) -> np.ndarray:
    """根据消融级别从完整 20 维向量中取出所需列。"""
    indices = LEVEL_INDICES.get(level, [])
    if not indices:
        return np.zeros((vec.shape[0], 0), dtype=np.float32)
    return vec[:, indices].astype(np.float32)


def _format_cuda_device(index: int) -> str:
    return f"cuda:{index}"


def log_sum_exp(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max, _ = x.max(dim=dim, keepdim=True)
    return x_max.squeeze(dim) + (x - x_max).exp().sum(dim=dim).log()


def lengths_to_last_idx(mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=1).long() - 1
    return lengths.clamp(min=0)


# ============================================================
# 联合标签的词汇映射
# ============================================================

def create_joint_vocab_maps(
    train_words: List[List[str]],
    train_tags: List[List[str]],
):
    """从训练集构建 字→索引 和 联合标签→索引 映射。

    Returns:
        char2idx, idx2char, tag2idx, idx2tag
    """
    char_set = {"<PAD>", "<UNK>"}
    for words in train_words:
        for w in words:
            char_set.update(w)
    char2idx = {c: i for i, c in enumerate(sorted(char_set))}
    idx2char = {i: c for c, i in char2idx.items()}

    # 规范化并构建联合标签映射
    norm_tags = [normalize_tags(ts) for ts in train_tags]
    tag2idx, idx2tag, _ = build_joint_label_map(norm_tags)
    return char2idx, idx2char, tag2idx, idx2tag


# ============================================================
# PyTorch BiLSTM-CRF 模型 (与 BiLSTMCRFModel 相同, 只是 tagset_size 更大)
# ============================================================

class BiLSTMCRFJointModel(nn.Module):
    """BiLSTM + CRF 联合分词+词性标注模型。

    与 BiLSTMCRFModel 架构完全相同，仅标签集不同。
    """

    def __init__(self, vocab_size: int, tagset_size: int,
                 embedding_dim: int = 100, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.5,
                 dict_feat_dim: int = 0,
                 internal_feat_dim: int = 0,
                 domain_dist_dim: int = 0,
                 dict_dropout: float = 0.2,
                 gap_feat_dim: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dict_feat_dim = dict_feat_dim
        self.internal_feat_dim = internal_feat_dim
        self.domain_dist_dim = domain_dist_dim
        self.gap_feat_dim = gap_feat_dim
        self.dict_dropout_rate = dict_dropout
        self.dict_dropout = nn.Dropout(dict_dropout) if dict_feat_dim > 0 else None

        lstm_input_dim = embedding_dim + dict_feat_dim + internal_feat_dim + domain_dist_dim + gap_feat_dim
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim // 2,
                            num_layers=num_layers, bidirectional=True,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size) * 0.01)
        self.start_transitions = nn.Parameter(torch.randn(tagset_size) * 0.01)
        self.end_transitions = nn.Parameter(torch.randn(tagset_size) * 0.01)

        self.tagset_size = tagset_size

    def _lstm_features(self, x: torch.Tensor, lengths: torch.Tensor,
                       dict_vecs: Optional[torch.Tensor] = None,
                       internal_vecs: Optional[torch.Tensor] = None,
                       domain_vecs: Optional[torch.Tensor] = None,
                       gap_vecs: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(x)
        parts = [emb]

        if self.dict_feat_dim > 0 and dict_vecs is not None:
            if self.training and self.dict_dropout is not None:
                dict_vecs = self.dict_dropout(dict_vecs)
            parts.append(dict_vecs)

        if self.internal_feat_dim > 0 and internal_vecs is not None:
            parts.append(internal_vecs)

        if self.domain_dist_dim > 0 and domain_vecs is not None:
            parts.append(domain_vecs)

        if self.gap_feat_dim > 0 and gap_vecs is not None:
            parts.append(gap_vecs)

        emb = torch.cat(parts, dim=-1)
        emb = self.dropout(emb)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def _crf_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        score = emissions.new_zeros(batch_size)
        score += (self.start_transitions[tags[:, 0]] +
                  emissions[range(batch_size), 0, tags[:, 0]]) * mask[:, 0]
        for t in range(seq_len - 1):
            cur_valid = mask[:, t + 1]
            score += (
                emissions[range(batch_size), t + 1, tags[:, t + 1]] * cur_valid
                + self.transitions[tags[:, t], tags[:, t + 1]] * cur_valid
            )
        last_tag = tags[range(batch_size), lengths_to_last_idx(mask)]
        score += self.end_transitions[last_tag]
        return score

    def _crf_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, tagset_size = emissions.shape
        mask_float = mask.float()
        alpha = self.start_transitions + emissions[:, 0]

        for t in range(1, seq_len):
            emit_t = emissions[:, t].unsqueeze(1)
            trans_t = self.transitions.unsqueeze(0)
            alpha_t = alpha.unsqueeze(2) + trans_t + emit_t
            next_alpha = log_sum_exp(alpha_t, dim=1)
            m = mask_float[:, t].unsqueeze(1)
            alpha = next_alpha * m + alpha * (1 - m)

        last_alpha = alpha + self.end_transitions.unsqueeze(0)
        return log_sum_exp(last_alpha, dim=1)

    def neg_log_likelihood(self, x: torch.Tensor, tags: torch.Tensor,
                           lengths: torch.Tensor,
                           dict_vecs: Optional[torch.Tensor] = None,
                           internal_vecs: Optional[torch.Tensor] = None,
                           domain_vecs: Optional[torch.Tensor] = None,
                           sample_weights: Optional[torch.Tensor] = None,
                           gap_vecs: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = x != 0
        emissions = self._lstm_features(x, lengths, dict_vecs, internal_vecs, domain_vecs, gap_vecs)
        log_likelihood = self._crf_log_likelihood(emissions, tags, mask)
        partition = self._crf_partition(emissions, mask)
        per_sample_loss = partition - log_likelihood
        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights
        return per_sample_loss.mean()

    def decode(self, x: torch.Tensor, lengths: torch.Tensor,
               dict_vecs: Optional[torch.Tensor] = None,
               internal_vecs: Optional[torch.Tensor] = None,
               domain_vecs: Optional[torch.Tensor] = None,
               gap_vecs: Optional[torch.Tensor] = None) -> List[List[int]]:
        mask = x != 0
        emissions = self._lstm_features(x, lengths, dict_vecs, internal_vecs, domain_vecs, gap_vecs)
        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions: torch.Tensor,
                        mask: torch.Tensor) -> List[List[int]]:
        batch_size, seq_len, tagset_size = emissions.shape
        mask_float = mask.float()

        scores = self.start_transitions + emissions[:, 0]
        backpointers = []
        for t in range(1, seq_len):
            next_scores = (scores.unsqueeze(2) + self.transitions.unsqueeze(0)
                           + emissions[:, t].unsqueeze(1))
            best_scores, best_tags = next_scores.max(dim=1)
            m = mask_float[:, t].unsqueeze(1)
            scores = best_scores * m + scores * (1 - m)
            backpointers.append(best_tags)

        scores = scores + self.end_transitions.unsqueeze(0)
        best_last_tags = scores.argmax(dim=1).tolist()

        best_paths = []
        for b in range(batch_size):
            path = [best_last_tags[b]]
            for bp in reversed(backpointers):
                path.append(bp[b, path[-1]].item())
            path.reverse()
            best_paths.append(path)
        return best_paths


# ============================================================
# Dataset (与 CharDataset 相同, 但标签是 joint BI+POS)
# ============================================================

class JointCharDataset(TorchDataset):
    """将句子转为字符索引 + 词典特征向量 + 联合 BI+POS 标签 + 领域分布向量 + 样本权重。"""

    def __init__(self, sentences_words: List[List[str]],
                 sentences_tags: List[List[str]],
                 char2idx: Dict[str, int], tag2idx: Dict[str, int],
                 dict_vectors: Optional[List[np.ndarray]] = None,
                 internal_vectors: Optional[List[np.ndarray]] = None,
                 domain_vectors: Optional[List[np.ndarray]] = None,
                 gap_vectors: Optional[List[np.ndarray]] = None,
                 sample_weights: Optional[List[float]] = None):
        self.has_dict = dict_vectors is not None
        self.has_internal = internal_vectors is not None
        self.has_domain = domain_vectors is not None
        self.has_gap = gap_vectors is not None

        norm_tags = [normalize_tags(ts) for ts in sentences_tags]
        self.data = []
        for i, words in enumerate(sentences_words):
            chars = list("".join(words))
            joint_tags = words_tags_to_bies_pos(words, norm_tags[i])

            char_ids = [char2idx.get(c, char2idx["<UNK>"]) for c in chars]
            tag_ids = [tag2idx.get(t, 0) for t in joint_tags]

            if self.has_dict:
                dv = dict_vectors[i]
            else:
                dv = np.zeros((len(chars), 0), dtype=np.float32)
            if self.has_internal:
                iv = internal_vectors[i]
            else:
                iv = np.zeros((len(chars), 0), dtype=np.float32)
            if self.has_domain:
                dm = domain_vectors[i]
            else:
                dm = np.zeros((len(chars), 0), dtype=np.float32)

            if self.has_gap:
                gv = gap_vectors[i]
            else:
                gv = np.zeros((len(chars), 0), dtype=np.float32)

            weight = sample_weights[i] if sample_weights is not None else 1.0
            self.data.append((char_ids, dv, tag_ids, iv, dm, weight, gv))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_joint(batch):
    """动态填充 + 按长度降序排序。"""
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    chars_list, dv_list, tags_list, iv_list, dm_list, weight_list, gv_list = zip(*batch)
    lengths = torch.tensor([len(c) for c in chars_list], dtype=torch.long)
    max_len = lengths[0].item()
    batch_size = len(batch)

    padded_chars = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_tags = torch.zeros(batch_size, max_len, dtype=torch.long)

    ext_feat_dim = dv_list[0].shape[1] if dv_list[0].ndim > 1 else 0
    int_feat_dim = iv_list[0].shape[1] if iv_list[0].ndim > 1 else 0
    domain_dim = dm_list[0].shape[1] if dm_list[0].ndim > 1 else 0
    gap_dim = gv_list[0].shape[1] if gv_list[0].ndim > 1 else 0

    padded_dv = torch.zeros(batch_size, max_len, max(ext_feat_dim, 0), dtype=torch.float32)
    padded_iv = torch.zeros(batch_size, max_len, max(int_feat_dim, 0), dtype=torch.float32)
    padded_ddom = torch.zeros(batch_size, max_len, max(domain_dim, 0), dtype=torch.float32)
    padded_gv = torch.zeros(batch_size, max_len, max(gap_dim, 0), dtype=torch.float32)
    sample_weights = torch.tensor(weight_list, dtype=torch.float32)

    for i, (c, dv, t, iv, dm, weight, gv) in enumerate(
            zip(chars_list, dv_list, tags_list, iv_list, dm_list, weight_list, gv_list)):
        L = len(c)
        padded_chars[i, :L] = torch.tensor(c, dtype=torch.long)
        padded_tags[i, :L] = torch.tensor(t, dtype=torch.long)
        if ext_feat_dim > 0:
            padded_dv[i, :L] = torch.from_numpy(dv)
        if int_feat_dim > 0:
            padded_iv[i, :L] = torch.from_numpy(iv)
        if domain_dim > 0:
            padded_ddom[i, :L] = torch.from_numpy(dm)
        if gap_dim > 0:
            padded_gv[i, :L] = torch.from_numpy(gv)

    return padded_chars, padded_dv, padded_tags, padded_iv, padded_ddom, sample_weights, lengths, padded_gv


# ============================================================
# Segmenter 封装
# ============================================================

class BiLSTMCRFJointTagger(Segmenter):
    """BiLSTM-CRF 联合分词+词性标注器。

    完全复用现有特征和向量, 仅标签集变为 BI+POS。
    支持 dict_feature_level 0-8 (6-8 对应 gap 消融)。
    """

    name = "BiLSTM-CRF-Joint"

    def __init__(self,
                 embedding_dim: int = 100,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 learning_rate: float = 5e-4,
                 batch_size: int = 512,
                 epochs: int = 10000,
                 device: str = "auto",
                 lr_patience: int = 5,
                 lr_factor: float = 0.3,
                 early_stop_patience: int = 3,
                 grad_clip: float = 5.0,
                 lexicon_extractor: Optional[LexiconFeatureExtractor] = None,
                 dict_feature_level: int = 5,
                 dict_dropout: float = 0.2,
                 jingshu_loss_weight: float = 1.0,
                 unlabeled_extractor: Optional[UnlabeledStatsExtractor] = None,
                 gap_feature_level: int = 0):
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout
        self._lr = learning_rate
        self._batch_size = batch_size
        self._epochs = epochs
        self._device = self._resolve_device(device)
        self._lr_patience = lr_patience
        self._lr_factor = lr_factor
        self._early_stop_patience = early_stop_patience
        self._grad_clip = grad_clip
        self._lexicon_extractor = lexicon_extractor
        self._dict_feature_level = dict_feature_level
        self._dict_feat_dim = 0
        self._internal_feat_dim = 0
        self._dict_dropout = dict_dropout
        self._domain_dist_dim = 2 if dict_feature_level >= 5 else 0
        self._domain_dist_for_inference = None
        self._domain_fallback_for_inference = None
        self._jingshu_loss_weight = jingshu_loss_weight
        self._best_model_state = None

        # gap 特征
        self._unlabeled_extractor = unlabeled_extractor
        self._gap_feature_level = gap_feature_level
        self._gap_feat_dim = 8 if gap_feature_level > 0 else 0

        self._model: Optional[BiLSTMCRFJointModel] = None
        self._char2idx: Dict[str, int] = {}
        self._idx2tag: Dict[int, str] = {}
        self._tag2idx: Dict[str, int] = {}
        self._internal_trie_for_inference = None

    @property
    def _use_dict(self) -> bool:
        return self._lexicon_extractor is not None and self._dict_feature_level > 0

    @property
    def _use_internal(self) -> bool:
        return self._dict_feature_level >= 4

    @property
    def _use_domain(self) -> bool:
        return self._domain_dist_dim > 0

    @property
    def _use_gap(self) -> bool:
        return self._unlabeled_extractor is not None and self._gap_feature_level > 0

    def _get_dict_feat_dim(self) -> int:
        indices = LEVEL_INDICES.get(min(self._dict_feature_level, 3), [])
        return len(indices)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        d = (device or "auto").lower()
        if d == "auto":
            return BiLSTMCRFJointTagger._choose_best_cuda_device()
        elif d.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(device)
            print("[WARN] CUDA not available, falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cpu")

    @staticmethod
    def _choose_best_cuda_device() -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        env_device = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_device and env_device.strip():
            return torch.device("cuda:0")
        try:
            query = [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(query, check=True, capture_output=True, text=True)
            candidates = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 4:
                    continue
                gpu_index = int(parts[0])
                util = int(parts[1])
                mem_used = int(parts[2])
                mem_total = max(int(parts[3]), 1)
                mem_ratio = mem_used / mem_total
                score = util + mem_ratio * 100
                candidates.append((score, mem_used, gpu_index))
            if candidates:
                _, _, best_gpu = min(candidates)
                return torch.device(_format_cuda_device(best_gpu))
        except Exception:
            pass
        return torch.device("cuda:0")

    # ---------- 训练 ----------
    def fit(self, train_words: List[List[str]],
            train_tags: List[List[str]] = None,
            dev_words: List[List[str]] = None,
            dev_tags: List[List[str]] = None,
            train_categories: Optional[List[str]] = None,
            dev_categories: Optional[List[str]] = None) -> None:
        """训练 BiLSTM-CRF 联合模型。"""
        if train_tags is None:
            raise ValueError("BiLSTMCRFJointTagger requires train_tags for training.")

        # 构建字/联合标签映射
        norm_train_tags = [normalize_tags(ts) for ts in train_tags]
        self._char2idx, _, self._tag2idx, self._idx2tag = create_joint_vocab_maps(
            train_words, train_tags)

        # ---- 逐句损失权重 ----
        train_weights: Optional[List[float]] = None
        if train_categories is not None and self._jingshu_loss_weight != 1.0:
            train_weights = [
                self._jingshu_loss_weight if c == "经书" else 1.0
                for c in train_categories
            ]

        # 词典特征维度
        self._dict_feat_dim = self._get_dict_feat_dim()
        self._internal_feat_dim = INTERNAL_BIE_DIM if self._use_internal else 0

        # ---- 领域分布向量 ----
        train_domain_vecs: Optional[List[np.ndarray]] = None
        dev_domain_vecs: Optional[List[np.ndarray]] = None
        if self._use_domain:
            if train_categories is None:
                train_categories = ["世俗文献"] * len(train_words)
            ext_trie = self._lexicon_extractor.trie
            train_domain_vecs = compute_oof_domain_vectors(
                train_words, train_categories, ext_trie,
                seed=config.RANDOM_SEED,
            )
            self._domain_dist_for_inference = compute_word_domain_distribution(
                train_words, train_categories,
            )
            if dev_words is not None and len(dev_words) > 0:
                if dev_categories is None:
                    dev_categories = ["世俗文献"] * len(dev_words)
                dev_domain_vecs = []
                dev_int_trie = build_internal_only_trie(train_words, ext_trie)
                for i, w in enumerate(dev_words):
                    chars = list("".join(w))
                    sent_fb = _sentence_domain_fallback(dev_categories[i])
                    dv = extract_domain_vec(
                        chars, ext_trie.find_all("".join(w)),
                        self._domain_dist_for_inference,
                        sent_fb,
                        internal_trie=dev_int_trie,
                    )
                    dev_domain_vecs.append(dv)

        # 预计算词典特征
        train_dict_vecs: Optional[List[np.ndarray]] = None
        dev_dict_vecs: Optional[List[np.ndarray]] = None
        if self._use_dict:
            if self._dict_feature_level >= 2:
                train_dict_vecs_full20 = compute_oof_dict_vectors(
                    train_words, self._lexicon_extractor.trie, self._lexicon_extractor,
                    seed=config.RANDOM_SEED,
                )
                import copy
                self._extractor_for_inference = copy.deepcopy(self._lexicon_extractor)
                full_rel = compute_lexicon_reliability(
                    train_words, self._lexicon_extractor.trie,
                )
                self._extractor_for_inference.set_reliability(full_rel)
                full_priors = compute_class_priors(
                    self._lexicon_extractor.trie, train_words,
                )
                self._extractor_for_inference.set_class_priors(full_priors)
            else:
                train_sents = ["".join(w) for w in train_words]
                train_dict_vecs_full20 = [
                    self._lexicon_extractor.extract(s) for s in train_sents
                ]
                self._extractor_for_inference = self._lexicon_extractor

            _ext_level = min(self._dict_feature_level, 3)
            train_dict_vecs = [
                _slice_dict_vec(v, _ext_level)
                for v in train_dict_vecs_full20
            ]

            if dev_words is not None and len(dev_words) > 0:
                dev_sents = ["".join(w) for w in dev_words]
                dev_dict_vecs = [
                    _slice_dict_vec(self._extractor_for_inference.extract(s), _ext_level)
                    for s in dev_sents
                ]

        # ---- internal-only BIE 格网 ----
        train_internal_vecs: Optional[List[np.ndarray]] = None
        dev_internal_vecs: Optional[List[np.ndarray]] = None
        if self._use_internal:
            ext_trie = self._lexicon_extractor.trie
            train_internal_vecs = compute_oof_internal_vectors(
                train_words, ext_trie,
                seed=config.RANDOM_SEED,
            )
            full_internal_trie = build_internal_only_trie(train_words, ext_trie)
            self._internal_trie_for_inference = full_internal_trie
            if dev_words is not None and len(dev_words) > 0:
                dev_internal_vecs = []
                for words in dev_words:
                    chars = list("".join(words))
                    vec = np.zeros((len(chars), INTERNAL_BIE_DIM), dtype=np.float32)
                    extract_internal_bie(chars, full_internal_trie, vec, 0)
                    dev_internal_vecs.append(vec)

        # ---- gap 特征 (无标注语料统计) ----
        train_gap_vecs: Optional[List[np.ndarray]] = None
        dev_gap_vecs: Optional[List[np.ndarray]] = None
        if self._use_gap:
            train_gap_vecs = []
            for words in train_words:
                sent = "".join(words)
                gv = self._unlabeled_extractor.extract(sent, self._gap_feature_level)
                train_gap_vecs.append(gv.astype(np.float32))
            if dev_words is not None and len(dev_words) > 0:
                dev_gap_vecs = []
                for words in dev_words:
                    sent = "".join(words)
                    gv = self._unlabeled_extractor.extract(sent, self._gap_feature_level)
                    dev_gap_vecs.append(gv.astype(np.float32))

        # 初始化模型
        tagset_size = len(self._tag2idx)
        print(f"  [BiLSTM-CRF-Joint] 标签集大小: {tagset_size} (BIES×POS)")
        self._model = BiLSTMCRFJointModel(
            vocab_size=len(self._char2idx),
            tagset_size=tagset_size,
            embedding_dim=self._embedding_dim,
            hidden_dim=self._hidden_dim,
            num_layers=self._num_layers,
            dropout=self._dropout,
            dict_feat_dim=self._dict_feat_dim,
            internal_feat_dim=self._internal_feat_dim,
            domain_dist_dim=self._domain_dist_dim,
            dict_dropout=self._dict_dropout,
            gap_feat_dim=self._gap_feat_dim,
        ).to(self._device)

        # 构建 DataLoader
        train_dataset = JointCharDataset(
            train_words, train_tags, self._char2idx, self._tag2idx,
            dict_vectors=train_dict_vecs,
            internal_vectors=train_internal_vecs,
            domain_vectors=train_domain_vecs,
            gap_vectors=train_gap_vecs,
            sample_weights=train_weights,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size,
            shuffle=True, collate_fn=collate_fn_joint,
            pin_memory=self._device.type == "cuda",
            num_workers=0,
        )

        dev_loader = None
        if dev_words is not None and len(dev_words) > 0:
            dev_dataset = JointCharDataset(
                dev_words, dev_tags, self._char2idx, self._tag2idx,
                dict_vectors=dev_dict_vecs,
                internal_vectors=dev_internal_vecs,
                domain_vectors=dev_domain_vecs,
                gap_vectors=dev_gap_vecs,
            )
            dev_loader = DataLoader(
                dev_dataset, batch_size=self._batch_size * 2,
                shuffle=False, collate_fn=collate_fn_joint,
                pin_memory=self._device.type == "cuda",
                num_workers=0,
            )

        optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self._lr_factor,
            patience=self._lr_patience)

        best_dev_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        self._model.train()
        for epoch in range(self._epochs):
            total_loss = 0.0
            for batch in train_loader:
                chars_batch, dv_batch, tags_batch, iv_batch, ddom_batch, weight_batch, lengths, gv_batch = batch
                chars_batch = chars_batch.to(self._device, non_blocking=True)
                tags_batch = tags_batch.to(self._device, non_blocking=True)
                lengths = lengths.to(self._device, non_blocking=True)
                dv_batch = dv_batch.to(self._device, non_blocking=True) if self._dict_feat_dim > 0 else None
                iv_batch = iv_batch.to(self._device, non_blocking=True) if self._internal_feat_dim > 0 else None
                ddom_batch = ddom_batch.to(self._device, non_blocking=True) if self._domain_dist_dim > 0 else None
                gv_batch = gv_batch.to(self._device, non_blocking=True) if self._gap_feat_dim > 0 else None
                weight_batch = weight_batch.to(self._device, non_blocking=True)

                optimizer.zero_grad()
                loss = self._model.neg_log_likelihood(
                    chars_batch, tags_batch, lengths, dv_batch, iv_batch, ddom_batch, weight_batch, gv_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / max(len(train_loader), 1)

            dev_loss_str = ""
            if dev_loader is not None:
                self._model.eval()
                dev_total_loss = 0.0
                with torch.no_grad():
                    for batch in dev_loader:
                        chars_batch, dv_batch, tags_batch, iv_batch, ddom_batch, weight_batch, lengths, gv_batch = batch
                        chars_batch = chars_batch.to(self._device, non_blocking=True)
                        tags_batch = tags_batch.to(self._device, non_blocking=True)
                        lengths = lengths.to(self._device, non_blocking=True)
                        dv_batch = dv_batch.to(self._device, non_blocking=True) if self._dict_feat_dim > 0 else None
                        iv_batch = iv_batch.to(self._device, non_blocking=True) if self._internal_feat_dim > 0 else None
                        ddom_batch = ddom_batch.to(self._device, non_blocking=True) if self._domain_dist_dim > 0 else None
                        gv_batch = gv_batch.to(self._device, non_blocking=True) if self._gap_feat_dim > 0 else None
                        weight_batch = weight_batch.to(self._device, non_blocking=True)
                        dev_loss = self._model.neg_log_likelihood(
                            chars_batch, tags_batch, lengths, dv_batch, iv_batch, ddom_batch, weight_batch, gv_batch)
                        dev_total_loss += dev_loss.item()
                avg_dev_loss = dev_total_loss / max(len(dev_loader), 1)
                dev_loss_str = f"  dev_loss={avg_dev_loss:.4f}"
                self._model.train()

                scheduler.step(avg_dev_loss)

                if avg_dev_loss < best_dev_loss:
                    best_dev_loss = avg_dev_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    self._best_model_state = {
                        k: v.cpu().clone() for k, v in self._model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self._early_stop_patience:
                        print(f"  Early stop at epoch {epoch + 1} "
                              f"(dev_loss 连续{self._early_stop_patience}次未下降, "
                              f"best: epoch {best_epoch}, dev_loss={best_dev_loss:.4f})")
                        break
            else:
                self._best_model_state = {
                    k: v.cpu().clone() for k, v in self._model.state_dict().items()
                }

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{self._epochs}  "
                      f"train_loss={avg_train_loss:.4f}{dev_loss_str}")

        if self._best_model_state is not None:
            self._model.load_state_dict(
                {k: v.to(self._device) for k, v in self._best_model_state.items()})
            print(f"  Loaded best model from epoch {best_epoch} (dev_loss={best_dev_loss:.4f})")

    # ---------- 预测 ----------
    def predict(self, sentence: str, category: str = "世俗文献") -> Tuple[List[str], List[str]]:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        chars = list(sentence)
        indices = [self._char2idx.get(c, self._char2idx.get("<UNK>", 0)) for c in chars]
        x = torch.tensor([indices], dtype=torch.long).to(self._device, non_blocking=True)
        lengths = torch.tensor([len(indices)], dtype=torch.long).to(self._device, non_blocking=True)

        dv_tensor = None
        if self._use_dict and hasattr(self, '_extractor_for_inference'):
            full_vec = self._extractor_for_inference.extract(sentence)
            sliced = _slice_dict_vec(full_vec, min(self._dict_feature_level, 3))
            dv_tensor = torch.from_numpy(sliced).unsqueeze(0).to(self._device, non_blocking=True)

        iv_tensor = None
        if self._use_internal and self._internal_trie_for_inference is not None:
            iv = np.zeros((len(chars), INTERNAL_BIE_DIM), dtype=np.float32)
            extract_internal_bie(chars, self._internal_trie_for_inference, iv, 0)
            iv_tensor = torch.from_numpy(iv).unsqueeze(0).to(self._device, non_blocking=True)

        ddom_tensor = None
        if self._use_domain and self._domain_dist_for_inference is not None:
            ext_trie = self._lexicon_extractor.trie
            ext_matches = ext_trie.find_all(sentence)
            sent_fb = _sentence_domain_fallback(category)
            dv = extract_domain_vec(
                chars, ext_matches,
                self._domain_dist_for_inference,
                sent_fb,
                internal_trie=self._internal_trie_for_inference,
            )
            ddom_tensor = torch.from_numpy(dv).unsqueeze(0).to(self._device, non_blocking=True)

        gv_tensor = None
        if self._use_gap:
            gv = self._unlabeled_extractor.extract(sentence, self._gap_feature_level)
            gv_tensor = torch.from_numpy(gv.astype(np.float32)).unsqueeze(0).to(self._device, non_blocking=True)

        self._model.eval()
        with torch.no_grad():
            tag_ids = self._model.decode(x, lengths, dv_tensor, iv_tensor, ddom_tensor, gv_tensor)[0]

        joint_tags = [self._idx2tag.get(tid, "S-x") for tid in tag_ids]
        return bies_pos_to_words_tags(chars, joint_tags)
