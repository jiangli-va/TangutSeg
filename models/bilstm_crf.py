"""基于 BiLSTM-CRF 的分词+词性标注器 (v4) —— 支持外部词典、internal-only 词典和领域分布向量。

架构:
    字符 → Embedding(随机初始化) → 拼接特征块 → BiLSTM → Linear → CRF → BIES 标签

特征块:
    - 外部词典格网 (20 维): BIE×词长 + rel_seen + rel_unseen + 元数据 (dict_full)
    - internal-only 格网 (11 维): 训练集独有词的 BIE×词长
    - 领域分布向量 (2 维): 逐位置拼接词级领域分布 [经书比例, 世俗比例]
    - 分布统计特征 (2/4/8 维): 词频、字符关联度和边界熵等 (按 gap_feature_level 切片)

"""

import math
import os
import subprocess
import random
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset as TorchDataset
from models.base import Segmenter
from data.dataset import words_to_bies, bies_to_words
import config
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
from models.unlabeled_stats import _GAP_LEVEL_INDICES
from data.dataset import words_to_bies, bies_to_words


def _dict_level_for_bilstm(level: int) -> int:
    """把 BiLSTM 的累积 level 映射为词典消融级别 (用于 _slice_dict_vec / 维度计算)。

    level 0-3 保持原消融级别; level >= 4 起外部词典特征与 CRF 的 dict_full(level 5)
    一致, 即完整 20 维 (BIE + rel_all + meta)。
    """
    if level <= 3:
        return level
    return 5  # dict_full (20 维)


def _slice_dict_vec(vec: np.ndarray, level: int) -> np.ndarray:
    """根据消融级别从完整 20 维向量中取出所需列。

    Args:
        vec: (seq_len, 20) 的 numpy 数组
        level: 词典消融级别 0-5 (见 _dict_level_for_bilstm)

    Returns:
        (seq_len, D_level) 的切片
    """
    indices = LEVEL_INDICES.get(level, [])
    if not indices:
        return np.zeros((vec.shape[0], 0), dtype=np.float32)
    return vec[:, indices].astype(np.float32)


# ======================== PyTorch BiLSTM-CRF 模型 ========================

class BiLSTMCRFModel(nn.Module):
    """BiLSTM + CRF 序列标注模型 (v5)。

    输入 = char_emb + ext_dict + internal_bie + domain_dist + gap_feat → BiLSTM → Linear → CRF
    dropout 施加在词典特征块整体上（而非逐元素）。
    """

    def __init__(self, vocab_size: int, tagset_size: int,
                 embedding_dim: int = 100, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.5,
                 dict_feat_dim: int = 0,
                 internal_feat_dim: int = 0,
                 domain_dist_dim: int = 0,
                 gap_feat_dim: int = 0,
                 dict_dropout: float = 0.2):
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
        """BiLSTM 编码 → 发射分数 [B, T, tagset_size]"""
        emb = self.embedding(x)  # [B, T, E]

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
        """计算 CRF 对数似然"""
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
        """前向算法计算 log Z"""
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
                           gap_vecs: Optional[torch.Tensor] = None,
                           sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算负对数似然损失 (支持逐样本加权)。

        sample_weights: [batch_size] float32, 每句的损失权重 (经书句 > 1.0 以加强)。
        """
        mask = x != 0
        emissions = self._lstm_features(x, lengths, dict_vecs, internal_vecs, domain_vecs, gap_vecs)
        log_likelihood = self._crf_log_likelihood(emissions, tags, mask)
        partition = self._crf_partition(emissions, mask)
        per_sample_loss = partition - log_likelihood  # [batch_size]
        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights
        return per_sample_loss.mean()

    def decode(self, x: torch.Tensor, lengths: torch.Tensor,
               dict_vecs: Optional[torch.Tensor] = None,
               internal_vecs: Optional[torch.Tensor] = None,
               domain_vecs: Optional[torch.Tensor] = None,
               gap_vecs: Optional[torch.Tensor] = None) -> List[List[int]]:
        """Viterbi 解码最优标签序列。"""
        mask = x != 0
        emissions = self._lstm_features(x, lengths, dict_vecs, internal_vecs, domain_vecs, gap_vecs)
        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions: torch.Tensor,
                        mask: torch.Tensor) -> List[List[int]]:
        """批量 Viterbi 解码。"""
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


def log_sum_exp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """数值稳定的 log-sum-exp。"""
    x_max, _ = x.max(dim=dim, keepdim=True)
    return x_max.squeeze(dim) + (x - x_max).exp().sum(dim=dim).log()


def lengths_to_last_idx(mask: torch.Tensor) -> torch.Tensor:
    """每个样本最后一个有效位置索引。"""
    lengths = mask.sum(dim=1).long() - 1
    return lengths.clamp(min=0)


def create_vocab_maps(train_words: List[List[str]]):
    """从训练集构建 字→索引 和 标签→索引 映射。"""
    char_set = {"<PAD>", "<UNK>"}
    for words in train_words:
        for w in words:
            char_set.update(w)
    char2idx = {c: i for i, c in enumerate(sorted(char_set))}
    idx2char = {i: c for c, i in char2idx.items()}

    tag2idx = {"B": 0, "I": 1, "E": 2, "S": 3}
    idx2tag = {i: t for t, i in tag2idx.items()}

    return char2idx, idx2char, tag2idx, idx2tag


# ======================== Dataset ========================

class CharDataset(TorchDataset):
    """将句子转为字符索引 + 词典特征向量 + BIES 标签索引 + 领域分布向量 + gap向量 + 样本权重。

    dict_vectors / internal_vectors / domain_vectors / gap_vectors 是预先计算好的 numpy 数组列表。
    sample_weights 是长度为 len(sentences_words) 的 float 列表 (None 表示全为 1.0)。
    """

    def __init__(self, sentences_words: List[List[str]],
                 char2idx: Dict[str, int], tag2idx: Dict[str, int],
                 dict_vectors: Optional[List[np.ndarray]] = None,
                 internal_vectors: Optional[List[np.ndarray]] = None,
                 domain_vectors: Optional[List[np.ndarray]] = None,
                 gap_vectors: Optional[List[np.ndarray]] = None,
                 sample_weights: Optional[List[float]] = None):
        self.data = []
        self.has_dict = dict_vectors is not None
        self.has_internal = internal_vectors is not None
        self.has_domain = domain_vectors is not None
        self.has_gap = gap_vectors is not None
        for i, words in enumerate(sentences_words):
            chars = list("".join(words))
            bies = words_to_bies(words)
            char_ids = [char2idx.get(c, char2idx["<UNK>"]) for c in chars]
            tag_ids = [tag2idx[t] for t in bies]
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
            self.data.append((char_ids, dv, tag_ids, iv, dm, gv, weight))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """动态填充 + 按长度降序排序。

    Returns:
        char_ids:       [batch, max_len] long
        dict_vecs:      [batch, max_len, ext_feat_dim] float32
        tags:           [batch, max_len] long
        internal_vecs:  [batch, max_len, int_feat_dim] float32
        domain_vecs:    [batch, max_len, domain_dim] float32
        gap_vecs:       [batch, max_len, gap_dim] float32
        sample_weights: [batch] float32
        lengths:        [batch] long
    """
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    chars_list, dv_list, tags_list, iv_list, dm_list, gv_list, weight_list = zip(*batch)
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
    padded_gap = torch.zeros(batch_size, max_len, max(gap_dim, 0), dtype=torch.float32)
    sample_weights = torch.tensor(weight_list, dtype=torch.float32)

    for i, (c, dv, t, iv, dm, gv, weight) in enumerate(zip(chars_list, dv_list, tags_list, iv_list, dm_list, gv_list, weight_list)):
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
            padded_gap[i, :L] = torch.from_numpy(gv)

    return padded_chars, padded_dv, padded_tags, padded_iv, padded_ddom, padded_gap, sample_weights, lengths


def _format_cuda_device(index: int) -> str:
    return f"cuda:{index}"


# ======================== Segmenter 封装 ========================

class BiLSTMCRFSegmenter(Segmenter):
    """BiLSTM-CRF 分词 + 词性标注器 (v5)。

    支持外部词典、internal-only 词典、领域分布向量和 gap 特征。

    dict_feature_level (累积式):
        0 = 不使用词典特征 (baseline)
        1 = 外部 BIE × 词长 (11 维)
        2 = 外部 BIE + rel_seen (14 维)
        3 = 外部 BIE + rel_all = dict_core (17 维)
        4 = dict_full (20 维, 同 CRF 的 level 5)
        5 = level 4 + internal-only BIE (20+11=31 维)
        6 = level 5 + 领域分布向量 (20+11+2=33 维)
        7 = level 6 + gap=freq (20+11+2+2=35 维)
        8 = level 6 + gap=freq+assoc (20+11+2+4=37 维)
        9 = level 6 + gap=all (20+11+2+8=41 维)

    gap_feature_level: 自动从 dict_feature_level 推导 (level-6)，仅 7/8/9 有效。
    """

    name = "BiLSTM-CRF"

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
                 dict_feature_level: int = 0,
                 dict_dropout: float = 0.2,
                 internal_trie: Optional["Trie"] = None,
                 domain_dist_dim: int = 0,
                 jingshu_loss_weight: float = 1.0,
                 unlabeled_extractor: Optional[object] = None,
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
        self._internal_trie = internal_trie
        self._internal_trie_for_inference = None
        self._domain_dist_dim = domain_dist_dim if dict_feature_level >= 6 else 0
        self._domain_dist_for_inference = None
        self._domain_fallback_for_inference = None
        self._jingshu_loss_weight = jingshu_loss_weight
        self._best_model_state = None

        # gap 特征
        self._unlabeled_extractor = unlabeled_extractor
        self._gap_feature_level = gap_feature_level
        self._gap_indices: List[int] = _GAP_LEVEL_INDICES.get(gap_feature_level, [])
        self._gap_feat_dim = len(self._gap_indices)

        self._model: Optional[BiLSTMCRFModel] = None
        self._char2idx: Dict[str, int] = {}
        self._idx2tag: Dict[int, str] = {}
        self._tag2idx: Dict[str, int] = {}
        self._word_pos: Dict[str, str] = {}
        self._word_set: set = set()

    @property
    def _use_dict(self) -> bool:
        return self._lexicon_extractor is not None and self._dict_feature_level > 0

    @property
    def _use_internal(self) -> bool:
        return self._dict_feature_level >= 5

    @property
    def _use_domain(self) -> bool:
        return self._domain_dist_dim > 0

    @property
    def _use_gap(self) -> bool:
        return self._unlabeled_extractor is not None and self._gap_feature_level > 0

    def _get_dict_feat_dim(self) -> int:
        """外部词典特征维度。level 0-3 按消融级别, level>=4 为 dict_full (20 维)。"""
        indices = LEVEL_INDICES.get(_dict_level_for_bilstm(self._dict_feature_level), [])
        return len(indices)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        d = (device or "auto").lower()
        if d == "auto":
            chosen = BiLSTMCRFSegmenter._choose_best_cuda_device()
        elif d.startswith("cuda"):
            if torch.cuda.is_available():
                chosen = torch.device(device)
            else:
                print("[WARN] 请求使用 CUDA，但当前不可用，已回退到 CPU。")
                chosen = torch.device("cpu")
        else:
            chosen = torch.device("cpu")
        print(f"[BiLSTM-CRF] 使用设备: {chosen}")
        return chosen

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
        """训练 BiLSTM-CRF 模型 (v4: 支持 internal-only BIE + 领域分布向量)。"""
        # 构建字/标签映射
        self._char2idx, _, self._tag2idx, self._idx2tag = create_vocab_maps(train_words)
        # 词→词性映射
        if train_tags is not None:
            pos_counter: Dict[str, Counter] = {}
            for words, tags in zip(train_words, train_tags):
                for w, t in zip(words, tags):
                    if w not in pos_counter:
                        pos_counter[w] = Counter()
                    pos_counter[w][t] += 1
            self._word_pos = {w: c.most_common(1)[0][0] for w, c in pos_counter.items()}
        # 词表
        self._word_set = set()
        for words in train_words:
            self._word_set.update(words)

        # ---- 逐句损失权重 (经书 vs 世俗文献) ----
        train_weights: Optional[List[float]] = None
        if train_categories is not None and self._jingshu_loss_weight != 1.0:
            train_weights = [
                self._jingshu_loss_weight if c == "经书" else 1.0
                for c in train_categories
            ]

        # 词典特征维度
        self._dict_feat_dim = self._get_dict_feat_dim()
        self._internal_feat_dim = INTERNAL_BIE_DIM if self._use_internal else 0

        # ---- 领域分布向量 (OOF + 逐句 fallback) ----
        train_domain_vecs: Optional[List[np.ndarray]] = None
        dev_domain_vecs: Optional[List[np.ndarray]] = None
        if self._use_domain:
            if train_categories is None:
                train_categories = ["世俗文献"] * len(train_words)
            ext_trie = self._lexicon_extractor.trie
            # OOF 计算领域分布向量 (已在 lexicon.py 中改为逐句 fallback)
            train_domain_vecs = compute_oof_domain_vectors(
                train_words, train_categories, ext_trie,
                seed=config.RANDOM_SEED,
            )
            # 为 inference 保存全训练集 domain_dist + fallback (但 predict 用逐句)
            self._domain_dist_for_inference = compute_word_domain_distribution(
                train_words, train_categories,
            )
            # 为 dev 集也生成 OOF domain vectors
            if dev_words is not None and len(dev_words) > 0:
                if dev_categories is None:
                    dev_categories = ["世俗文献"] * len(dev_words)
                # dev 用全训练集统计 → OOF 不针对 dev 算，直接用全训练集 domain_dist
                dev_sents = ["".join(w) for w in dev_words]
                dev_domain_vecs = []
                dev_int_trie = build_internal_only_trie(train_words, ext_trie)
                for i, s in enumerate(dev_sents):
                    chars = list(s)
                    ext_matches = ext_trie.find_all(chars)
                    sent_fb = _sentence_domain_fallback(dev_categories[i])
                    dv = extract_domain_vec(
                        chars, ext_matches,
                        self._domain_dist_for_inference,
                        sent_fb,
                        internal_trie=dev_int_trie,
                    )
                    dev_domain_vecs.append(dv)

        # 预计算词典特征
        train_dict_vecs: Optional[List[np.ndarray]] = None
        dev_dict_vecs: Optional[List[np.ndarray]] = None
        if self._use_dict:
            # 复用与 CRF 完全相同的 OOF 流程:
            # level>=2 时需要可靠度 → 用 compute_oof_dict_vectors 得到 OOF 可靠度特征
            if self._dict_feature_level >= 2:
                train_dict_vecs_full20 = compute_oof_dict_vectors(
                    train_words, self._lexicon_extractor.trie, self._lexicon_extractor,
                    seed=config.RANDOM_SEED,
                )
                # 为 predict/dev 准备全训练集可靠度 + priors
                full_rel = compute_lexicon_reliability(
                    train_words, self._lexicon_extractor.trie,
                )
                # 深拷贝 extractor 以免影响 CRF 等共享实例
                import copy
                self._extractor_for_inference = copy.deepcopy(self._lexicon_extractor)
                self._extractor_for_inference.set_reliability(full_rel)
                full_priors = compute_class_priors(
                    self._lexicon_extractor.trie, train_words,
                )
                self._extractor_for_inference.set_class_priors(full_priors)
            else:
                # level 1: 只需 BIE, 无需可靠度
                train_sents = ["".join(w) for w in train_words]
                train_dict_vecs_full20 = [
                    self._lexicon_extractor.extract(s) for s in train_sents
                ]
                self._extractor_for_inference = self._lexicon_extractor

            # 按 level 切片 (level>=4 的外部特征与 dict_full 相同: 20 维)
            _ext_level = _dict_level_for_bilstm(self._dict_feature_level)
            train_dict_vecs = [
                _slice_dict_vec(v, _ext_level)
                for v in train_dict_vecs_full20
            ]

            # 开发集用全训练集可靠度
            if dev_words is not None and len(dev_words) > 0:
                dev_sents = ["".join(w) for w in dev_words]
                dev_dict_vecs = [
                    _slice_dict_vec(
                        self._extractor_for_inference.extract(s),
                        _ext_level,
                    )
                    for s in dev_sents
                ]

        # ---- internal-only BIE 格网 ----
        train_internal_vecs: Optional[List[np.ndarray]] = None
        dev_internal_vecs: Optional[List[np.ndarray]] = None
        if self._use_internal:
            # OOF 生成 internal-only BIE
            ext_trie = self._lexicon_extractor.trie
            train_internal_vecs = compute_oof_internal_vectors(
                train_words, ext_trie,
                seed=config.RANDOM_SEED,
            )
            # 构建 inference 用的 full internal trie
            full_internal_trie = build_internal_only_trie(train_words, ext_trie)
            self._internal_trie_for_inference = full_internal_trie
            if dev_words is not None and len(dev_words) > 0:
                dev_internal_vecs = []
                for words in dev_words:
                    chars = list("".join(words))
                    vec = np.zeros((len(chars), INTERNAL_BIE_DIM), dtype=np.float32)
                    extract_internal_bie(chars, full_internal_trie, vec, 0)
                    dev_internal_vecs.append(vec)

        # ---- gap 特征 (从无标注经书语料提取) ----
        train_gap_vecs: Optional[List[np.ndarray]] = None
        dev_gap_vecs: Optional[List[np.ndarray]] = None
        if self._use_gap:
            self._gap_feat_dim = len(self._gap_indices)  # 按 gap_level 切片后的维度
            ext = self._unlabeled_extractor
            train_sents = ["".join(w) for w in train_words]
            if self._gap_indices:
                train_gap_vecs = [
                    ext.extract(s)[:, self._gap_indices]
                    for s in train_sents
                ]
            else:
                train_gap_vecs = [
                    np.zeros((len(s), 0), dtype=np.float32)
                    for s in train_sents
                ]
            if dev_words is not None and len(dev_words) > 0:
                dev_sents = ["".join(w) for w in dev_words]
                if self._gap_indices:
                    dev_gap_vecs = [
                        ext.extract(s)[:, self._gap_indices]
                        for s in dev_sents
                    ]
                else:
                    dev_gap_vecs = [
                        np.zeros((len(s), 0), dtype=np.float32)
                        for s in dev_sents
                    ]

        # 初始化模型
        self._model = BiLSTMCRFModel(
            vocab_size=len(self._char2idx),
            tagset_size=len(self._tag2idx),
            embedding_dim=self._embedding_dim,
            hidden_dim=self._hidden_dim,
            num_layers=self._num_layers,
            dropout=self._dropout,
            dict_feat_dim=self._dict_feat_dim,
            internal_feat_dim=self._internal_feat_dim,
            domain_dist_dim=self._domain_dist_dim,
            gap_feat_dim=self._gap_feat_dim,
            dict_dropout=self._dict_dropout,
        ).to(self._device)

        # 构建 DataLoader
        train_dataset = CharDataset(
            train_words, self._char2idx, self._tag2idx,
            dict_vectors=train_dict_vecs,
            internal_vectors=train_internal_vecs,
            domain_vectors=train_domain_vecs,
            gap_vectors=train_gap_vecs,
            sample_weights=train_weights,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size,
            shuffle=True, collate_fn=collate_fn,
            pin_memory=self._device.type == "cuda",
            num_workers=0,  # 设为0保证可复现, 多worker会导致shuffle顺序非确定性
        )

        dev_loader = None
        if dev_words is not None and len(dev_words) > 0:
            dev_dataset = CharDataset(
                dev_words, self._char2idx, self._tag2idx,
                dict_vectors=dev_dict_vecs,
                internal_vectors=dev_internal_vecs,
                domain_vectors=dev_domain_vecs,
                gap_vectors=dev_gap_vecs,
            )
            dev_loader = DataLoader(
                dev_dataset, batch_size=self._batch_size * 2,
                shuffle=False, collate_fn=collate_fn,
                pin_memory=self._device.type == "cuda",
                num_workers=0,
            )

        if self._device.type == "cuda":
            # NOTE: 不覆盖 benchmark, 由 run.py 统一配置 (True=速度 / False=可复现)
            pass

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
                chars_batch, dv_batch, tags_batch, iv_batch, ddom_batch, gap_batch, weight_batch, lengths = batch
                chars_batch = chars_batch.to(self._device, non_blocking=True)
                tags_batch = tags_batch.to(self._device, non_blocking=True)
                lengths = lengths.to(self._device, non_blocking=True)
                dv_batch = dv_batch.to(self._device, non_blocking=True) if self._dict_feat_dim > 0 else None
                iv_batch = iv_batch.to(self._device, non_blocking=True) if self._internal_feat_dim > 0 else None
                ddom_batch = ddom_batch.to(self._device, non_blocking=True) if self._domain_dist_dim > 0 else None
                gap_batch = gap_batch.to(self._device, non_blocking=True) if self._gap_feat_dim > 0 else None
                weight_batch = weight_batch.to(self._device, non_blocking=True)

                optimizer.zero_grad()
                loss = self._model.neg_log_likelihood(
                    chars_batch, tags_batch, lengths, dv_batch, iv_batch, ddom_batch, gap_batch, weight_batch)
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
                        chars_batch, dv_batch, tags_batch, iv_batch, ddom_batch, gap_batch, weight_batch, lengths = batch
                        chars_batch = chars_batch.to(self._device, non_blocking=True)
                        tags_batch = tags_batch.to(self._device, non_blocking=True)
                        lengths = lengths.to(self._device, non_blocking=True)
                        dv_batch = dv_batch.to(self._device, non_blocking=True) if self._dict_feat_dim > 0 else None
                        iv_batch = iv_batch.to(self._device, non_blocking=True) if self._internal_feat_dim > 0 else None
                        ddom_batch = ddom_batch.to(self._device, non_blocking=True) if self._domain_dist_dim > 0 else None
                        gap_batch = gap_batch.to(self._device, non_blocking=True) if self._gap_feat_dim > 0 else None
                        weight_batch = weight_batch.to(self._device, non_blocking=True)
                        dev_loss = self._model.neg_log_likelihood(
                            chars_batch, tags_batch, lengths, dv_batch, iv_batch, ddom_batch, gap_batch, weight_batch)
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
            sliced = _slice_dict_vec(full_vec, _dict_level_for_bilstm(self._dict_feature_level))
            dv_tensor = torch.from_numpy(sliced).unsqueeze(0).to(self._device, non_blocking=True)

        iv_tensor = None
        if self._use_internal and self._internal_trie_for_inference is not None:
            iv = np.zeros((len(chars), INTERNAL_BIE_DIM), dtype=np.float32)
            extract_internal_bie(chars, self._internal_trie_for_inference, iv, 0)
            iv_tensor = torch.from_numpy(iv).unsqueeze(0).to(self._device, non_blocking=True)

        ddom_tensor = None
        if self._use_domain and self._domain_dist_for_inference is not None:
            ext_matches = self._lexicon_extractor.trie.find_all(chars)
            sent_fb = _sentence_domain_fallback(category)
            dvec = extract_domain_vec(
                chars, ext_matches, self._domain_dist_for_inference,
                sent_fb,
                internal_trie=self._internal_trie_for_inference,
            )
            ddom_tensor = torch.from_numpy(dvec).unsqueeze(0).to(self._device, non_blocking=True)

        gap_tensor = None
        if self._use_gap:
            gv = self._unlabeled_extractor.extract(sentence)
            if self._gap_indices:
                gv = gv[:, self._gap_indices]
            gap_tensor = torch.from_numpy(gv).unsqueeze(0).to(self._device, non_blocking=True)

        self._model.eval()
        with torch.no_grad():
            paths = self._model.decode(x, lengths, dv_tensor, iv_tensor, ddom_tensor, gap_tensor)
        bies_tags = [self._idx2tag[t] for t in paths[0][:len(chars)]]
        words = bies_to_words(chars, bies_tags)
        pos = [self._word_pos.get(w, "x") for w in words]
        return words, pos

    def predict_batch(self, sentences: List[str],
                      categories: Optional[List[str]] = None) -> Tuple[List[List[str]], List[List[str]]]:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if categories is None:
            categories = ["世俗文献"] * len(sentences)

        all_indices = []
        all_dv = []
        all_iv = []
        all_ddom = []
        all_gap = []
        all_lengths = []
        ext_trie = self._lexicon_extractor.trie if self._lexicon_extractor is not None else None

        for idx, sent in enumerate(sentences):
            chars = list(sent)
            indices = [self._char2idx.get(c, self._char2idx.get("<UNK>", 0)) for c in chars]
            all_indices.append(indices)
            all_lengths.append(len(indices))
            if self._use_dict and hasattr(self, '_extractor_for_inference'):
                full_vec = self._extractor_for_inference.extract(sent)
                all_dv.append(_slice_dict_vec(full_vec, _dict_level_for_bilstm(self._dict_feature_level)))
            if self._use_internal and self._internal_trie_for_inference is not None:
                iv = np.zeros((len(chars), INTERNAL_BIE_DIM), dtype=np.float32)
                extract_internal_bie(chars, self._internal_trie_for_inference, iv, 0)
                all_iv.append(iv)
            if self._use_domain and self._domain_dist_for_inference is not None and ext_trie is not None:
                ext_matches = ext_trie.find_all(chars)
                sent_fb = _sentence_domain_fallback(categories[idx])
                dvec = extract_domain_vec(
                    chars, ext_matches, self._domain_dist_for_inference,
                    sent_fb,
                    internal_trie=self._internal_trie_for_inference,
                )
                all_ddom.append(dvec)
            if self._use_gap:
                gv = self._unlabeled_extractor.extract(sent)
                if self._gap_indices:
                    gv = gv[:, self._gap_indices]
                all_gap.append(gv)

        sorted_idx = sorted(range(len(all_indices)), key=lambda i: all_lengths[i], reverse=True)
        max_len = all_lengths[sorted_idx[0]]
        batch_size = len(sentences)
        feat_dim = self._dict_feat_dim
        int_dim = self._internal_feat_dim
        domain_dim = self._domain_dist_dim
        gap_dim = self._gap_feat_dim

        x = torch.zeros(batch_size, max_len, dtype=torch.long)
        dv_tensor = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32) if feat_dim > 0 else None
        iv_tensor = torch.zeros(batch_size, max_len, int_dim, dtype=torch.float32) if int_dim > 0 else None
        ddom_tensor = torch.zeros(batch_size, max_len, domain_dim, dtype=torch.float32) if domain_dim > 0 else None
        gap_tensor = torch.zeros(batch_size, max_len, gap_dim, dtype=torch.float32) if gap_dim > 0 else None
        for new_i, old_i in enumerate(sorted_idx):
            L = all_lengths[old_i]
            x[new_i, :L] = torch.tensor(all_indices[old_i], dtype=torch.long)
            if dv_tensor is not None and len(all_dv) > 0:
                dv_tensor[new_i, :L] = torch.from_numpy(all_dv[old_i])
            if iv_tensor is not None and len(all_iv) > 0:
                iv_tensor[new_i, :L] = torch.from_numpy(all_iv[old_i])
            if ddom_tensor is not None and len(all_ddom) > 0:
                ddom_tensor[new_i, :L] = torch.from_numpy(all_ddom[old_i])
            if gap_tensor is not None and len(all_gap) > 0:
                gap_tensor[new_i, :L] = torch.from_numpy(all_gap[old_i])

        lengths = torch.tensor([all_lengths[i] for i in sorted_idx], dtype=torch.long)
        x = x.to(self._device, non_blocking=True)
        lengths = lengths.to(self._device, non_blocking=True)
        if dv_tensor is not None:
            dv_tensor = dv_tensor.to(self._device, non_blocking=True)
        if iv_tensor is not None:
            iv_tensor = iv_tensor.to(self._device, non_blocking=True)
        if ddom_tensor is not None:
            ddom_tensor = ddom_tensor.to(self._device, non_blocking=True)
        if gap_tensor is not None:
            gap_tensor = gap_tensor.to(self._device, non_blocking=True)

        self._model.eval()
        with torch.no_grad():
            paths = self._model.decode(x, lengths, dv_tensor, iv_tensor, ddom_tensor, gap_tensor)

        all_words = [None] * batch_size
        all_pos = [None] * batch_size
        for new_i, old_i in enumerate(sorted_idx):
            chars = list(sentences[old_i])
            bies = [self._idx2tag[t] for t in paths[new_i][:all_lengths[old_i]]]
            words = bies_to_words(chars, bies)
            pos = [self._word_pos.get(w, "x") for w in words]
            all_words[old_i] = words
            all_pos[old_i] = pos
        return all_words, all_pos

    # ---------- 持久化 ----------
    def save(self, path: str) -> None:
        import joblib
        data = {
            "model_state": self._model.state_dict() if self._model else None,
            "char2idx": self._char2idx,
            "tag2idx": self._tag2idx,
            "idx2tag": self._idx2tag,
            "word_pos": self._word_pos,
            "word_set": self._word_set,
            "config": {
                "embedding_dim": self._embedding_dim,
                "hidden_dim": self._hidden_dim,
                "num_layers": self._num_layers,
                "dropout": self._dropout,
                "dict_feature_level": self._dict_feature_level,
                "dict_feat_dim": self._dict_feat_dim,
                "internal_feat_dim": self._internal_feat_dim,
                "domain_dist_dim": self._domain_dist_dim,
                "dict_dropout": self._dict_dropout,
                "gap_feature_level": self._gap_feature_level,
                "gap_feat_dim": self._gap_feat_dim,
            },
            "extractor_for_inference": self._extractor_for_inference if hasattr(self, '_extractor_for_inference') else None,
            "internal_trie_for_inference": self._internal_trie_for_inference,
            "domain_dist_for_inference": self._domain_dist_for_inference,
            "unlabeled_extractor": self._unlabeled_extractor,
        }
        joblib.dump(data, path)

    def load(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self._char2idx = data["char2idx"]
        self._tag2idx = data["tag2idx"]
        self._idx2tag = data["idx2tag"]
        self._word_pos = data["word_pos"]
        self._word_set = data["word_set"]
        cfg = data["config"]
        self._embedding_dim = cfg["embedding_dim"]
        self._hidden_dim = cfg["hidden_dim"]
        self._num_layers = cfg["num_layers"]
        self._dropout = cfg["dropout"]
        self._dict_feature_level = cfg.get("dict_feature_level", 0)
        self._dict_feat_dim = cfg.get("dict_feat_dim", 0)
        self._internal_feat_dim = cfg.get("internal_feat_dim", 0)
        self._domain_dist_dim = cfg.get("domain_dist_dim", 0)
        self._dict_dropout = cfg.get("dict_dropout", 0.0)
        self._gap_feature_level = cfg.get("gap_feature_level", 0)
        self._gap_indices = _GAP_LEVEL_INDICES.get(self._gap_feature_level, [])
        self._gap_feat_dim = cfg.get("gap_feat_dim", len(self._gap_indices))
        self._unlabeled_extractor = data.get("unlabeled_extractor", None)
        self._extractor_for_inference = data.get("extractor_for_inference", None)
        self._internal_trie_for_inference = data.get("internal_trie_for_inference", None)
        self._domain_dist_for_inference = data.get("domain_dist_for_inference", None)

        self._model = BiLSTMCRFModel(
            vocab_size=len(self._char2idx),
            tagset_size=len(self._tag2idx),
            embedding_dim=self._embedding_dim,
            hidden_dim=self._hidden_dim,
            num_layers=self._num_layers,
            dropout=self._dropout,
            dict_feat_dim=self._dict_feat_dim,
            internal_feat_dim=self._internal_feat_dim,
            domain_dist_dim=self._domain_dist_dim,
            dict_dropout=self._dict_dropout,
        )
        if data["model_state"] is not None:
            self._model.load_state_dict(
                {k: v.to(self._device) for k, v in data["model_state"].items()})
