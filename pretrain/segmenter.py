"""下游 BIES-CRF 分词器 —— 使用 TangutEncoder 上下文表示。

支持两种模式:
    - 无词典特征: TangutEncoder(192) → Dropout → Linear(192→4) → CRF
    - 有词典特征: TangutEncoder(192) + DictProj(17→32) → Linear(224→4) → CRF

微调两阶段:
    A) 冻结编码器，训练投影层+CRF (2-3 epochs)
    B) 解冻全量微调，编码器 lr 较低 (5e-5)
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset as TorchDataset

from models.base import Segmenter
from models.lexicon import (
    LexiconFeatureExtractor, LEVEL_INDICES,
    compute_lexicon_reliability, compute_class_priors, compute_oof_dict_vectors,
    DICT_FEATURE_DIM,
)
from models.unlabeled_stats import (
    UnlabeledStatsExtractor, GAP_FEATURE_NAMES, _GAP_LEVEL_INDICES,
)
from data.dataset import words_to_bies, bies_to_words
import config
from pretrain.model import TangutEncoderS


# ============================================================
# CRF 辅助函数 (从 bilstm_crf.py 复用)
# ============================================================

def log_sum_exp(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max, _ = x.max(dim=dim, keepdim=True)
    return x_max.squeeze(dim) + (x - x_max).exp().sum(dim=dim).log()


def lengths_to_last_idx(mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=1).long() - 1
    return lengths.clamp(min=0)


# ============================================================# Word2Vec 字符向量加载
# ============================================================

def _load_w2v_into_encoder(
    encoder: TangutEncoderS,
    char2idx: Dict[str, int],
    w2v_path: str,
) -> None:
    """将预训练的 Word2Vec 向量加载到 TangutEncoder 的 char_embedding 中。

    处理逻辑:
        1. 加载 Word2Vec 向量矩阵
        2. 计算随机初始化 char_embedding 的标准差 (target_std)
        3. Scale Normalization: w2v 向量缩放到 target_std
        4. [PAD] 保持全零 (padding_idx)
        5. [UNK]/[CLS]/[SEP]/[MASK] 随机初始化 (与 random init 同分布)
        6. char_embedding 保持可训练 (允许微调)
        7. 打印 type/token 覆盖率
    """
    w2v_data = torch.load(w2v_path, map_location="cpu", weights_only=False)
    w2v_vectors = w2v_data["vectors"]  # numpy (vocab_size, d_model)
    w2v_char_to_idx = w2v_data["char_to_idx"]

    # 计算随机初始化的标准差作为目标 scale
    with torch.no_grad():
        random_weights = encoder.char_embedding.weight.detach().clone()
    target_std = random_weights.std().item()

    # 计算 w2v 向量的标准差
    w2v_std = float(np.std(w2v_vectors))
    scale = target_std / (w2v_std + 1e-8)

    print(f"  [W2V] Scale normalization: w2v_std={w2v_std:.4f} -> target_std={target_std:.4f}, scale={scale:.4f}")

    # 构建新的 embedding 权重
    special_tokens = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
    new_weight = random_weights.clone()  # 保留随机初始化的特殊 token

    covered = 0
    missing = 0
    for char, idx in char2idx.items():
        if char in special_tokens:
            continue  # 保持随机或全零
        if char in w2v_char_to_idx:
            w2v_idx = w2v_char_to_idx[char]
            vec = torch.from_numpy(w2v_vectors[w2v_idx]).float() * scale
            new_weight[idx] = vec
            covered += 1
        else:
            missing += 1

    # 强制 [PAD] 全零
    pad_idx = char2idx.get("[PAD]", 0)
    new_weight[pad_idx] = 0.0

    total_tangut = len(char2idx) - len(special_tokens)
    type_coverage = covered / max(total_tangut, 1)

    print(f"  [W2V] Type coverage: {covered}/{total_tangut} = {type_coverage:.2%}, missing={missing}")

    # Token coverage (如果 w2v_data 中有保存)
    w2v_coverage = w2v_data.get("coverage", {})
    if w2v_coverage.get("token_coverage") is not None:
        print(f"  [W2V] Token coverage (from training): {w2v_coverage['token_coverage']:.2%}")

    if missing > 0:
        missing_chars = sorted(
            [c for c in char2idx if c not in special_tokens and c not in w2v_char_to_idx]
        )[:10]
        print(f"  [W2V] Missing chars (sample): {missing_chars}")

    # 写入
    with torch.no_grad():
        encoder.char_embedding.weight.copy_(new_weight)

    # char_embedding 保持可训练
    print(f"  [W2V] char_embedding remains TRAINABLE (will be fine-tuned)")


# ============================================================# TangutEncoder + BIES-CRF 模型
# ============================================================

class TangutEncoderBIESCRF(nn.Module):
    """TangutEncoder-S + 可选词典特征投射 + Linear + CRF。"""

    def __init__(
        self,
        encoder: TangutEncoderS,
        tagset_size: int = 4,  # B, I, E, S
        dict_feat_dim: int = 0,
        dict_proj_dim: int = 32,
        gap_feat_dim: int = 0,
        gap_proj_dim: int = 16,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_model = encoder.d_model
        self.dict_feat_dim = dict_feat_dim
        self.gap_feat_dim = gap_feat_dim
        self.pad_idx = pad_idx

        # 词典特征投射
        if dict_feat_dim > 0:
            self.dict_proj = nn.Sequential(
                nn.Linear(dict_feat_dim, dict_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.dict_dropout = nn.Dropout(0.2)  # dictionary dropout
        else:
            self.dict_proj = None
            self.dict_dropout = None

        # 无标注语料 gap 特征投射
        if gap_feat_dim > 0:
            self.gap_proj = nn.Sequential(
                nn.Linear(gap_feat_dim, gap_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.gap_dropout = nn.Dropout(0.2)  # gap dropout
        else:
            self.gap_proj = None
            self.gap_dropout = None

        # 最终 Linear output dim
        total_dim = self.d_model
        if dict_feat_dim > 0:
            total_dim += dict_proj_dim
        if gap_feat_dim > 0:
            total_dim += gap_proj_dim
        self.dropout = nn.Dropout(dropout)
        self.emission = nn.Linear(total_dim, tagset_size)

        # CRF 参数
        self.tagset_size = tagset_size
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size) * 0.01)
        self.start_transitions = nn.Parameter(torch.randn(tagset_size) * 0.01)
        self.end_transitions = nn.Parameter(torch.randn(tagset_size) * 0.01)

        # BIES 转移约束: B=0, I=1, E=2, S=3
        # 合法转移: B→I, B→E | I→E | E→B, E→S | S→B, S→S
        self._trans_mask = torch.ones(tagset_size, tagset_size, dtype=torch.bool)
        self._trans_mask[0, 0] = False  # B → B
        self._trans_mask[0, 3] = False  # B → S
        self._trans_mask[1, 0] = False  # I → B
        self._trans_mask[1, 1] = False  # I → I
        self._trans_mask[1, 3] = False  # I → S
        self._trans_mask[2, 1] = False  # E → I
        self._trans_mask[2, 2] = False  # E → E
        self._trans_mask[3, 1] = False  # S → I
        self._trans_mask[3, 2] = False  # S → E
        self.register_buffer('_trans_mask_buffer', self._trans_mask.clone(), persistent=False)

        # 句首/句尾约束
        self._start_mask = torch.tensor([True, False, False, True], dtype=torch.bool)  # B, S only
        self._end_mask = torch.tensor([False, False, True, True], dtype=torch.bool)  # E, S only
        self.register_buffer('_start_mask_buffer', self._start_mask.clone(), persistent=False)
        self.register_buffer('_end_mask_buffer', self._end_mask.clone(), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        dict_vecs: Optional[torch.Tensor] = None,
        gap_vecs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算 CRF 发射分数。

        Args:
            input_ids: (batch, seq_len)
            lengths: (batch,) 有效长度
            dict_vecs: (batch, seq_len, D_dict) 词典特征，可为 None
            gap_vecs: (batch, seq_len, D_gap) 无标注语料 gap 特征，可为 None

        Returns:
            (batch, seq_len, tagset_size) 发射分数
        """
        # 获取上下文表示
        hidden = self.encoder.get_contextual_embeddings(input_ids)  # (B, S, D)

        # 拼接词典特征
        if self.dict_feat_dim > 0 and dict_vecs is not None:
            if self.training and self.dict_dropout is not None:
                dict_vecs = self.dict_dropout(dict_vecs)
            dict_h = self.dict_proj(dict_vecs)  # (B, S, dict_proj_dim)
            hidden = torch.cat([hidden, dict_h], dim=-1)

        # 拼接 gap 特征
        if self.gap_feat_dim > 0 and gap_vecs is not None:
            if self.training and self.gap_dropout is not None:
                gap_vecs = self.gap_dropout(gap_vecs)
            gap_h = self.gap_proj(gap_vecs)  # (B, S, gap_proj_dim)
            hidden = torch.cat([hidden, gap_h], dim=-1)

        hidden = self.dropout(hidden)
        emissions = self.emission(hidden)  # (B, S, tagset_size)
        return emissions

    def _crf_log_likelihood(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
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

    def _crf_partition(
        self, emissions: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
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

    def neg_log_likelihood(
        self,
        input_ids: torch.Tensor,
        tags: torch.Tensor,
        lengths: torch.Tensor,
        dict_vecs: Optional[torch.Tensor] = None,
        gap_vecs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = input_ids != self.pad_idx
        emissions = self.forward(input_ids, lengths, dict_vecs, gap_vecs)
        log_likelihood = self._crf_log_likelihood(emissions, tags, mask)
        partition = self._crf_partition(emissions, mask)
        return (partition - log_likelihood).mean()

    def decode(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        dict_vecs: Optional[torch.Tensor] = None,
        gap_vecs: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Viterbi 解码。"""
        mask = input_ids != self.pad_idx
        emissions = self.forward(input_ids, lengths, dict_vecs, gap_vecs)
        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(
        self, emissions: torch.Tensor, mask: torch.Tensor,
    ) -> List[List[int]]:
        batch_size, seq_len, tagset_size = emissions.shape
        mask_float = mask.float()

        # 应用 BIES 转移约束: 非法转移设为 -inf
        trans_constrained = self.transitions.clone()
        trans_constrained[~self._trans_mask_buffer] = float('-inf')

        # 句首约束
        start_constrained = self.start_transitions.clone()
        start_constrained[~self._start_mask_buffer] = float('-inf')

        # 句尾约束
        end_constrained = self.end_transitions.clone()
        end_constrained[~self._end_mask_buffer] = float('-inf')

        scores = start_constrained + emissions[:, 0]
        backpointers = []
        for t in range(1, seq_len):
            next_scores = (scores.unsqueeze(2) + trans_constrained.unsqueeze(0)
                           + emissions[:, t].unsqueeze(1))
            best_scores, best_tags = next_scores.max(dim=1)
            m = mask_float[:, t].unsqueeze(1)
            scores = best_scores * m + scores * (1 - m)
            backpointers.append(best_tags)

        scores = scores + end_constrained.unsqueeze(0)
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
# Dataset
# ============================================================

class BIESDataset(TorchDataset):
    """下游分词数据集。"""

    def __init__(
        self,
        sentences_words: List[List[str]],
        char2idx: Dict[str, int],
        tag2idx: Dict[str, int],
        dict_vectors: Optional[List[np.ndarray]] = None,
        gap_vectors: Optional[List[np.ndarray]] = None,
    ):
        self.has_dict = dict_vectors is not None
        self.has_gap = gap_vectors is not None
        self.data = []
        for i, words in enumerate(sentences_words):
            chars = list("".join(words))
            bies_tags = words_to_bies(words)

            char_ids = [char2idx.get(c, char2idx.get("[UNK]", 0)) for c in chars]
            tag_ids = [tag2idx[t] for t in bies_tags]

            if self.has_dict:
                dv = dict_vectors[i]
            else:
                dv = np.zeros((len(chars), 0), dtype=np.float32)

            if self.has_gap:
                gv = gap_vectors[i]
            else:
                gv = np.zeros((len(chars), 0), dtype=np.float32)

            self.data.append((char_ids, dv, gv, tag_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_bies(batch):
    """动态填充 + 按长度降序排序。"""
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    chars_list, dv_list, gv_list, tags_list = zip(*batch)
    lengths = torch.tensor([len(c) for c in chars_list], dtype=torch.long)
    max_len = lengths[0].item()
    batch_size = len(batch)

    padded_chars = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_tags = torch.zeros(batch_size, max_len, dtype=torch.long)

    ext_feat_dim = dv_list[0].shape[1] if dv_list[0].ndim > 1 else 0
    padded_dv = torch.zeros(batch_size, max_len, max(ext_feat_dim, 0), dtype=torch.float32)

    gap_feat_dim = gv_list[0].shape[1] if gv_list[0].ndim > 1 else 0
    padded_gv = torch.zeros(batch_size, max_len, max(gap_feat_dim, 0), dtype=torch.float32)

    for i, (c, dv, gv, t) in enumerate(zip(chars_list, dv_list, gv_list, tags_list)):
        L = len(c)
        padded_chars[i, :L] = torch.tensor(c, dtype=torch.long)
        padded_tags[i, :L] = torch.tensor(t, dtype=torch.long)
        if ext_feat_dim > 0:
            padded_dv[i, :L] = torch.from_numpy(dv)
        if gap_feat_dim > 0:
            padded_gv[i, :L] = torch.from_numpy(gv)

    return padded_chars, padded_dv, padded_gv, padded_tags, lengths


# ============================================================
# Segmenter 封装
# ============================================================

class TangutEncoderSegmenter(Segmenter):
    """TangutEncoder-S + BIES-CRF 分词器。

    支持:
        - 预训练编码器 (pretrained TangutEncoder-S)
        - 随机初始化编码器对照 (Random Transformer)
        - Word2Vec 字符向量初始化 (TEnc-Char2Vec)
        - 可选词典特征
        - 两阶段微调 (A: freeze, B: full)
    """

    name = "TEnc-BIES-CRF"

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        random_encoder: bool = False,
        w2v_path: Optional[str] = None,
        vocab_size: int = 6305,
        d_model: int = 192,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 768,
        max_length: int = 128,
        encoder_dropout: float = 0.15,
        head_dropout: float = 0.2,
        dict_feature_level: int = 0,
        lexicon_extractor: Optional[LexiconFeatureExtractor] = None,
        gap_feature_level: int = 0,
        unlabeled_extractor: Optional[Any] = None,
        learning_rate_encoder: float = 5e-5,
        learning_rate_head: float = 5e-4,
        frozen_epochs: int = 3,
        batch_size: int = 32,
        epochs: int = 10000,
        device: str = "auto",
        early_stop_patience: int = 10,
        grad_clip: float = 1.0,
        lr_patience: int = 5,
        lr_factor: float = 0.3,
    ):
        self._pretrained_path = pretrained_path
        self._random_encoder = random_encoder
        self._w2v_path = w2v_path
        self._vocab_size = vocab_size
        self._d_model = d_model
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dim_feedforward = dim_feedforward
        self._max_length = max_length
        self._encoder_dropout = encoder_dropout
        self._head_dropout = head_dropout
        self._dict_feature_level = dict_feature_level
        self._lexicon_extractor = lexicon_extractor
        self._gap_feature_level = gap_feature_level
        self._unlabeled_extractor = unlabeled_extractor

        self._lr_encoder = learning_rate_encoder
        self._lr_head = learning_rate_head
        self._frozen_epochs = frozen_epochs
        self._batch_size = batch_size
        self._epochs = epochs
        self._device = self._resolve_device(device)
        self._early_stop_patience = early_stop_patience
        self._grad_clip = grad_clip
        self._lr_patience = lr_patience
        self._lr_factor = lr_factor

        self._model: Optional[TangutEncoderBIESCRF] = None
        self._char2idx: Dict[str, int] = {}
        self._idx2char: Dict[int, str] = {}
        self._tag2idx: Dict[str, int] = {"B": 0, "I": 1, "E": 2, "S": 3}
        self._idx2tag: Dict[int, str] = {0: "B", 1: "I", 2: "E", 3: "S"}
        self._extractor_for_inference = None

    @property
    def _use_dict(self) -> bool:
        return self._lexicon_extractor is not None and self._dict_feature_level > 0

    @property
    def _use_gap(self) -> bool:
        return self._unlabeled_extractor is not None and self._gap_feature_level > 0

    def _get_dict_feat_dim(self) -> int:
        indices = LEVEL_INDICES.get(min(self._dict_feature_level, 3), [])
        return len(indices)

    def _get_gap_feat_dim(self) -> int:
        indices = _GAP_LEVEL_INDICES.get(min(self._gap_feature_level, 3), [])
        return len(indices)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        d = (device or "auto").lower()
        if d == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    # ---------- 训练 ----------
    def fit(
        self,
        train_words: List[List[str]],
        train_tags: List[List[str]] = None,
        dev_words: Optional[List[List[str]]] = None,
        dev_tags: Optional[List[List[str]]] = None,
    ) -> None:
        """训练 BIES-CRF 分词器 (两阶段微调)。

        Args:
            train_words, train_tags: 训练数据
            dev_words, dev_tags: 可选验证集，用于早停。
                不传则退化为用训练 loss 做早停（不推荐）。
        """

        # ---- 加载/构建 TangutEncoder ----
        if self._pretrained_path is not None:
            checkpoint = torch.load(self._pretrained_path, map_location="cpu", weights_only=False)
            self._char2idx = checkpoint["char2idx"]
            self._idx2char = checkpoint.get("idx2char", {i: c for c, i in self._char2idx.items()})
            encoder = TangutEncoderS(
                vocab_size=len(self._char2idx),
                d_model=checkpoint["config"]["d_model"],
                num_layers=checkpoint["config"]["num_layers"],
                num_heads=checkpoint["config"]["num_heads"],
                dim_feedforward=checkpoint["config"]["dim_feedforward"],
                max_length=checkpoint["config"]["max_length"],
                dropout=checkpoint["config"]["dropout"],
                pad_idx=self._char2idx["[PAD]"],
            )
            encoder.load_state_dict(checkpoint["model_state_dict"])
            print(f"  [TEnc] Loaded pretrained encoder from {self._pretrained_path}")
        else:
            # 随机初始化 Transformer
            self._char2idx = {}
            self._idx2char = {}
            # 从训练数据构建最小词表
            char_set = {"[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]"}
            for words in train_words:
                for w in words:
                    for ch in w:
                        char_set.add(ch)
            all_chars = ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]"] + sorted(
                [c for c in char_set if c not in {"[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]"}]
            )
            self._char2idx = {ch: i for i, ch in enumerate(all_chars)}
            self._idx2char = {i: ch for ch, i in self._char2idx.items()}

            encoder = TangutEncoderS(
                vocab_size=len(self._char2idx),
                d_model=self._d_model,
                num_layers=self._num_layers,
                num_heads=self._num_heads,
                dim_feedforward=self._dim_feedforward,
                max_length=self._max_length,
                dropout=self._encoder_dropout,
                pad_idx=self._char2idx["[PAD]"],
            )
            print(f"  [TEnc] Random encoder, vocab={len(self._char2idx)}")

            # ---- Word2Vec 字符向量初始化 (TEnc-Char2Vec) ----
            if self._w2v_path is not None:
                _load_w2v_into_encoder(
                    encoder, self._char2idx, self._w2v_path,
                )

        # ---- 词典特征 ----
        dict_feat_dim = 0
        train_dict_vecs: Optional[List[np.ndarray]] = None
        if self._use_dict:
            dict_feat_dim = self._get_dict_feat_dim()
            _ext_level = min(self._dict_feature_level, 3)
            if self._dict_feature_level >= 2:
                train_dict_vecs_full = compute_oof_dict_vectors(
                    train_words, self._lexicon_extractor.trie, self._lexicon_extractor,
                    seed=config.RANDOM_SEED,
                )
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
                train_dict_vecs_full = [
                    self._lexicon_extractor.extract(s) for s in train_sents
                ]
                self._extractor_for_inference = self._lexicon_extractor

            from models.bilstm_crf import _slice_dict_vec
            train_dict_vecs = [
                _slice_dict_vec(v, _ext_level)
                for v in train_dict_vecs_full
            ]

        # ---- 无标注语料 gap 特征 ----
        gap_feat_dim = 0
        train_gap_vecs: Optional[List[np.ndarray]] = None
        if self._use_gap:
            gap_feat_dim = self._get_gap_feat_dim()
            train_sents = ["".join(w) for w in train_words]
            train_gap_vecs = [
                self._unlabeled_extractor.extract(s, gap_level=min(self._gap_feature_level, 3))
                for s in train_sents
            ]

        # ---- 构建模型 ----
        self._model = TangutEncoderBIESCRF(
            encoder=encoder,
            tagset_size=4,
            dict_feat_dim=dict_feat_dim,
            gap_feat_dim=gap_feat_dim,
            dropout=self._head_dropout,
            pad_idx=self._char2idx["[PAD]"],
        ).to(self._device)

        # ---- Dataset ----
        train_dataset = BIESDataset(
            train_words, self._char2idx, self._tag2idx,
            dict_vectors=train_dict_vecs,
            gap_vectors=train_gap_vecs,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size,
            shuffle=True, collate_fn=collate_bies,
            pin_memory=self._device.type == "cuda",
            num_workers=0,
        )

        # ---- 验证集 DataLoader (用于早停) ----
        dev_loader = None
        if dev_words is not None:
            dev_dict_vecs = None
            dev_gap_vecs = None
            if self._use_dict:
                dev_sents = ["".join(w) for w in dev_words]
                dev_full = [self._extractor_for_inference.extract(s) for s in dev_sents]
                from models.bilstm_crf import _slice_dict_vec
                dev_dict_vecs = [
                    _slice_dict_vec(v, min(self._dict_feature_level, 3))
                    for v in dev_full
                ]
            if self._use_gap:
                dev_sents = ["".join(w) for w in dev_words]
                dev_gap_vecs = [
                    self._unlabeled_extractor.extract(s, gap_level=min(self._gap_feature_level, 3))
                    for s in dev_sents
                ]
            dev_dataset = BIESDataset(
                dev_words, self._char2idx, self._tag2idx,
                dict_vectors=dev_dict_vecs,
                gap_vectors=dev_gap_vecs,
            )
            dev_loader = DataLoader(
                dev_dataset, batch_size=self._batch_size,
                shuffle=False, collate_fn=collate_bies,
                pin_memory=self._device.type == "cuda",
                num_workers=0,
            )

        # ---- 阶段 A: 冻结编码器 ----
        if self._frozen_epochs > 0:
            print(f"  [TEnc] Stage A: freeze encoder, {self._frozen_epochs} epochs")
            for param in self._model.encoder.parameters():
                param.requires_grad = False

            optimizer = optim.AdamW(
                [p for p in self._model.parameters() if p.requires_grad],
                lr=self._lr_head,
            )
            self._train_loop(train_loader, optimizer, self._frozen_epochs, "A", dev_loader)

        # ---- 阶段 B: 全量微调 ----
        print(f"  [TEnc] Stage B: full finetune")
        for param in self._model.encoder.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW([
            {"params": self._model.encoder.parameters(), "lr": self._lr_encoder},
            {"params": [p for n, p in self._model.named_parameters()
                        if not n.startswith("encoder.") and p.requires_grad],
             "lr": self._lr_head},
        ])
        self._train_loop(train_loader, optimizer, self._epochs, "B", dev_loader)

    def _train_loop(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        max_epochs: int,
        stage: str,
        dev_loader: Optional[DataLoader] = None,
    ):
        """通用训练循环，使用验证集 loss 做早停。

        若无验证集，退化为监控训练 loss（不推荐，早停几乎不触发）。
        """
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self._lr_factor,
            patience=self._lr_patience,
        )
        best_loss = float("inf")
        best_model_state: Optional[Dict] = None
        patience_counter = 0

        for epoch in range(max_epochs):
            self._model.train()
            total_loss = 0.0
            for batch in train_loader:
                chars_batch, dv_batch, gv_batch, tags_batch, lengths = batch
                chars_batch = chars_batch.to(self._device, non_blocking=True)
                tags_batch = tags_batch.to(self._device, non_blocking=True)
                lengths = lengths.to(self._device, non_blocking=True)
                dv_batch = dv_batch.to(self._device, non_blocking=True) if self._use_dict else None
                gv_batch = gv_batch.to(self._device, non_blocking=True) if self._use_gap else None

                optimizer.zero_grad()
                loss = self._model.neg_log_likelihood(
                    chars_batch, tags_batch, lengths, dv_batch, gv_batch,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / max(len(train_loader), 1)

            # 计算验证集 loss
            if dev_loader is not None:
                self._model.eval()
                dev_total = 0.0
                with torch.no_grad():
                    for batch in dev_loader:
                        chars_batch, dv_batch, gv_batch, tags_batch, lengths = batch
                        chars_batch = chars_batch.to(self._device, non_blocking=True)
                        tags_batch = tags_batch.to(self._device, non_blocking=True)
                        lengths = lengths.to(self._device, non_blocking=True)
                        dv_batch = dv_batch.to(self._device, non_blocking=True) if self._use_dict else None
                        gv_batch = gv_batch.to(self._device, non_blocking=True) if self._use_gap else None
                        loss = self._model.neg_log_likelihood(
                            chars_batch, tags_batch, lengths, dv_batch, gv_batch,
                        )
                        dev_total += loss.item()
                dev_loss = dev_total / max(len(dev_loader), 1)
                monitor_loss = dev_loss
            else:
                dev_loss = None
                monitor_loss = train_loss

            if (epoch + 1) % 5 == 0 or epoch == 0:
                if dev_loss is not None:
                    print(f"  [TEnc-{stage}] Epoch {epoch + 1}: train={train_loss:.4f}  dev={dev_loss:.4f}")
                else:
                    print(f"  [TEnc-{stage}] Epoch {epoch + 1}: loss={train_loss:.4f}")

            scheduler.step(monitor_loss)
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self._early_stop_patience:
                    print(f"  [TEnc-{stage}] Early stop at epoch {epoch + 1}, best_loss={best_loss:.4f}")
                    break

        # 恢复最佳模型
        if best_model_state is not None:
            self._model.load_state_dict(best_model_state)

    # ---------- 预测 ----------
    def predict(self, sentence: str) -> Tuple[List[str], List[str]]:
        if self._model is None:
            raise RuntimeError("Model not fitted.")

        chars = list(sentence)
        unk_idx = self._char2idx.get("[UNK]", 0)
        indices = [self._char2idx.get(c, unk_idx) for c in chars]
        x = torch.tensor([indices], dtype=torch.long).to(self._device, non_blocking=True)
        lengths = torch.tensor([len(indices)], dtype=torch.long).to(self._device, non_blocking=True)

        dv_tensor = None
        if self._use_dict and self._extractor_for_inference is not None:
            full_vec = self._extractor_for_inference.extract(sentence)
            _ext_level = min(self._dict_feature_level, 3)
            from models.bilstm_crf import _slice_dict_vec
            sliced = _slice_dict_vec(full_vec, _ext_level)
            dv_tensor = torch.from_numpy(sliced).unsqueeze(0).to(self._device, non_blocking=True)

        gv_tensor = None
        if self._use_gap:
            gv = self._unlabeled_extractor.extract(sentence, gap_level=min(self._gap_feature_level, 3))
            gv_tensor = torch.from_numpy(gv).unsqueeze(0).to(self._device, non_blocking=True)

        self._model.eval()
        with torch.no_grad():
            tag_ids = self._model.decode(x, lengths, dv_tensor, gv_tensor)[0]

        bies_tags = [self._idx2tag[tid] for tid in tag_ids]
        words = bies_to_words(chars, bies_tags)
        # 简单词性标注：全部标为 "x" (与纯分词评估兼容)
        tags = ["x"] * len(words)
        return words, tags

    # ---------- 持久化 ----------
    def save(self, path: str) -> None:
        """保存完整推理模型到 .pt 文件。

        包含:
            - 完整 TangutEncoderBIESCRF 权重 (encoder + dict/gap proj + CRF)
            - 词表 (char2idx, tag2idx)
            - 架构配置 (用于重建模型)
            - 特征级别 (dict_feature_level, gap_feature_level)
        """
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "char2idx": self._char2idx,
            "tag2idx": self._tag2idx,
            # 架构配置
            "vocab_size": len(self._char2idx),
            "d_model": self._d_model,
            "num_layers": self._num_layers,
            "num_heads": self._num_heads,
            "dim_feedforward": self._dim_feedforward,
            "max_length": self._max_length,
            "dropout": self._encoder_dropout,
            # 特征配置
            "dict_feature_dim": self._get_dict_feat_dim() if self._use_dict else 0,
            "gap_feature_dim": self._get_gap_feat_dim() if self._use_gap else 0,
            "dict_feature_level": self._dict_feature_level,
            "gap_feature_level": self._gap_feature_level,
        }, path)

    def load(self, path: str) -> None:
        """从 .pt 文件加载模型权重与词表。"""
        checkpoint = torch.load(path, map_location=self._device)
        self._char2idx = checkpoint["char2idx"]
        self._tag2idx = checkpoint["tag2idx"]
        self._idx2tag = {v: k for k, v in self._tag2idx.items()}

        # 恢复架构配置
        self._vocab_size = checkpoint.get("vocab_size", len(self._char2idx))
        self._d_model = checkpoint.get("d_model", 192)
        self._num_layers = checkpoint.get("num_layers", 3)
        self._num_heads = checkpoint.get("num_heads", 4)
        self._dim_feedforward = checkpoint.get("dim_feedforward", 768)
        self._max_length = checkpoint.get("max_length", 128)
        self._encoder_dropout = checkpoint.get("dropout", 0.15)
        self._dict_feature_level = checkpoint.get("dict_feature_level", 0)
        self._gap_feature_level = checkpoint.get("gap_feature_level", 0)

        dict_feat_dim = checkpoint.get("dict_feature_dim", 0)
        gap_feat_dim = checkpoint.get("gap_feature_dim", 0)

        # 重建编码器
        encoder = TangutEncoderS(
            vocab_size=self._vocab_size,
            d_model=self._d_model,
            num_layers=self._num_layers,
            num_heads=self._num_heads,
            dim_feedforward=self._dim_feedforward,
            max_length=self._max_length,
            dropout=self._encoder_dropout,
            pad_idx=self._char2idx.get("[PAD]", 0),
        )

        # 重建完整模型
        self._model = TangutEncoderBIESCRF(
            encoder=encoder,
            tagset_size=4,
            dict_feat_dim=dict_feat_dim,
            gap_feat_dim=gap_feat_dim,
            dropout=self._head_dropout,
            pad_idx=self._char2idx.get("[PAD]", 0),
        ).to(self._device)

        self._model.load_state_dict(checkpoint["model_state_dict"])

    def set_extractors(
        self,
        lexicon_extractor=None,
        unlabeled_extractor=None,
    ) -> None:
        """设置推理时需要的特征提取器（从 saved_models/ 的 .pkl 加载后调用）。"""
        if lexicon_extractor is not None:
            self._lexicon_extractor = lexicon_extractor
            self._extractor_for_inference = lexicon_extractor
        if unlabeled_extractor is not None:
            self._unlabeled_extractor = unlabeled_extractor
