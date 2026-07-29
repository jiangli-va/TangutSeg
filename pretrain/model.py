"""TangutEncoder-S: 西夏文预训练字符级 Transformer 编码器。

架构:
    字符ID → Character Embedding(192) + Position Embedding(192)
          → 3层 Transformer Encoder
          → LayerNorm
          → MLM prediction head (共享权重)

参数约 250万~270万。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TangutEncoderS(nn.Module):
    """西夏文字符级 Transformer 编码器 (S=Small)。

    Args:
        vocab_size: 词表大小 (~6305)
        d_model: 隐层维度 (192)
        num_layers: Transformer 层数 (3)
        num_heads: 注意力头数 (4)
        dim_feedforward: FFN 隐层维度 (768)
        max_length: 最大序列长度 (128)
        dropout: 全局 dropout (0.15)
        pad_idx: padding token 索引
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 768,
        max_length: int = 128,
        dropout: float = 0.15,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_length = max_length

        # ---- Character Embedding ----
        self.char_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # ---- Learned Position Embedding ----
        self.position_embedding = nn.Embedding(max_length, d_model)

        # ---- Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN: 更稳定的训练
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---- Final LayerNorm ----
        self.final_norm = nn.LayerNorm(d_model)

        # ---- MLM Prediction Head ----
        # 输出层权重与输入 embedding 共享 (weight tying)
        self.mlm_head = nn.Linear(d_model, vocab_size)
        # Weight tying: mlm_head.weight 指向 char_embedding.weight
        self.mlm_head.weight = self.char_embedding.weight

        # ---- Dropout ----
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """初始化权重。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Position embedding: 正态初始化
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def _create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """创建 padding mask (True = 忽略该位置)。

        PyTorch Transformer: src_key_padding_mask shape (batch, seq_len)
        为 True 的位置被忽略。
        """
        return input_ids == self.pad_idx

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播。

        Args:
            input_ids: (batch, seq_len) 输入 token ids
            return_hidden: 如果 True，同时返回 hidden states 用于下游任务

        Returns:
            logits: (batch, seq_len, vocab_size) MLM logits
            hidden: (batch, seq_len, d_model) 编码器输出 (仅当 return_hidden=True)
        """
        batch_size, seq_len = input_ids.shape

        # ---- 1. Character Embedding ----
        char_emb = self.char_embedding(input_ids)  # (B, S, D)

        # ---- 2. Position Embedding ----
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # (B, S, D)

        # ---- 3. Combine ----
        x = char_emb + pos_emb
        x = self.dropout(x)

        # ---- 4. Transformer Encoder ----
        padding_mask = self._create_padding_mask(input_ids)
        x = self.encoder(x, src_key_padding_mask=padding_mask)  # (B, S, D)

        # ---- 5. Final LayerNorm ----
        x = self.final_norm(x)

        # ---- 6. MLM Head ----
        logits = self.mlm_head(x)  # (B, S, vocab_size)

        hidden = x if return_hidden else None
        return logits, hidden

    def get_contextual_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取上下文化 embedding（用于下游分词）。

        Args:
            input_ids: (batch, seq_len)

        Returns:
            (batch, seq_len, d_model) 上下文表示
        """
        logits, hidden = self.forward(input_ids, return_hidden=True)
        return hidden


def compute_mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 MLM 损失和准确率。

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len), padding 位置 = pad_idx
        pad_idx: padding token 索引（用于忽略）

    Returns:
        loss: MLM cross-entropy loss
        top1_acc: masked-token top-1 accuracy
        top5_acc: masked-token top-5 accuracy
    """
    # 只计算被遮盖位置的 loss
    mask = labels != pad_idx  # (B, S)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device), torch.tensor(1.0), torch.tensor(1.0)

    # Flatten
    logits_flat = logits[mask]  # (num_masked, vocab_size)
    labels_flat = labels[mask]  # (num_masked,)

    loss = F.cross_entropy(logits_flat, labels_flat)

    with torch.no_grad():
        preds = logits_flat.argmax(dim=-1)
        top1_acc = (preds == labels_flat).float().mean()

        # Top-5 accuracy
        _, top5_indices = logits_flat.topk(5, dim=-1)
        top5_acc = (top5_indices == labels_flat.unsqueeze(1)).any(dim=1).float().mean()

    return loss, top1_acc, top5_acc


# ============================================================
# 词感知 Span 打分头 (Phase 2)
# ============================================================

class WordSpanHead(nn.Module):
    """词感知 Span 打分头。

    对任意 span [i, j]（inclusive，即从 i 到 j 共 j-i+1 个字符），
    计算一个实值分数。Span 表示:
        [h_i; h_{j-1}; MeanPool(i..j); MaxPool(i..j); e_len(16)]

    输入维度: d_model × 4 + 16 = 784 (d_model=192)
    结构: Linear(784, 256) → GELU → Dropout(0.2) → Linear(256, 1)

    Args:
        d_model: 编码器隐层维度
        max_len: 最大 span 长度（用于长度嵌入表大小）
        dropout: dropout 比率
    """

    def __init__(self, d_model: int = 192, max_len: int = 128, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.len_embed = nn.Embedding(max_len + 1, 16)
        input_dim = d_model * 4 + 16  # 192*4+16 = 784

        self.scorer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.len_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        hidden: torch.Tensor,
        spans: list,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """计算 span 分数。

        Args:
            hidden: (B, S, D) 上下文表示
            spans: list of (batch_idx, start, end)，end 为 inclusive
            lengths: (B,) 每个序列的有效长度

        Returns:
            scores: (num_spans,) 每个 span 的实值分数
        """
        if not spans:
            return torch.empty(0, device=hidden.device)

        device = hidden.device
        span_reprs = []

        for batch_idx, start, end in spans:
            h = hidden[batch_idx]           # (S, D)
            L = int(lengths[batch_idx].item())

            # h_start
            h_start = h[start]

            # h_{end-1} (span 结束前一位，若 end==start 则退化为 h_start)
            h_prev = h[max(end - 1, 0)] if end > start else h_start

            # MeanPool & MaxPool over span [start, end]
            span_slice = h[start:end + 1]   # (span_len, D)
            mean_pool = span_slice.mean(dim=0)
            max_pool = span_slice.max(dim=0).values

            # Length embedding
            span_len = end - start + 1
            len_emb = self.len_embed(torch.tensor(span_len, device=device))

            span_repr = torch.cat([h_start, h_prev, mean_pool, max_pool, len_emb])
            span_reprs.append(span_repr)

        span_reprs_t = torch.stack(span_reprs)  # (N, 784)
        return self.scorer(span_reprs_t).squeeze(-1)  # (N,)


def compute_word_ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """计算词排序损失。

    L_word = -1/(|W+|·k) Σ_w⁺ Σ_k log σ(s(w⁺) - s(w⁻_k))

    Args:
        pos_scores: (num_pos,) 正样本 span 分数
        neg_scores: (num_pos * k,) 负样本 span 分数
        k: 每个正样本的负样本数

    Returns:
        loss: 标量
    """
    if pos_scores.numel() == 0:
        return torch.tensor(0.0, device=pos_scores.device)

    # Reshape neg_scores to (num_pos, k)
    neg_scores = neg_scores.view(-1, k)
    pos_scores = pos_scores.unsqueeze(1)  # (num_pos, 1)

    # 对每对 (pos, neg_k) 计算 logsigmoid
    diffs = pos_scores - neg_scores  # (num_pos, k)
    loss = -F.logsigmoid(diffs).mean()

    return loss
