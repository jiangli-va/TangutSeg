"""Phase 2: 词感知预训练 —— MLM + Word Ranking 联合训练。

在 Phase 1 MLM 预训练基础上，加入词典词排序损失，
让 TangutEncoder 学会词级完整性。

用法:
    python -m pretrain.train_phase2 \\
        --pretrained-model output/pretrain/tangut_encoder_mixed/best_model0721.pt \\
        --output-dir output/pretrain/tangut_encoder_phase2 \\
        --lambda-word 0.3 --max-steps 3000

Config 外部接口 (通过 --config-* 系列参数覆盖):
    lambda_word       : 词损失权重, default 0.3
    lr_encoder        : 编码器学习率, default 5e-5
    lr_span_head      : Span head 学习率, default 3e-4
    max_steps         : 最大训练步数, default 3000
    batch_size        : batch 大小, default 32
    word_warmup_steps : 词损失线性 warmup 步数, default 500
    num_neg_per_pos   : 每正样本负样本数, default 5
    eval_interval     : 评估间隔, default 200
    early_stop_patience: 早停耐心, default 5
    grad_clip         : 梯度裁剪, default 1.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 确保项目根目录在 sys.path 中
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from pretrain.model import (
    TangutEncoderS, WordSpanHead,
    compute_mlm_loss, compute_word_ranking_loss,
)
from pretrain.data import (
    build_vocab, build_word_vocab, build_neg_pool,
    split_uuids, load_pretrain_segments,
    WordRankingDataset, collate_joint,
)

# ============================================================
# 默认 Phase 2 配置
# ============================================================

DEFAULT_PHASE2_CONFIG = {
    "lambda_word": 0.3,
    "lr_encoder": 5e-5,
    "lr_span_head": 3e-4,
    "max_steps": 3000,
    "batch_size": 32,
    "word_warmup_steps": 500,
    "num_neg_per_pos": 5,
    "eval_interval": 200,
    "early_stop_patience": 5,
    "grad_clip": 1.0,
    "weight_decay": 0.01,
    "neg_pool_size": 5000,
    "valid_seed": 42,
}


# ============================================================
# LR Warmup Scheduler (线性增 → 常数)
# ============================================================

def _create_lr_scheduler(optimizer, total_steps, warmup_steps):
    """线性 warmup → 常数 LR。"""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _create_word_lambda_scheduler(total_steps, warmup_steps):
    """词损失权重 warmup 调度器: 线性 0→1，之后保持 1.0。"""
    def word_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return word_lambda


# ============================================================
# 保存 / 加载辅助
# ============================================================

def _save_checkpoint(
    model: nn.Module,
    span_head: WordSpanHead,
    char2idx: Dict[str, int],
    config: dict,
    path: Path,
):
    """保存完整 checkpoint（encoder + span_head + 词表 + config）。"""
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        "model_state_dict": state,
        "span_head_state_dict": span_head.state_dict(),
        "char2idx": char2idx,
        "config": config,
    }, path)
    print(f"  [Phase2] Checkpoint saved to {path}")


# ============================================================
# 主训练函数
# ============================================================

def train_phase2(
    pretrain_json_path: str,
    lexicon_path: str,
    tongyin_json_path: str,
    pretrained_model_path: str,
    output_dir: str,
    config: dict = None,
):
    """Phase 2 词感知联合训练。

    Args:
        pretrain_json_path: 四行对译 JSON 路径
        lexicon_path: 西夏文词典 JSON 路径
        tongyin_json_path: 同音 JSON 路径
        pretrained_model_path: Phase 1 最优模型路径
        output_dir: 输出目录
        config: 训练配置 (覆盖默认值)
    """
    cfg = DEFAULT_PHASE2_CONFIG.copy()
    if config:
        cfg.update(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase2] Device: {device}")
    print(f"[Phase2] Config: {json.dumps(cfg, indent=2)}")

    # ---- 1. 加载 Phase 1 checkpoint ----
    ckpt = torch.load(pretrained_model_path, map_location="cpu")
    char2idx = ckpt["char2idx"]
    idx2char = ckpt.get("idx2char", {i: c for c, i in char2idx.items()})
    mlm_config = ckpt["config"]

    print(f"[Phase2] Loaded Phase 1 model: vocab_size={len(char2idx)}, "
          f"d_model={mlm_config['d_model']}, num_layers={mlm_config['num_layers']}")

    # ---- 2. 构建词表 ----
    word_vocab = build_word_vocab(
        lexicon_path, tongyin_json_path, char2idx,
        min_len=2, max_len=4,
    )
    first_chars = len(word_vocab)
    total_words = sum(len(v) for v in word_vocab.values())
    print(f"[Phase2] Word vocab: {first_chars} first-chars, {total_words} words")

    # ---- 3. 加载预训练数据 ----
    train_uuids, valid_uuids = split_uuids(
        pretrain_json_path,
        num_valid_uuids=mlm_config.get("num_valid_uuids", 5),
    )

    train_chunks = load_pretrain_segments(
        pretrain_json_path, train_uuids,
        max_length=mlm_config["max_length"],
        min_length=mlm_config.get("min_chunk_length", 4),
    )
    valid_chunks = load_pretrain_segments(
        pretrain_json_path, valid_uuids,
        max_length=mlm_config["max_length"],
        min_length=mlm_config.get("min_chunk_length", 4),
    )
    print(f"[Phase2] Train chunks: {len(train_chunks)}, Valid chunks: {len(valid_chunks)}")

    # ---- 4. 构建负样本池 ----
    neg_pool = build_neg_pool(
        train_chunks, word_vocab,
        pool_size=cfg["neg_pool_size"],
        min_len=2, max_len=4,
    )

    # ---- 5. 构建数据集 ----
    train_dataset = WordRankingDataset(
        train_chunks, char2idx, word_vocab, neg_pool,
        max_length=mlm_config["max_length"],
        mask_ratio=mlm_config["mask_ratio"],
        span_ratio=mlm_config["span_ratio"],
        num_neg_per_pos=cfg["num_neg_per_pos"],
        random_seed=None,  # 每个 epoch 不同
    )
    valid_dataset = WordRankingDataset(
        valid_chunks, char2idx, word_vocab, neg_pool,
        max_length=mlm_config["max_length"],
        mask_ratio=mlm_config["mask_ratio"],
        span_ratio=mlm_config["span_ratio"],
        num_neg_per_pos=cfg["num_neg_per_pos"],
        random_seed=cfg["valid_seed"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, collate_fn=collate_joint,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg["batch_size"],
        shuffle=False, collate_fn=collate_joint,
        pin_memory=device.type == "cuda",
    )

    # ---- 6. 构建模型 ----
    model = TangutEncoderS(
        vocab_size=len(char2idx),
        d_model=mlm_config["d_model"],
        num_layers=mlm_config["num_layers"],
        num_heads=mlm_config["num_heads"],
        dim_feedforward=mlm_config["dim_feedforward"],
        max_length=mlm_config["max_length"],
        dropout=mlm_config["dropout"],
        pad_idx=char2idx["[PAD]"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    span_head = WordSpanHead(
        d_model=mlm_config["d_model"],
        max_len=mlm_config["max_length"],
        dropout=mlm_config.get("dropout", 0.15),
    ).to(device)

    # ---- 7. 优化器 & 调度器 ----
    optimizer = optim.AdamW([
        {"params": model.parameters(), "lr": cfg["lr_encoder"]},
        {"params": span_head.parameters(), "lr": cfg["lr_span_head"]},
    ], weight_decay=cfg["weight_decay"])

    warmup_steps = int(cfg["max_steps"] * 0.1)
    scheduler = _create_lr_scheduler(optimizer, cfg["max_steps"], warmup_steps)
    word_lambda_fn = _create_word_lambda_scheduler(cfg["max_steps"], cfg["word_warmup_steps"])

    pad_idx = char2idx["[PAD]"]

    # ---- 8. 训练循环 ----
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_log = {
        "step": [], "train_loss": [], "train_mlm_loss": [], "train_word_loss": [],
        "train_top1": [], "train_top5": [],
        "valid_loss": [], "valid_mlm_loss": [], "valid_word_loss": [],
        "valid_top1": [], "valid_top5": [], "valid_word_acc": [],
        "lr_encoder": [], "lr_span_head": [],
    }

    best_valid_loss = float("inf")
    best_step = 0
    patience_counter = 0
    global_step = 0
    lambda_word = cfg["lambda_word"]

    while global_step < cfg["max_steps"]:
        for batch in train_loader:
            if global_step >= cfg["max_steps"]:
                break

            model.train()
            span_head.train()
            input_ids, labels, pos_spans, neg_spans, lengths = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()

            # ---- Forward: MLM ----
            logits, hidden = model(input_ids, return_hidden=True)
            mlm_loss, top1, top5 = compute_mlm_loss(logits, labels, pad_idx)

            # ---- Forward: Word Ranking ----
            all_spans = pos_spans + neg_spans
            if all_spans:
                span_scores = span_head(hidden, all_spans, lengths)
                n_pos = len(pos_spans)
                pos_scores = span_scores[:n_pos]
                neg_scores = span_scores[n_pos:]
                word_loss = compute_word_ranking_loss(
                    pos_scores, neg_scores, k=cfg["num_neg_per_pos"],
                )
            else:
                word_loss = torch.tensor(0.0, device=device)
                n_pos = 0

            # ---- 词损失 warmup ----
            w_factor = word_lambda_fn(global_step)
            total_loss = mlm_loss + lambda_word * w_factor * word_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(span_head.parameters()),
                cfg["grad_clip"],
            )
            optimizer.step()
            scheduler.step()
            global_step += 1

            # ---- 日志 ----
            training_log["step"].append(global_step)
            training_log["train_loss"].append(total_loss.item())
            training_log["train_mlm_loss"].append(mlm_loss.item())
            training_log["train_word_loss"].append(word_loss.item())
            training_log["train_top1"].append(top1.item())
            training_log["train_top5"].append(top5.item())
            training_log["lr_encoder"].append(optimizer.param_groups[0]["lr"])
            training_log["lr_span_head"].append(optimizer.param_groups[1]["lr"])

            # ---- 验证 ----
            if global_step % cfg["eval_interval"] == 0 or global_step == 1:
                model.eval()
                span_head.eval()

                valid_total_loss = 0.0
                valid_mlm_loss = 0.0
                valid_word_loss = 0.0
                valid_top1 = 0.0
                valid_top5 = 0.0
                valid_batches = 0
                total_pos = 0
                total_correct = 0

                with torch.no_grad():
                    for v_batch in valid_loader:
                        v_input, v_labels, v_pos, v_neg, v_lengths = v_batch
                        v_input = v_input.to(device)
                        v_labels = v_labels.to(device)
                        v_lengths = v_lengths.to(device)

                        v_logits, v_hidden = model(v_input, return_hidden=True)
                        v_mlm_loss, v_top1, v_top5 = compute_mlm_loss(v_logits, v_labels, pad_idx)

                        v_all = v_pos + v_neg
                        n_v_pos = len(v_pos)
                        if v_all and n_v_pos > 0:
                            v_scores = span_head(v_hidden, v_all, v_lengths)
                            v_pos_scores = v_scores[:n_v_pos]
                            v_neg_scores = v_scores[n_v_pos:]
                            v_word_loss = compute_word_ranking_loss(
                                v_pos_scores, v_neg_scores, k=cfg["num_neg_per_pos"],
                            )
                            # Accuracy: pos > neg (pairwise)
                            v_pos_scores_r = v_pos_scores.unsqueeze(1)
                            v_neg_scores_r = v_neg_scores.view(-1, cfg["num_neg_per_pos"])
                            correct = (v_pos_scores_r > v_neg_scores_r).float().mean().item()
                            total_correct += correct * n_v_pos
                            total_pos += n_v_pos
                        else:
                            v_word_loss = torch.tensor(0.0, device=device)

                        valid_total_loss += (v_mlm_loss + lambda_word * v_word_loss).item()
                        valid_mlm_loss += v_mlm_loss.item()
                        valid_word_loss += v_word_loss.item()
                        valid_top1 += v_top1.item()
                        valid_top5 += v_top5.item()
                        valid_batches += 1

                avg_loss = valid_total_loss / max(valid_batches, 1)
                avg_mlm = valid_mlm_loss / max(valid_batches, 1)
                avg_word = valid_word_loss / max(valid_batches, 1)
                avg_top1 = valid_top1 / max(valid_batches, 1)
                avg_top5 = valid_top5 / max(valid_batches, 1)
                word_acc = total_correct / max(total_pos, 1)

                training_log["valid_loss"].append(avg_loss)
                training_log["valid_mlm_loss"].append(avg_mlm)
                training_log["valid_word_loss"].append(avg_word)
                training_log["valid_top1"].append(avg_top1)
                training_log["valid_top5"].append(avg_top5)
                training_log["valid_word_acc"].append(word_acc)

                print(
                    f"[Phase2] Step {global_step}/{cfg['max_steps']}  "
                    f"loss={avg_loss:.4f}  mlm={avg_mlm:.4f}  word={avg_word:.4f}  "
                    f"top1={avg_top1:.4f}  top5={avg_top5:.4f}  "
                    f"word_acc={word_acc:.4f}  w_factor={w_factor:.2f}  "
                    f"lr_e={optimizer.param_groups[0]['lr']:.2e}  "
                    f"lr_h={optimizer.param_groups[1]['lr']:.2e}"
                )

                # Early stopping
                if avg_loss < best_valid_loss:
                    best_valid_loss = avg_loss
                    best_step = global_step
                    patience_counter = 0
                    _save_checkpoint(model, span_head, char2idx, mlm_config,
                                    output_path / "best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= cfg["early_stop_patience"]:
                        print(f"[Phase2] Early stop at step {global_step} "
                              f"(best: step {best_step}, valid_loss={best_valid_loss:.4f})")
                        break

        if patience_counter >= cfg["early_stop_patience"]:
            break

    # ---- 9. 完成 ----
    print(f"[Phase2] Training complete. Best step: {best_step}, "
          f"best valid_loss: {best_valid_loss:.4f}")

    # 保存训练日志
    with open(output_path / "training_log_phase2.json", "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)

    return model, span_head, training_log


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: 词感知预训练 (MLM + Word Ranking)",
    )
    parser.add_argument("--pretrained-model", type=str, required=True,
                        help="Phase 1 预训练模型路径")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--lambda-word", type=float, default=None,
                        help=f"词损失权重 (default: {DEFAULT_PHASE2_CONFIG['lambda_word']})")
    parser.add_argument("--lr-encoder", type=float, default=None,
                        help=f"编码器学习率 (default: {DEFAULT_PHASE2_CONFIG['lr_encoder']})")
    parser.add_argument("--lr-span-head", type=float, default=None,
                        help=f"Span head 学习率 (default: {DEFAULT_PHASE2_CONFIG['lr_span_head']})")
    parser.add_argument("--max-steps", type=int, default=None,
                        help=f"最大训练步数 (default: {DEFAULT_PHASE2_CONFIG['max_steps']})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size (default: {DEFAULT_PHASE2_CONFIG['batch_size']})")
    parser.add_argument("--word-warmup-steps", type=int, default=None,
                        help=f"词损失 warmup 步数 (default: {DEFAULT_PHASE2_CONFIG['word_warmup_steps']})")
    parser.add_argument("--num-neg-per-pos", type=int, default=None,
                        help=f"每正样本负样本数 (default: {DEFAULT_PHASE2_CONFIG['num_neg_per_pos']})")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help=f"评估间隔 (default: {DEFAULT_PHASE2_CONFIG['eval_interval']})")
    parser.add_argument("--early-stop-patience", type=int, default=None,
                        help=f"早停耐心 (default: {DEFAULT_PHASE2_CONFIG['early_stop_patience']})")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help=f"梯度裁剪 (default: {DEFAULT_PHASE2_CONFIG['grad_clip']})")

    args = parser.parse_args()

    # 固定路径 (相对于项目根目录)
    pretrain_json = str(BASE / "corpus" / "提取四行对译中的西夏字（以典籍的图片为单位）.json")
    lexicon_json = str(BASE / "corpus" / "西夏文词典.json")
    tongyin_json = str(BASE / "corpus" / "同音.json")

    # 构建 config，只覆盖用户显式传入的参数
    config_overrides = {}
    for key in DEFAULT_PHASE2_CONFIG:
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            config_overrides[key] = val

    train_phase2(
        pretrain_json_path=pretrain_json,
        lexicon_path=lexicon_json,
        tongyin_json_path=tongyin_json,
        pretrained_model_path=args.pretrained_model,
        output_dir=args.output_dir,
        config=config_overrides if config_overrides else None,
    )


if __name__ == "__main__":
    main()
