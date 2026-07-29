"""TangutEncoder-S MLM 预训练。

支持:
    - 单字遮盖 (single) 和 混合遮盖 (mixed: 单字 + span)
    - 多 GPU (DataParallel) / 单 GPU / CPU
    - 按步数早停
    - 保存最佳 checkpoint
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from pretrain.data import (
    MLMDataset, collate_mlm, load_pretrain_segments, split_uuids, build_vocab,
)
from pretrain.model import TangutEncoderS, compute_mlm_loss

# ============================================================
# 默认配置
# ============================================================

DEFAULT_PRETRAIN_CONFIG = {
    "d_model": 192,
    "num_layers": 3,
    "num_heads": 4,
    "dim_feedforward": 768,
    "max_length": 128,
    "dropout": 0.15,
    "max_steps": 5000,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "grad_clip": 1.0,
    "mask_ratio": 0.15,
    "span_ratio": 0.5,  # 0.0 = only single | 0.5 = mixed
    "eval_interval": 200,
    "early_stop_patience": 5,
    "num_valid_uuids": 5,
    "min_chunk_length": 4,
    "valid_seed": 42,
    "device": "auto",
}


# ============================================================
# LR Scheduler: linear warmup + constant
# ============================================================

def _create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# 设备选择 (复用现有模式)
# ============================================================

def _resolve_device(device: str) -> torch.device:
    d = (device or "auto").lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    elif d.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device)
        print("[WARN] CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


# ============================================================
# 主训练函数
# ============================================================

def train_mlm(
    pretrain_json_path: str,
    lexicon_path: str,
    output_dir: str,
    config: Optional[dict] = None,
    mask_mode: str = "mixed",  # "single" | "mixed"
) -> Tuple[TangutEncoderS, Dict[str, List[float]]]:
    """训练 TangutEncoder-S MLM 模型。

    返回 (model, training_log) 其中 training_log 包含各步的 loss/accuracy。
    """
    cfg = {**DEFAULT_PRETRAIN_CONFIG, **(config or {})}
    if mask_mode == "single":
        cfg["span_ratio"] = 0.0

    device = _resolve_device(cfg["device"])
    print(f"[MLM] Device: {device}")
    print(f"[MLM] Mask mode: {mask_mode}, max_steps={cfg['max_steps']}, "
          f"batch_size={cfg['batch_size']}")

    # ---- 1. UUID 划分 ----
    train_uuids, valid_uuids = split_uuids(
        pretrain_json_path,
        num_valid_uuids=cfg["num_valid_uuids"],
        seed=cfg["valid_seed"],
    )
    print(f"[MLM] Train UUIDs: {len(train_uuids)}, Valid UUIDs: {len(valid_uuids)}")
    print(f"[MLM] Valid: {valid_uuids}")

    # ---- 2. 加载数据 ----
    train_chunks = load_pretrain_segments(
        pretrain_json_path, train_uuids,
        max_length=cfg["max_length"],
        min_length=cfg["min_chunk_length"],
    )
    valid_chunks = load_pretrain_segments(
        pretrain_json_path, valid_uuids,
        max_length=cfg["max_length"],
        min_length=cfg["min_chunk_length"],
    )
    print(f"[MLM] Train chunks: {len(train_chunks)}, "
          f"Valid chunks: {len(valid_chunks)}")

    # ---- 3. 构建词表 ----
    char2idx, idx2char = build_vocab(lexicon_path, pretrain_json_path)
    vocab_size = len(char2idx)
    print(f"[MLM] Vocab size: {vocab_size}")

    # ---- 4. 构建 Dataset / DataLoader ----
    train_dataset = MLMDataset(
        train_chunks, char2idx,
        max_length=cfg["max_length"],
        mask_ratio=cfg["mask_ratio"],
        span_ratio=cfg["span_ratio"],
        random_seed=None,  # 训练集每次不同
    )
    valid_dataset = MLMDataset(
        valid_chunks, char2idx,
        max_length=cfg["max_length"],
        mask_ratio=cfg["mask_ratio"],
        span_ratio=cfg["span_ratio"],
        random_seed=cfg["valid_seed"],  # 验证集固定遮盖
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, collate_fn=collate_mlm,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg["batch_size"],
        shuffle=False, collate_fn=collate_mlm,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )

    # ---- 5. 构建模型 ----
    model = TangutEncoderS(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dim_feedforward=cfg["dim_feedforward"],
        max_length=cfg["max_length"],
        dropout=cfg["dropout"],
        pad_idx=char2idx["[PAD]"],
    ).to(device)

    # 多 GPU
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[MLM] Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MLM] Total parameters: {total_params:,}")

    # ---- 6. Optimizer & Scheduler ----
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    warmup_steps = int(cfg["max_steps"] * cfg["warmup_ratio"])
    scheduler = _create_lr_scheduler(optimizer, cfg["max_steps"], warmup_steps)

    pad_idx = char2idx["[PAD]"]

    # ---- 7. Training loop ----
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_log: Dict[str, List[float]] = {
        "step": [], "train_loss": [], "train_top1": [], "train_top5": [],
        "valid_loss": [], "valid_top1": [], "valid_top5": [], "lr": [],
    }

    best_valid_loss = float("inf")
    best_step = 0
    patience_counter = 0
    global_step = 0

    # 保存词表映射 (供下游使用)
    torch.save({
        "char2idx": char2idx,
        "idx2char": idx2char,
        "config": cfg,
        "valid_uuids": valid_uuids,
    }, output_path / "vocab.pt")

    while global_step < cfg["max_steps"]:
        for batch in train_loader:
            if global_step >= cfg["max_steps"]:
                break

            model.train()
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss, top1, top5 = compute_mlm_loss(logits, labels, pad_idx)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

            optimizer.step()
            scheduler.step()
            global_step += 1

            training_log["step"].append(global_step)
            training_log["train_loss"].append(loss.item())
            training_log["train_top1"].append(top1.item())
            training_log["train_top5"].append(top5.item())
            training_log["lr"].append(optimizer.param_groups[0]["lr"])

            # ---- Validation ----
            if global_step % cfg["eval_interval"] == 0 or global_step == 1:
                model.eval()
                valid_total_loss = 0.0
                valid_total_top1 = 0.0
                valid_total_top5 = 0.0
                valid_batches = 0

                with torch.no_grad():
                    for v_batch in valid_loader:
                        v_input, v_labels = v_batch
                        v_input = v_input.to(device)
                        v_labels = v_labels.to(device)
                        v_logits, _ = model(v_input)
                        v_loss, v_top1, v_top5 = compute_mlm_loss(v_logits, v_labels, pad_idx)
                        valid_total_loss += v_loss.item()
                        valid_total_top1 += v_top1.item()
                        valid_total_top5 += v_top5.item()
                        valid_batches += 1

                avg_valid_loss = valid_total_loss / max(valid_batches, 1)
                avg_valid_top1 = valid_total_top1 / max(valid_batches, 1)
                avg_valid_top5 = valid_total_top5 / max(valid_batches, 1)

                training_log["valid_loss"].append(avg_valid_loss)
                training_log["valid_top1"].append(avg_valid_top1)
                training_log["valid_top5"].append(avg_valid_top5)

                print(
                    f"[MLM] Step {global_step}/{cfg['max_steps']}  "
                    f"train_loss={loss.item():.4f}  "
                    f"valid_loss={avg_valid_loss:.4f}  "
                    f"valid_top1={avg_valid_top1:.4f}  "
                    f"valid_top5={avg_valid_top5:.4f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

                # Early stopping
                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    best_step = global_step
                    patience_counter = 0
                    # 保存最佳 checkpoint
                    _save_checkpoint(model, char2idx, cfg, output_path / "best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= cfg["early_stop_patience"]:
                        print(f"[MLM] Early stop at step {global_step} "
                              f"(best: step {best_step}, valid_loss={best_valid_loss:.4f})")
                        break
            else:
                # 填充日志占位
                if len(training_log["valid_loss"]) < len(training_log["step"]):
                    pass  # 只在 eval step 记录

        if patience_counter >= cfg["early_stop_patience"]:
            break

    # ---- 8. 保存最终日志 ----
    print(f"[MLM] Training complete. Best step: {best_step}, "
          f"best valid_loss: {best_valid_loss:.4f}")

    # 加载最佳模型
    best_state = torch.load(output_path / "best_model.pt", map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_state["model_state_dict"])
    else:
        model.load_state_dict(best_state["model_state_dict"])

    # 保存训练日志
    with open(output_path / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)

    # 如果用了 DataParallel，返回 unwrapped model
    if isinstance(model, nn.DataParallel):
        model = model.module

    return model, training_log


def _save_checkpoint(
    model: nn.Module,
    char2idx: Dict[str, int],
    config: dict,
    path: Path,
):
    """保存 checkpoint（放到 CPU 再存）。"""
    if isinstance(model, nn.DataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    torch.save(
        {
            "model_state_dict": {k: v.cpu().clone() for k, v in state.items()},
            "char2idx": char2idx,
            "config": config,
        },
        path,
    )
