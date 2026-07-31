#!/usr/bin/env python3
"""西夏文分词 HTTP 服务 —— 加载 saved_models/ 下的模型，提供 REST API。

启动:
    python serve.py --model TEnc-MLM+dict+gap --port 8000

调用:
    # 单句分词
    curl -X POST http://localhost:8000/segment \
        -H "Content-Type: application/json" \
        -d '{"text": "𘝵𗯩𗰭𗏣𗅋𘙰𗢳𗂧𗅁𗥩𗄭"}'

    # 批量分词
    curl -X POST http://localhost:8000/segment/batch \
        -H "Content-Type: application/json" \
        -d '{"sentences": ["𘝵𗯩𗰭", "𗏣𗅋𘙰"]}'

    # 交互式文档
    浏览器打开 http://localhost:8000/docs
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


# ============================================================
# 请求/响应模型
# ============================================================

class SegmentRequest(BaseModel):
    text: str = Field(..., description="待分词的西夏文句子", min_length=1)


class SegmentResponse(BaseModel):
    text: str = Field(..., description="原文")
    words: List[str] = Field(..., description="分词结果（词列表）")
    segmented: str = Field(..., description="空格分隔的分词结果")


class BatchSegmentRequest(BaseModel):
    sentences: List[str] = Field(..., description="待分词的句子列表", min_length=1)


class BatchSegmentResponse(BaseModel):
    results: List[SegmentResponse] = Field(..., description="每句的分词结果")


class HealthResponse(BaseModel):
    status: str
    model: str


# ============================================================
# 模型加载 (lifespan)
# ============================================================

_segmenter = None
_model_name = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时加载模型，关闭时释放资源。"""
    global _segmenter, _model_name
    model_name = app.state.model_name
    _model_name = model_name

    from pretrain.segmenter import TangutEncoderSegmenter
    from models.lexicon import LexiconFeatureExtractor
    from models.unlabeled_stats import UnlabeledStatsExtractor

    SAVED_MODELS = BASE / "saved_models"
    model_path = SAVED_MODELS / f"{model_name}_model.pt"
    lex_path = SAVED_MODELS / f"{model_name}_lexicon.pkl"
    gap_path = SAVED_MODELS / f"{model_name}_gap.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}\n"
            f"请先运行 train_pretrain.py --seg --save-model {model_name}"
        )

    print(f"Loading model: {model_path}")
    segmenter = TangutEncoderSegmenter()
    segmenter.load(str(model_path))

    if lex_path.exists():
        print(f"Loading lexicon: {lex_path}")
        lexicon = LexiconFeatureExtractor()
        lexicon.load_state(str(lex_path))
        segmenter.set_extractors(lexicon_extractor=lexicon)

    if gap_path.exists():
        print(f"Loading gap stats: {gap_path}")
        gap = UnlabeledStatsExtractor()
        gap.load_state(str(gap_path))
        segmenter.set_extractors(unlabeled_extractor=gap)

    _segmenter = segmenter
    print(f"Model '{model_name}' loaded. Ready.")
    yield

    # 清理
    _segmenter = None
    print("Shutdown.")


# ============================================================
# FastAPI 应用
# ============================================================

def create_app(model_name: str) -> FastAPI:
    app = FastAPI(
        title="西夏文分词服务",
        description="基于 TEnc-MLM+dict+gap 的西夏文自动分词 API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.model_name = model_name
    return app


# ============================================================
# API 路由
# ============================================================

def build_app(model_name: str) -> FastAPI:
    app = create_app(model_name)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok", model=_model_name)

    @app.post("/segment", response_model=SegmentResponse)
    async def segment(req: SegmentRequest):
        if _segmenter is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        words, _tags = _segmenter.predict(req.text)
        return SegmentResponse(
            text=req.text,
            words=words,
            segmented=" ".join(words),
        )

    @app.post("/segment/batch", response_model=BatchSegmentResponse)
    async def segment_batch(req: BatchSegmentRequest):
        if _segmenter is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        results = []
        for sent in req.sentences:
            words, _tags = _segmenter.predict(sent)
            results.append(SegmentResponse(
                text=sent,
                words=words,
                segmented=" ".join(words),
            ))
        return BatchSegmentResponse(results=results)

    return app


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="西夏文分词 HTTP 服务")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="模型名称 (saved_models/ 下的前缀)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="监听地址 (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8000,
                        help="监听端口 (default: 8000)")
    args = parser.parse_args()

    app = build_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
