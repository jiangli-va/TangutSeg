"""外部分词工具统一封装 —— LTP / HanLP / stanza / trankit。

每个工具都包装为 Segmenter 子类，暴露统一的 fit/predict 接口。
fit() 用于加载预训练模型（这些工具自带模型，不需要你用语料训练）。
predict() 返回 (词列表, 词性列表)，没有词性返回 "x"。
"""

from typing import List, Tuple
from models.base import Segmenter


class LTPSegmenter(Segmenter):
    """哈工大 LTP 分词器封装（含词性标注）。"""

    name = "LTP"

    def __init__(self, model_path: str = "LTP"):
        self._model_path = model_path
        self._ltp = None

    def fit(self, train_words: List[List[str]] = None,
            train_tags: List[List[str]] = None) -> None:
        """加载 LTP 预训练模型（无需训练数据）。"""
        try:
            from ltp import LTP
            self._ltp = LTP(self._model_path)
        except ImportError:
            raise ImportError("请安装 LTP: pip install ltp")
        except Exception as e:
            raise RuntimeError(f"LTP 加载失败: {e}")

    def predict(self, sentence: str) -> Tuple[List[str], List[str]]:
        if self._ltp is None:
            raise RuntimeError("请先调用 fit() 加载模型")
        result = self._ltp.pipeline([sentence], tasks=["cws", "pos"])
        return result.cws[0], result.pos[0]

    def predict_batch(self, sentences: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
        if self._ltp is None:
            raise RuntimeError("请先调用 fit() 加载模型")
        result = self._ltp.pipeline(sentences, tasks=["cws", "pos"])
        return result.cws, result.pos


class HanLPSegmenter(Segmenter):
    """HanLP 分词器封装（含词性标注）。"""

    name = "HanLP"

    def __init__(self, model: str = None):
        self._model_name = model
        self._hanlp = None

    def fit(self, train_words: List[List[str]] = None,
            train_tags: List[List[str]] = None) -> None:
        try:
            import hanlp
            # 使用联合分词+词性标注模型
            self._hanlp = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
            self._hanlp_pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
        except ImportError:
            raise ImportError("请安装 HanLP: pip install hanlp")
        except Exception as e:
            raise RuntimeError(f"HanLP 加载失败: {e}")

    def predict(self, sentence: str) -> Tuple[List[str], List[str]]:
        if self._hanlp is None:
            raise RuntimeError("请先调用 fit() 加载模型")
        words = self._hanlp(sentence)
        pos_tags = self._hanlp_pos(words)
        return words, pos_tags

    def predict_batch(self, sentences: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
        if self._hanlp is None:
            raise RuntimeError("请先调用 fit() 加载模型")
        all_words = self._hanlp(sentences)
        all_pos = [self._hanlp_pos(words) for words in all_words]
        return all_words, all_pos


class StanzaSegmenter(Segmenter):
    """Stanford Stanza 分词器封装（含词性标注）。"""

    name = "Stanza"

    def __init__(self, lang: str = "zh"):
        self._lang = lang
        self._nlp = None

    def fit(self, train_words: List[List[str]] = None,
            train_tags: List[List[str]] = None) -> None:
        try:
            import stanza
            stanza.download(self._lang, verbose=False)
            self._nlp = stanza.Pipeline(self._lang, processors="tokenize,pos",
                                         verbose=False)
        except ImportError:
            raise ImportError("请安装 stanza: pip install stanza")
        except Exception as e:
            raise RuntimeError(f"Stanza 加载失败: {e}")

    def predict(self, sentence: str) -> Tuple[List[str], List[str]]:
        if self._nlp is None:
            raise RuntimeError("请先调用 fit() 加载模型")
        doc = self._nlp(sentence)
        words, pos = [], []
        for sent in doc.sentences:
            for token in sent.tokens:
                words.append(token.text)
            for word in sent.words:
                pos.append(word.upos if word.upos else "x")
        return words, pos


class TrankitSegmenter(Segmenter):
    """trankit 分词器封装（含词性标注）。"""

    name = "Trankit"

    def __init__(self, lang: str = "chinese"):
        self._lang = lang
        self._pipeline = None

    def fit(self, train_words: List[List[str]] = None,
            train_tags: List[List[str]] = None) -> None:
        try:
            import trankit
            self._pipeline = trankit.Pipeline(self._lang)
        except ImportError:
            raise ImportError("请安装 trankit: pip install trankit")
        except Exception as e:
            raise RuntimeError(f"Trankit 加载失败: {e}")

    def predict(self, sentence: str) -> Tuple[List[str], List[str]]:
        if self._pipeline is None:
            raise RuntimeError("请先调用 fit() 加载模型")
        doc = self._pipeline(sentence)
        words, pos = [], []
        for sent in doc["sentences"]:
            for token in sent["tokens"]:
                words.append(token["text"])
                pos.append(token.get("upos", "x"))
        return words, pos
