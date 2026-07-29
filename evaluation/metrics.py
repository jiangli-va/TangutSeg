"""评估指标 —— 分词 P/R/F1/OOV + 词性标注准确率。

所有评估以"词"为单位（而非 BIES 标签级别）。
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Metrics:
    """分词评估结果。"""
    precision: float
    recall: float
    f1: float
    num_correct: int
    num_pred: int      # 模型切出的词数
    num_gold: int       # 标准词数
    oov_rate: float = 0.0       # 测试集中 OOV 词比例
    oov_recall: float = 0.0     # OOV 召回率
    iv_recall: float = 0.0      # IV（已见词）召回率
    pos_acc_correct: int = 0    # 正确分词中词性也正确的个数
    pos_acc_total: int = 0      # 正确分词的个数（即词性评估的基数）
    pos_accuracy: float = 0.0   # 词性准确率 = pos_acc_correct / pos_acc_total
    detail: Dict = field(default_factory=dict)

    def __repr__(self):
        s = (f"P={self.precision:.4f}  R={self.recall:.4f}  "
             f"F1={self.f1:.4f}  OOV-R={self.oov_recall:.4f}  "
             f"IV-R={self.iv_recall:.4f}")
        if self.pos_acc_total > 0:
            s += f"  POS-Acc={self.pos_accuracy:.4f}"
        return s


# ======================== 核心评估函数 ========================

def evaluate(
    pred_words_list: List[List[str]],
    gold_words_list: List[List[str]],
    train_vocab: Optional[Set[str]] = None,
    pred_pos_list: Optional[List[List[str]]] = None,
    gold_pos_list: Optional[List[List[str]]] = None,
) -> Metrics:
    """评估分词 + 词性标注结果。

    Args:
        pred_words_list: 模型预测的词列表（句级）
        gold_words_list: 标准答案词列表（句级）
        train_vocab: 训练集词表（用于 OOV 分析，可选）
        pred_pos_list: 模型预测的词性列表（句级，可选）
        gold_pos_list: 标准答案词性列表（句级，可选）

    Returns:
        Metrics 对象（包含分词指标 + 词性准确率）
    """
    total_correct = 0
    total_pred = 0
    total_gold = 0
    oov_correct = 0
    oov_total = 0
    iv_correct = 0
    iv_total = 0

    # 词性评估计数
    pos_acc_correct = 0
    pos_acc_total = 0
    has_pos = pred_pos_list is not None and gold_pos_list is not None

    for sent_idx, (pred_words, gold_words) in enumerate(zip(pred_words_list, gold_words_list)):
        pred_set = _word_span_set(pred_words)
        gold_set = _word_span_set(gold_words)

        total_pred += len(pred_set)
        total_gold += len(gold_set)
        common = set(pred_set.keys()) & set(gold_set.keys())
        total_correct += len(common)

        # 词性评估：只有分词正确的词才计入
        if has_pos:
            pred_pos = pred_pos_list[sent_idx]  # type: ignore[index]
            gold_pos = gold_pos_list[sent_idx]  # type: ignore[index]
            # 构建 gold span -> (word, pos) 映射
            gold_span_to_info = _word_span_set_with_pos(gold_words, gold_pos)
            pred_span_to_info = _word_span_set_with_pos(pred_words, pred_pos)
            for span in common:
                pos_acc_total += 1
                if gold_span_to_info[span][1] == pred_span_to_info[span][1]:
                    pos_acc_correct += 1

        if train_vocab is not None:
            for span, word in gold_set.items():
                is_oov = word not in train_vocab
                if is_oov:
                    oov_total += 1
                    if span in pred_set:
                        oov_correct += 1
                else:
                    iv_total += 1
                    if span in pred_set:
                        iv_correct += 1

    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    oov_recall = oov_correct / oov_total if oov_total > 0 else 0.0
    iv_recall = iv_correct / iv_total if iv_total > 0 else 0.0
    oov_rate = oov_total / total_gold if total_gold > 0 else 0.0

    pos_accuracy = pos_acc_correct / pos_acc_total if pos_acc_total > 0 else 0.0

    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        num_correct=total_correct,
        num_pred=total_pred,
        num_gold=total_gold,
        oov_rate=oov_rate,
        oov_recall=oov_recall,
        iv_recall=iv_recall,
        pos_acc_correct=pos_acc_correct,
        pos_acc_total=pos_acc_total,
        pos_accuracy=pos_accuracy,
    )


def evaluate_with_verbose(
    pred_words_list: List[List[str]],
    gold_words_list: List[List[str]],
    train_vocab: Optional[Set[str]] = None,
    show_k: int = 5,
    pred_pos_list: Optional[List[List[str]]] = None,
    gold_pos_list: Optional[List[List[str]]] = None,
) -> Metrics:
    """带详细错误样例的评估。"""
    metrics = evaluate(pred_words_list, gold_words_list, train_vocab,
                       pred_pos_list, gold_pos_list)

    errors = []
    for i, (pred, gold) in enumerate(zip(pred_words_list, gold_words_list)):
        pred_set = _word_span_set(pred)
        gold_set = _word_span_set(gold)
        fp = [(s, w) for s, w in pred_set.items() if s not in gold_set]
        fn = [(s, w) for s, w in gold_set.items() if s not in pred_set]
        if fp or fn:
            errors.append({
                "idx": i,
                "sentence": "".join(gold),
                "gold": gold,
                "pred": pred,
                "fp": [w for _, w in fp],
                "fn": [w for _, w in fn],
            })

    metrics.detail = {
        "num_error_sents": len(errors),
        "error_samples": errors[:show_k],
    }
    return metrics


# ======================== 辅助函数 ========================

def _word_span_set(words: List[str]) -> Dict[tuple, str]:
    """将词列表转为 {(start, end): word} 的 set-like 结构。

    用于精确的 span-level 比较，避免重复词的歧义。
    """
    spans = {}
    pos = 0
    for w in words:
        spans[(pos, pos + len(w))] = w
        pos += len(w)
    return spans


def _word_span_set_with_pos(words: List[str], pos_tags: List[str]) -> Dict[tuple, Tuple[str, str]]:
    """将词列表 + 词性列表转为 {(start, end): (word, pos)} 的 set-like 结构。"""
    spans = {}
    char_pos = 0
    for w, p in zip(words, pos_tags):
        spans[(char_pos, char_pos + len(w))] = (w, p)
        char_pos += len(w)
    return spans
