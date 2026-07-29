"""POS 标签规范化与联合 BI+POS 标签映射。

处理语料中的拼写/格式问题，生成规范化的标签体系。
"""

import re
from typing import Dict, List, Tuple, Set
from collections import Counter


# ============================================================
# 标签规范化规则
# ============================================================

# 需要统一为带句点的标签（语法标记）
_UNIFY_DOT = {
    "Quot": "Quot.", "Quet": "Quot.",
    "Obj": "Obj.",
    "Erg": "Erg.",
    "Dir1": "Dir1.", "Dir1,": "Dir1.",
    "Dir2": "Dir2.",
    "Loc": "Loc.", "Loc,": "Loc.",
    "Nom": "Nom.",
    "Fut": "Fut.",
    "Pfv": "Pfv.",
    "1sg": "1sg.",
    "2sg": "2sg.",
    "pl": "pl.",
}

# 需要去掉多余句点的标签
_UNIFY_NO_DOT = {
    "mc.": "mc",
}


def normalize_tag(raw_tag: str) -> str:
    """规范化单个 POS 标签。

    处理步骤:
    1. 去除空白
    2. 应用 _UNIFY_DOT / _UNIFY_NO_DOT 映射
    3. 返回规范化标签

    Returns:
        规范化后的标签；纯 "?" 或空字符串返回 None 表示排除
    """
    tag = raw_tag.strip()
    if not tag or tag == "?":
        return None  # 从训练/评估中排除
    if tag in _UNIFY_DOT:
        return _UNIFY_DOT[tag]
    if tag in _UNIFY_NO_DOT:
        return _UNIFY_NO_DOT[tag]
    return tag


def normalize_tags(tags: List[str]) -> List[str]:
    """规范化整句的词性列表，过滤掉 None。"""
    result = []
    for t in tags:
        nt = normalize_tag(t)
        if nt is not None:
            result.append(nt)
    return result


# ============================================================
# 联合 BI+POS 标签
# ============================================================

BIES_LABELS = ["B", "I", "E", "S"]


def build_joint_label_map(
    tags_list: List[List[str]]
) -> Tuple[Dict[str, int], Dict[int, str], Set[str]]:
    """从语料构建 BI+POS 联合标签映射。

    Args:
        tags_list: 已规范化的词性列表，句 -> [pos1, pos2, ...]

    Returns:
        (tag2idx, idx2tag, pos_set) - 联合标签映射和 POS 集合
    """
    pos_set: Set[str] = set()
    for tags in tags_list:
        pos_set.update(tags)

    # 生成所有联合标签
    joint_tags = []
    for bies in BIES_LABELS:
        for pos in sorted(pos_set):
            joint_tags.append(f"{bies}-{pos}")

    tag2idx = {t: i for i, t in enumerate(joint_tags)}
    idx2tag = {i: t for t, i in tag2idx.items()}
    return tag2idx, idx2tag, pos_set


# ============================================================
# 联合 BI+POS 标签序列转换 (复用 data.dataset 中的函数逻辑)
# ============================================================

def words_tags_to_bies_pos(words: List[str], pos_tags: List[str]) -> List[str]:
    """将词列表 + 词性列表转为联合 BIES-POS 标签序列。

    与 data/dataset.py 中的实现相同，但使用规范化后的标签。
    """
    assert len(words) == len(pos_tags), \
        f"words({len(words)}) and pos({len(pos_tags)}) length mismatch"
    tags = []
    for w, pos in zip(words, pos_tags):
        if len(w) == 1:
            tags.append(f"S-{pos}")
        else:
            tags.append(f"B-{pos}")
            for _ in range(len(w) - 2):
                tags.append(f"I-{pos}")
            tags.append(f"E-{pos}")
    return tags


def bies_pos_to_words_tags(
    chars: List[str], bies_pos_tags: List[str]
) -> Tuple[List[str], List[str]]:
    """将联合 BIES-POS 标签序列还原为 (词列表, 词性列表)。

    与 data/dataset.py 中的实现相同。
    """
    words, pos = [], []
    buf = ""
    cur_pos = ""
    for ch, tag in zip(chars, bies_pos_tags):
        tag_type, tag_pos = tag.split("-", 1)
        if tag_type == "S":
            words.append(ch)
            pos.append(tag_pos)
        elif tag_type == "B":
            buf = ch
            cur_pos = tag_pos
        elif tag_type == "I":
            buf += ch
        elif tag_type == "E":
            buf += ch
            words.append(buf)
            pos.append(tag_pos)
            buf = ""
    if buf:
        words.append(buf)
        pos.append(cur_pos)
    return words, pos


# ============================================================
# 统计辅助
# ============================================================

def compute_pos_statistics(tags_list: List[List[str]]) -> Dict:
    """计算 POS 标签统计信息。

    Returns:
        {pos: {"count": N, "word_types": M, "ambiguous_words": K}}
    """
    pos_count = Counter()
    pos_word_count: Dict[str, Counter] = {}
    word_pos_count: Dict[str, Counter] = {}

    for words, tags in tags_list:
        for w, t in zip(words, tags):
            pos_count[t] += 1
            if t not in pos_word_count:
                pos_word_count[t] = Counter()
            pos_word_count[t][w] += 1
            if w not in word_pos_count:
                word_pos_count[w] = Counter()
            word_pos_count[w][t] += 1

    stats = {}
    for pos, count in pos_count.most_common():
        word_types = len(pos_word_count[pos])
        # 统计该 POS 中有多少个词也同时有其他 POS（一词多性）
        ambiguous = sum(
            1 for w in pos_word_count[pos]
            if len(word_pos_count.get(w, {})) > 1
        )
        stats[pos] = {
            "count": count,
            "word_types": word_types,
            "ambiguous_words": ambiguous,
        }

    return stats
