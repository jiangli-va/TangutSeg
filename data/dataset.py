"""数据集抽象 —— 训练/开发/测试集划分、BIES 标注转换。

BIES 标注方案:
    B = 词首 (Begin)
    I = 词中 (Inside)
    E = 词尾 (End)
    S = 单字词 (Single)

联合 BI+POS 标注方案（用于 CRF 同时做分词+词性标注）:
    标签格式: BIES标签-POS标签
    例如: "S-n" = 单字名词, "B-v" = 动词词首, "E-v" = 动词词尾
"""

import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


# ======================== 数据结构 ========================

@dataclass
class Dataset:
    """统一的数据容器。"""

    train_words: List[List[str]]   # 训练集：句 -> 词列表
    train_tags: List[List[str]]    # 训练集：句 -> 词性列表
    dev_words: List[List[str]]
    dev_tags: List[List[str]]
    test_words: List[List[str]]
    test_tags: List[List[str]]
    categories: List[str] = field(default_factory=list)  # 测试集每句的类别（如 "经书"/"世俗文献"）
    train_categories: List[str] = field(default_factory=list)  # 训练集每句的类别
    dev_categories: List[str] = field(default_factory=list)    # 开发集每句的类别

    # 预计算缓存
    _train_bies: Optional[List[List[str]]] = field(default=None, repr=False)
    _dev_bies: Optional[List[List[str]]] = field(default=None, repr=False)
    _test_bies: Optional[List[List[str]]] = field(default=None, repr=False)

    _train_bies_pos: Optional[List[List[str]]] = field(default=None, repr=False)
    _dev_bies_pos: Optional[List[List[str]]] = field(default=None, repr=False)
    _test_bies_pos: Optional[List[List[str]]] = field(default=None, repr=False)

    @property
    def train_bies(self) -> List[List[str]]:
        if self._train_bies is None:
            self._train_bies = [words_to_bies(ws) for ws in self.train_words]
        return self._train_bies

    @property
    def dev_bies(self) -> List[List[str]]:
        if self._dev_bies is None:
            self._dev_bies = [words_to_bies(ws) for ws in self.dev_words]
        return self._dev_bies

    @property
    def test_bies(self) -> List[List[str]]:
        if self._test_bies is None:
            self._test_bies = [words_to_bies(ws) for ws in self.test_words]
        return self._test_bies

    @property
    def train_bies_pos(self) -> List[List[str]]:
        if self._train_bies_pos is None:
            self._train_bies_pos = [words_tags_to_bies_pos(ws, ts) for ws, ts in zip(self.train_words, self.train_tags)]
        return self._train_bies_pos

    @property
    def dev_bies_pos(self) -> List[List[str]]:
        if self._dev_bies_pos is None:
            self._dev_bies_pos = [words_tags_to_bies_pos(ws, ts) for ws, ts in zip(self.dev_words, self.dev_tags)]
        return self._dev_bies_pos

    @property
    def test_bies_pos(self) -> List[List[str]]:
        if self._test_bies_pos is None:
            self._test_bies_pos = [words_tags_to_bies_pos(ws, ts) for ws, ts in zip(self.test_words, self.test_tags)]
        return self._test_bies_pos

    @property
    def train_sents(self) -> List[str]:
        """训练集原始字符串列表。"""
        return ["".join(ws) for ws in self.train_words]

    @property
    def dev_sents(self) -> List[str]:
        return ["".join(ws) for ws in self.dev_words]

    @property
    def test_sents(self) -> List[str]:
        return ["".join(ws) for ws in self.test_words]


# ======================== BIES 转换 ========================

def words_to_bies(words: List[str]) -> List[str]:
    """将词列表转为 BIES 标签列表。

    Example:
        ["我", "喜欢", "吃", "苹果"] -> ["S", "B", "E", "B", "E"]
    """
    tags = []
    for w in words:
        if len(w) == 1:
            tags.append("S")
        else:
            tags.append("B")
            for _ in range(len(w) - 2):
                tags.append("I")
            tags.append("E")
    return tags


def bies_to_words(chars: List[str], bies_tags: List[str]) -> List[str]:
    """将 BIES 标签序列 + 字符序列还原为词列表。

    >>> bies_to_words(list("我喜欢吃苹果"), ["S","B","E","S","B","E"])
    ["我", "喜欢", "吃", "苹果"]
    """
    words = []
    buf = ""
    for ch, tag in zip(chars, bies_tags):
        if tag == "S":
            words.append(ch)
        elif tag == "B":
            buf = ch
        elif tag == "I":
            buf += ch
        elif tag == "E":
            buf += ch
            words.append(buf)
            buf = ""
    if buf:
        words.append(buf)
    return words


# ======================== 联合 BI+POS 标签转换 ========================

def words_tags_to_bies_pos(words: List[str], pos_tags: List[str]) -> List[str]:
    """将词列表 + 词性列表转为联合 BIES-POS 标签序列。

    Example:
        words=["我","喜欢","吃","苹果"], pos=["r","v","v","n"]
        -> ["S-r", "B-v", "E-v", "S-v", "B-n", "E-n"]
    """
    assert len(words) == len(pos_tags), f"words({len(words)}) and pos({len(pos_tags)}) length mismatch"
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

    Example:
        chars=list("我喜欢吃苹果"), tags=["S-r","B-v","E-v","S-v","B-n","E-n"]
        -> (["我","喜欢","吃","苹果"], ["r","v","v","n"])
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


# ======================== 数据划分 ========================

def split_dataset(
    words_list: List[List[str]],
    tags_list: List[List[str]],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
    categories: Optional[List[str]] = None,
) -> Dataset:
    """将语料划分为训练/开发/测试集。

    如果提供 categories，则按类别做分层抽样：每类内部独立打乱并切分，
    保证 train/dev/test 中各类比例与总体一致。
    """
    assert 0 < train_ratio + dev_ratio <= 1.0, "比例之和 <= 1.0"
    rng = random.Random(seed)

    if categories is not None and len(categories) == len(words_list):
        # ---- 分层抽样: 每类独立 shuffle + split，再合并 ----
        cat_groups: Dict[str, List[Tuple]] = {}
        for ws, ts, cat in zip(words_list, tags_list, categories):
            cat_groups.setdefault(cat, []).append((ws, ts, cat))

        train_all, dev_all, test_all = [], [], []
        train_cats, dev_cats, test_cats = [], [], []
        for cat, items in cat_groups.items():
            rng.shuffle(items)
            n = len(items)
            n_train = max(1, int(n * train_ratio))
            n_dev = max(1, int(n * dev_ratio))
            n_test = n - n_train - n_dev
            if n_test < 1:
                # 过小的类别全部放入 train
                n_train, n_dev = n, 0

            train_all.extend(items[:n_train])
            dev_all.extend(items[n_train:n_train + n_dev])
            test_all.extend(items[n_train + n_dev:])
            train_cats.extend([cat] * n_train)
            dev_cats.extend([cat] * min(n_dev, n - n_train))
            test_cats.extend([cat] * max(0, n - n_train - n_dev))

        # 再全局打乱（破坏类别聚集）
        def _shuffle_and_unzip(items, cat_labels):
            combined = list(zip(items, cat_labels))
            rng.shuffle(combined)
            ws = [it[0][0] for it in combined]
            ts = [it[0][1] for it in combined]
            cs = [it[1] for it in combined]
            return ws, ts, cs

        tw, tt, tc = _shuffle_and_unzip(train_all, train_cats)
        dw, dt, dc = _shuffle_and_unzip(dev_all, dev_cats)
        tew, tet, tec = _shuffle_and_unzip(test_all, test_cats)

        return Dataset(
            train_words=tw, train_tags=tt,
            dev_words=dw, dev_tags=dt,
            test_words=tew, test_tags=tet,
            categories=tec,  # 只保留 test 的类别（评估时用）
            train_categories=tc,
            dev_categories=dc,
        )
    else:
        # ---- 原有逻辑: 不分层 ----
        pairs = list(zip(words_list, tags_list))
        rng.shuffle(pairs)

        n = len(pairs)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)

        train_pairs = pairs[:n_train]
        dev_pairs = pairs[n_train:n_train + n_dev]
        test_pairs = pairs[n_train + n_dev:]

        def _unzip(ps):
            ws = [p[0] for p in ps]
            ts = [p[1] for p in ps]
            return ws, ts

        tw, tt = _unzip(train_pairs)
        dw, dt = _unzip(dev_pairs)
        tew, tet = _unzip(test_pairs)

        return Dataset(
            train_words=tw, train_tags=tt,
            dev_words=dw, dev_tags=dt,
            test_words=tew, test_tags=tet,
            categories=[],
        )


def make_kfolds(
    words_list: List[List[str]],
    tags_list: List[List[str]],
    k: int = 5,
    dev_ratio_in_train: float = 0.111,
    seed: int = 42,
    categories: Optional[List[str]] = None,
) -> List[Dataset]:
    """生成 k 折交叉验证数据集列表。

    如果提供 categories，则按类别做分层 K 折：每类独立 shuffle + 分折后合并，
    保证每折中各类比例与总体一致。test_categories 存入 Dataset。

    外层 K 折：全部句子随机分成 k 折，轮流用 1 折做 test、其余 k-1 折做训练池。
    训练池内部再切出一小块 dev 供 BiLSTM 早停（默认 ~1/9，使 train:dev:test≈8:1:1）。
    test 折对训练过程完全不可见。

    Args:
        words_list: 全部句子的词列表
        tags_list: 全部句子的词性列表
        k: 折数
        dev_ratio_in_train: 从每折训练池中划出的 dev 比例
        seed: 随机种子（保证每次划分一致、各方法用同一划分）
        categories: 每句话的类别标签列表（可选，用于分层 K 折）

    Returns:
        长度为 k 的 Dataset 列表，第 i 个元素以第 i 折为测试集。
    """
    assert k >= 2, "k 必须 >= 2"
    rng = random.Random(seed)

    if categories is not None and len(categories) == len(words_list):
        # ---- 分层 K 折 ----
        # 每类独立 shuffle + 分折
        cat_groups: Dict[str, List[Tuple]] = {}
        for ws, ts, cat in zip(words_list, tags_list, categories):
            cat_groups.setdefault(cat, []).append((ws, ts, cat))
        all_folds: List[List[Tuple]] = [[] for _ in range(k)]
        for cat, items in cat_groups.items():
            rng.shuffle(items)
            n_cat = len(items)
            # 按连续切分
            bounds = [round(i * n_cat / k) for i in range(k + 1)]
            for i in range(k):
                all_folds[i].extend(items[bounds[i]:bounds[i + 1]])

        # 每折内再 shuffle 破坏类别聚集
        for i in range(k):
            rng.shuffle(all_folds[i])

        def _unzip_with_cat(pairs):
            ws = [p[0] for p in pairs]
            ts = [p[1] for p in pairs]
            cs = [p[2] for p in pairs]
            return ws, ts, cs
    else:
        # ---- 随机 K 折 ----
        pairs = list(zip(words_list, tags_list))
        rng.shuffle(pairs)
        n = len(pairs)
        bounds = [round(i * n / k) for i in range(k + 1)]
        all_folds = [pairs[bounds[i]:bounds[i + 1]] for i in range(k)]
        categories = None

    def _unzip_plain(ps):
        return [p[0] for p in ps], [p[1] for p in ps]

    datasets: List[Dataset] = []
    for i in range(k):
        test_items = all_folds[i]
        train_pool = [item for j in range(k) if j != i for item in all_folds[j]]
        dev_rng = random.Random(seed + 1000 + i)
        dev_rng.shuffle(train_pool)
        n_dev = max(1, int(len(train_pool) * dev_ratio_in_train))
        dev_items = train_pool[:n_dev]
        train_items = train_pool[n_dev:]

        if categories is not None:
            tw, tt, tc = _unzip_with_cat(train_items)
            dw, dt, dc = _unzip_with_cat(dev_items)
            tew, tet, tec = _unzip_with_cat(test_items)
        else:
            tw, tt = _unzip_plain(train_items)
            dw, dt = _unzip_plain(dev_items)
            tew, tet = _unzip_plain(test_items)
            tec, tc, dc = [], [], []

        datasets.append(Dataset(
            train_words=tw, train_tags=tt,
            dev_words=dw, dev_tags=dt,
            test_words=tew, test_tags=tet,
            categories=tec,
            train_categories=tc,
            dev_categories=dc,
        ))
    return datasets


# ======================== 训练集统计 ========================

def compute_vocabulary_stats(words_list: List[List[str]]) -> Dict:
    """从词列表计算基本统计。"""
    all_words = [w for ws in words_list for w in ws]
    all_chars = [c for w in all_words for c in w]
    return {
        "num_sentences": len(words_list),
        "num_tokens": len(all_words),
        "num_chars": len(all_chars),
        "unique_words": len(set(all_words)),
        "unique_chars": len(set(all_chars)),
        "avg_sentence_len_tokens": len(all_words) / max(len(words_list), 1),
        "avg_sentence_len_chars": len(all_chars) / max(len(words_list), 1),
    }
