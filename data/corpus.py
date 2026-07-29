"""西夏文语料 解析器"""

import re
from typing import List, Tuple, Optional
from pathlib import Path


class CorpusParser:
    """解析分词/词性标注语料。

    支持格式:
        西夏文:    "68.1.1  词1/词性1  词2/词性2  ..."

    输出:         List[(词列表, 词性列表), ...]
    """

    # 行首编号: 西夏文格式（如 68.1.1）
    _LINE_HEADER_RE = re.compile(
        r"\d{8}-\d{2}-\d{3}-\d{3}/\w+\s+"   # 测试人民日报语料: 19980101-01-001-001/m
        r"|\d+\.\d+\.\d+\s+"                # 西夏文:   68.1.1
    )

    # 匹配命名实体组 [xxx/pos yyy/pos]pos
    _GROUP_RE = re.compile(r"\[([^\]]+)\]([a-zA-Z.]+)")

    # 匹配普通标记: 词/词性  —— 词性可含字母、数字、下划线、点
    _TOKEN_RE = re.compile(r"(\S+?)/([a-zA-Z0-9_.!]+)")

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else None

    # ------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------
    def parse_file(self, path: Optional[Path] = None,
                   max_sentences: Optional[int] = None) -> Tuple[
                       List[List[str]], List[List[str]]]:
        """解析整个文件，返回 (all_words, all_tags) —— 各为句子级嵌套列表。"""
        ws, ts, _ = self._parse_impl(path, max_sentences, with_categories=False)
        return ws, ts

    def parse_file_with_categories(self, path: Optional[Path] = None,
                                   max_sentences: Optional[int] = None) -> Tuple[
                                       List[List[str]], List[List[str]], List[str]]:
        """解析整个文件，返回 (all_words, all_tags, categories)。

        categories 根据行首编号推断:
            "68." → "经书"
            others → "世俗文献"
        """
        return self._parse_impl(path, max_sentences, with_categories=True)

    def _parse_impl(self, path: Optional[Path] = None,
                    max_sentences: Optional[int] = None,
                    with_categories: bool = False):
        """内部解析实现。"""
        src = Path(path) if path else self._path
        if src is None:
            raise ValueError("No corpus path provided.")
        words_list, tags_list, cat_list = [], [], []
        with open(src, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                ws, ts = self.parse_line(line)
                if ws and ts:
                    words_list.append(ws)
                    tags_list.append(ts)
                    if with_categories:
                        cat_list.append(self._get_category(line))
                if max_sentences and len(words_list) >= max_sentences:
                    break
        if with_categories:
            return words_list, tags_list, cat_list
        return words_list, tags_list, []

    @staticmethod
    def _get_category(raw_line: str) -> str:
        """根据行首编号推断句子类别。

        all_cleaned.txt 由 jingshu.txt (前缀 68) + shisu.txt (前缀 N, N≠68) 拼接而成。
        """
        m = re.match(r"(\d+)\.", raw_line)
        if m:
            prefix = m.group(1)
            if prefix == "68":
                return "经书"
            else:
                return "世俗文献"
        return "未知"

    def parse_line(self, raw: str) -> Tuple[List[str], List[str]]:
        """解析一行原始文本，返回 (词语列表, 词性列表)。"""
        # Step 1: 去掉行首编号
        stripped = self._remove_line_header(raw)
        words, tags = [], []

        # Step 2: 展开命名实体组
        expanded = stripped
        for match in self._GROUP_RE.finditer(stripped):
            inner = match.group(1)
            expanded = expanded.replace(match.group(0), inner)

        # Step 3: 逐 token 解析 (词/词性)
        for m in self._TOKEN_RE.finditer(expanded):
            word = m.group(1)
            tag = m.group(2)
            if self._is_pure_meta(word, tag, stripped):
                continue
            tag = self._clean_tag(tag)
            words.append(word)
            tags.append(tag)

        # Step 4: 容错——尝试修复少数缺 "/" 的 token（如 "𗅔v" → "𗅔/v"）
        # 用启发式: 如果 words 很短但 raw 很长，说明很多 token 丢失了，尝试补全
        if len(words) < 3 and len(stripped.split()) > 3:
            recovered = self._try_recover_tokens(stripped)
            if recovered:
                return recovered

        return words, tags

    # ------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------
    def _remove_line_header(self, text: str) -> str:
        """去掉行首编号（测试人民日报 或 西夏文格式）。"""
        m = self._LINE_HEADER_RE.match(text)
        if m:
            return text[m.end():]
        return text

    def _is_pure_meta(self, word: str, tag: str, context: str) -> bool:
        """判定是否为纯编号/元信息而非真实语料。"""
        if re.match(r"\d{8}-\d{2}-\d{3}-\d{3}", word):
            return True
        if re.match(r"\d+\.\d+\.\d+", word):
            return True
        if word in ("", " ", "\u3000"):
            return True
        return False

    @staticmethod
    def _clean_tag(tag: str) -> str:
        """清理词性标签中的特殊扩展（如 vc!2 -> vc, Ng!B -> Ng）。"""
        return re.sub(r"[!^].*$", "", tag)

    # ------------------------------------------------------
    # 容错方法
    # ------------------------------------------------------
    def _try_recover_tokens(self, text: str) -> Optional[Tuple[List[str], List[str]]]:
        """尝试修复少数缺 '/' 的 token: 如 '𗅔v' → 拆为 '𗅔/v'。

        启发式: 对于每个空格分隔的段，如果在末尾能看到字母+数字词性后缀，
        就把最后一个字符前插入 '/'.
        """
        words, tags = [], []
        for seg in text.split():
            seg = seg.strip()
            if not seg:
                continue
            # 尝试找 'word/pos' 格式
            m = self._TOKEN_RE.match(seg)
            if m:
                words.append(m.group(1))
                tags.append(self._clean_tag(m.group(2)))
                continue
            # 尝试修复: 找末尾的字母序列作为词性
            m2 = re.match(r"^(\S+?)([a-zA-Z.][a-zA-Z0-9_.]*)$", seg)
            if m2:
                words.append(m2.group(1))
                tags.append(self._clean_tag(m2.group(2)))
            # 否则跳过
        if words:
            return (words, tags)
        return None
