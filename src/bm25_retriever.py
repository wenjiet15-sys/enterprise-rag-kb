# -*- coding: utf-8 -*-
# src/bm25_retriever.py
import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional, Set
from src.access_control import UserContext, can_access_document


_PROBE_STOPWORDS = {
    "what", "are", "the", "for", "in", "of", "to", "a", "an", "is",
    "and", "or", "on", "with", "how", "can", "do", "does", "be", "it",
    "的", "了", "是", "在", "有", "和", "与", "吗", "呢", "啊",
}


def tokenize(text: str) -> List[str]:
    """中英文混合词法分词（保留工单号/集群代号等完整 token + 中文单字）"""
    if not text:
        return []
    return re.findall(r'[a-zA-Z0-9_\-]+|[\u4e00-\u9fa5]', text.lower())


class BM25Retriever:
    """
    基于倒排索引的高性能 BM25 检索器：
    - 支持 Chunk 级（万级切片毫秒检索）与检索前 RBAC/ACL 硬过滤
    - 支持运行时热增量追加 (add_docs)、按父文档删除 (remove_by_parent_id) 与实时 ACL 更新
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.text_key: str = "enriched_text"
        self.corpus: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.total_tokens: int = 0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.corpus_size: int = 0

    def _get_doc_text(self, doc: Dict[str, Any]) -> str:
        return doc.get(self.text_key) or doc.get("text", "") or ""

    def _recompute_idf(self) -> None:
        self.avg_doc_len = (self.total_tokens / self.corpus_size) if self.corpus_size > 0 else 0.0
        n = self.corpus_size
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
            if freq > 0
        }

    def fit(self, docs: List[Dict[str, Any]], text_key: str = "enriched_text") -> None:
        self.text_key = text_key
        self.corpus = list(docs)
        self.corpus_size = len(self.corpus)
        self.doc_lengths = []
        self.total_tokens = 0
        self.df = {}
        self.inverted_index = defaultdict(list)

        for idx, doc in enumerate(self.corpus):
            tokens = tokenize(self._get_doc_text(doc))
            length = len(tokens)
            self.doc_lengths.append(length)
            self.total_tokens += length

            counts = Counter(tokens)
            for term, freq in counts.items():
                self.df[term] = self.df.get(term, 0) + 1
                self.inverted_index[term].append((idx, freq))

        self._recompute_idf()

    def add_docs(self, new_docs: List[Dict[str, Any]]) -> int:
        """实时热增量追加新 Chunk 到倒排索引，无需重启服务"""
        if not new_docs:
            return 0
        start_idx = len(self.corpus)
        for offset, doc in enumerate(new_docs):
            idx = start_idx + offset
            self.corpus.append(doc)
            tokens = tokenize(self._get_doc_text(doc))
            length = len(tokens)
            self.doc_lengths.append(length)
            self.total_tokens += length

            counts = Counter(tokens)
            for term, freq in counts.items():
                self.df[term] = self.df.get(term, 0) + 1
                self.inverted_index[term].append((idx, freq))

        self.corpus_size = len(self.corpus)
        self._recompute_idf()
        return len(new_docs)

    def remove_by_parent_id(self, parent_doc_id: str) -> int:
        """按父文档 ID 删除关联全部 Chunk 并重建紧凑倒排索引"""
        if not parent_doc_id:
            return 0
        kept = [
            d for d in self.corpus
            if (d.get("parent_doc_id") or d.get("uuid")) != parent_doc_id
        ]
        removed = len(self.corpus) - len(kept)
        if removed > 0:
            self.fit(kept, text_key=self.text_key)
        return removed

    def update_acl_by_parent_id(
        self,
        parent_doc_id: str,
        security_level: int,
        source_type: Optional[str] = None,
    ) -> int:
        updated = 0
        for d in self.corpus:
            if (d.get("parent_doc_id") or d.get("uuid")) == parent_doc_id:
                d["security_level"] = int(security_level)
                if source_type:
                    d["source_type"] = source_type.lower().strip()
                updated += 1
        return updated

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_domain: Optional[str] = None,
        filter_source: Optional[str] = None,
        user_ctx: Optional[UserContext] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        query_tokens = tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        src_filter_l = filter_source.lower().strip() if filter_source else None
        avg_len = self.avg_doc_len or 1.0
        scores_map: Dict[int, float] = {}
        allowed_cache: Dict[int, bool] = {}

        for qt in query_tokens:
            postings = self.inverted_index.get(qt)
            if not postings:
                continue
            idf_val = self.idf.get(qt, 0.0)
            if idf_val <= 0.0:
                continue
            for idx, freq in postings:
                ok = allowed_cache.get(idx)
                if ok is None:
                    doc = self.corpus[idx]
                    # 检索前硬过滤（Pre-Retrieval ACL Filtering）：无权 Chunk 不参与打分与召回
                    ok = True
                    if user_ctx is not None and not can_access_document(user_ctx, doc):
                        ok = False
                    elif src_filter_l and (doc.get("source_type") or "").lower().strip() != src_filter_l:
                        ok = False
                    elif filter_domain and doc.get("domain") != filter_domain:
                        ok = False
                    allowed_cache[idx] = ok
                if not ok:
                    continue

                doc_len = self.doc_lengths[idx]
                if doc_len == 0:
                    continue
                numerator = idf_val * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / avg_len))
                scores_map[idx] = scores_map.get(idx, 0.0) + (numerator / denominator)

        if not scores_map:
            return []

        ranked = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.corpus[idx], score) for idx, score in ranked]

    def probe_blocked_scores(
        self,
        query: str,
        user_ctx: Optional[UserContext],
        min_score: float = 3.5,
    ) -> Tuple[int, float, float]:
        """
        静默越权探针（基于倒排索引毫秒级计算，仅返回统计数字，绝不暴露无权文档标题或正文）：
        返回 (blocked_high_hit_doc_count, top_blocked_score, top_allowed_score)。
        """
        if user_ctx is None or user_ctx.is_admin:
            return 0, 0.0, 0.0
        query_tokens = [t for t in tokenize(query) if t not in _PROBE_STOPWORDS]
        if not query_tokens or self.corpus_size == 0:
            return 0, 0.0, 0.0

        avg_len = self.avg_doc_len or 1.0
        scores_map: Dict[int, float] = {}
        for qt in query_tokens:
            postings = self.inverted_index.get(qt)
            if not postings:
                continue
            idf_val = self.idf.get(qt, 0.0)
            if idf_val <= 0.0:
                continue
            for idx, freq in postings:
                doc_len = self.doc_lengths[idx]
                if doc_len == 0:
                    continue
                numerator = idf_val * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / avg_len))
                scores_map[idx] = scores_map.get(idx, 0.0) + (numerator / denominator)

        blocked_parent_ids: Set[str] = set()
        top_blocked = 0.0
        top_allowed = 0.0
        for idx, score in scores_map.items():
            doc = self.corpus[idx]
            if can_access_document(user_ctx, doc):
                if score > top_allowed:
                    top_allowed = score
            else:
                if score > top_blocked:
                    top_blocked = score
                if score >= min_score:
                    blocked_parent_ids.add(doc.get("parent_doc_id") or doc.get("uuid") or str(idx))

        return len(blocked_parent_ids), round(top_blocked, 3), round(top_allowed, 3)
