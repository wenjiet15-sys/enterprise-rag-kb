import os
import re
import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from src.access_control import UserContext, can_access_document

def tokenize(text: str) -> List[str]:
    # Support both English words and Chinese characters/words
    tokens = re.findall(r'[a-zA-Z0-9_\-]+|[\u4e00-\u9fa5]', text.lower())
    return tokens

class BM25Retriever:
    """
    Lightweight, fast BM25 implementation for hybrid search.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_token_counts: List[Dict[str, int]] = []
        self.corpus_size: int = 0

    def fit(self, docs: List[Dict[str, Any]], text_key: str = 'text'):
        self.corpus = docs
        self.corpus_size = len(docs)
        self.doc_lengths = []
        self.doc_token_counts = []
        self.df = {}

        total_tokens = 0
        for doc in docs:
            tokens = tokenize(doc.get(text_key, ''))
            length = len(tokens)
            self.doc_lengths.append(length)
            total_tokens += length

            counts = Counter(tokens)
            self.doc_token_counts.append(counts)
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1

        self.avg_doc_len = (total_tokens / self.corpus_size) if self.corpus_size > 0 else 0

        # Precompute IDF
        self.idf = {}
        for term, freq in self.df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

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
        scores = []
        for i, counts in enumerate(self.doc_token_counts):
            doc = self.corpus[i]
            # 检索前硬过滤（Pre-Retrieval ACL Filtering）：无权文档不参与打分与召回
            if user_ctx is not None and not can_access_document(user_ctx, doc):
                continue
            if src_filter_l and (doc.get('source_type') or '').lower().strip() != src_filter_l:
                continue
            if filter_domain and doc.get('domain') != filter_domain:
                continue

            doc_len = self.doc_lengths[i]
            if doc_len == 0:
                continue

            score = 0.0
            for qt in query_tokens:
                if qt in counts:
                    freq = counts[qt]
                    numerator = self.idf.get(qt, 0.0) * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                    score += numerator / denominator
            if score > 0:
                scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def probe_blocked_scores(
        self,
        query: str,
        user_ctx: Optional[UserContext],
        min_score: float = 3.5,
    ) -> Tuple[int, float, float]:
        """
        静默越权探针（仅返回统计数字，绝不暴露任何无权文档标题或正文）：
        返回 (blocked_high_hit_count, top_blocked_score, top_allowed_score)。
        """
        if user_ctx is None or user_ctx.is_admin:
            return 0, 0.0, 0.0
        _stop = {
            "what", "are", "the", "for", "in", "of", "to", "a", "an", "is",
            "and", "or", "on", "with", "how", "can", "do", "does", "be", "it",
            "的", "了", "是", "在", "有", "和", "与", "吗", "呢", "啊",
        }
        query_tokens = [t for t in tokenize(query) if t not in _stop]
        if not query_tokens or self.corpus_size == 0:
            return 0, 0.0, 0.0

        blocked_count = 0
        top_blocked = 0.0
        top_allowed = 0.0
        for i, counts in enumerate(self.doc_token_counts):
            doc_len = self.doc_lengths[i]
            if doc_len == 0:
                continue
            score = 0.0
            for qt in query_tokens:
                if qt in counts:
                    freq = counts[qt]
                    numerator = self.idf.get(qt, 0.0) * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                    score += numerator / denominator
            if score <= 0:
                continue
            doc = self.corpus[i]
            if can_access_document(user_ctx, doc):
                if score > top_allowed:
                    top_allowed = score
            else:
                if score > top_blocked:
                    top_blocked = score
                if score >= min_score:
                    blocked_count += 1
        return blocked_count, round(top_blocked, 3), round(top_allowed, 3)
