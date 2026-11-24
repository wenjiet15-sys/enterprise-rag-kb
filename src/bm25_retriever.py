import os
import re
import math
from collections import Counter
from typing import List, Dict, Any, Tuple

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

    def search(self, query: str, top_k: int = 10, filter_domain: str = None) -> List[Tuple[Dict[str, Any], float]]:
        query_tokens = tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores = []
        for i, counts in enumerate(self.doc_token_counts):
            doc = self.corpus[i]
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
