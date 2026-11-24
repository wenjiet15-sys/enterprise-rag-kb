import os
import json
import time
from typing import List, Dict, Any, Tuple
from src.bm25_retriever import BM25Retriever
from src.vector_retriever import VectorRetriever

def reciprocal_rank_fusion(dense_hits: List[Tuple[Dict[str, Any], float]], sparse_hits: List[Tuple[Dict[str, Any], float]], k: int = 60) -> List[Dict[str, Any]]:
    scores = {}
    doc_map = {}
    dense_scores = {}
    sparse_scores = {}

    for rank, item in enumerate(dense_hits):
        doc = item[0] if isinstance(item, tuple) else item
        s = item[1] if isinstance(item, tuple) else 0.0
        uid = doc['uuid']
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[uid] = doc
        dense_scores[uid] = s

    for rank, item in enumerate(sparse_hits):
        doc = item[0] if isinstance(item, tuple) else item
        s = item[1] if isinstance(item, tuple) else 0.0
        uid = doc['uuid']
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[uid] = doc
        sparse_scores[uid] = s

    sorted_uids = sorted(scores.keys(), key=lambda u: scores[u], reverse=True)
    fused = []
    for u in sorted_uids:
        d = dict(doc_map[u])
        d['rrf_score'] = round(scores[u], 5)
        d['vector_score'] = round(dense_scores.get(u, 0.0), 4)
        d['bm25_score'] = round(sparse_scores.get(u, 0.0), 4)
        fused.append(d)
    return fused

class HybridReranker:
    def __init__(self, bm25_retriever: BM25Retriever, vector_retriever: VectorRetriever):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever

    def hybrid_search(self, query: str, top_k: int = 5, domain_filter: str = None) -> List[Dict[str, Any]]:
        dense_hits = self.vector.search(query, top_k=top_k * 4, filter_domain=domain_filter)
        sparse_hits = self.bm25.search(query, top_k=top_k * 4, filter_domain=domain_filter)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60)
        return fused[:top_k]
