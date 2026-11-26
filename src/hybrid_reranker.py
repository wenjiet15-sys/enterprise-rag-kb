# -*- coding: utf-8 -*-
# src/hybrid_reranker.py
from typing import List, Dict, Any, Tuple, Optional
from src.bm25_retriever import BM25Retriever
from src.vector_retriever import VectorRetriever
from src.access_control import UserContext, can_access_document


def reciprocal_rank_fusion(
    dense_hits: List[Tuple[Dict[str, Any], float]],
    sparse_hits: List[Tuple[Dict[str, Any], float]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Chunk 级倒数排名融合 (RRF)：
    使用 chunk_id（兼容旧 uuid）作为唯一键，融合稠密向量与 BM25 词法排序。
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}
    dense_scores: Dict[str, float] = {}
    sparse_scores: Dict[str, float] = {}

    for rank, item in enumerate(dense_hits):
        doc = item[0] if isinstance(item, tuple) else item
        s = item[1] if isinstance(item, tuple) else 0.0
        uid = doc.get("chunk_id") or doc["uuid"]
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[uid] = doc
        dense_scores[uid] = s

    for rank, item in enumerate(sparse_hits):
        doc = item[0] if isinstance(item, tuple) else item
        s = item[1] if isinstance(item, tuple) else 0.0
        uid = doc.get("chunk_id") or doc["uuid"]
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[uid] = doc
        sparse_scores[uid] = s

    sorted_uids = sorted(scores.keys(), key=lambda u: scores[u], reverse=True)
    fused = []
    for u in sorted_uids:
        d = dict(doc_map[u])
        d["rrf_score"] = round(scores[u], 5)
        d["vector_score"] = round(dense_scores.get(u, 0.0), 4)
        d["bm25_score"] = round(sparse_scores.get(u, 0.0), 4)
        fused.append(d)
    return fused


class HybridReranker:
    def __init__(self, bm25_retriever: BM25Retriever, vector_retriever: VectorRetriever):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        user_ctx: Optional[UserContext] = None,
        max_chunks_per_doc: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Chunk 级双路混合召回 + RRF 融合 + 单篇父文档防霸屏配额 (max_chunks_per_doc)
        """
        cand_k = max(top_k * 5, 20)
        dense_hits = self.vector.search(
            query,
            top_k=cand_k,
            filter_domain=domain_filter,
            filter_source=source_filter,
            user_ctx=user_ctx,
        )
        sparse_hits = self.bm25.search(
            query,
            top_k=cand_k,
            filter_domain=domain_filter,
            filter_source=source_filter,
            user_ctx=user_ctx,
        )
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60)
        if user_ctx is not None:
            fused = [d for d in fused if can_access_document(user_ctx, d)]

        # 单篇父文档多样性配额控制：防止一篇 3 万字长文档霸占全部 Top-K
        selected: List[Dict[str, Any]] = []
        parent_counts: Dict[str, int] = {}
        for d in fused:
            pid = d.get("parent_doc_id") or d.get("uuid")
            if max_chunks_per_doc > 0 and parent_counts.get(pid, 0) >= max_chunks_per_doc:
                continue
            parent_counts[pid] = parent_counts.get(pid, 0) + 1
            selected.append(d)
            if len(selected) >= top_k:
                break
        return selected
