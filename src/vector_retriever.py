# -*- coding: utf-8 -*-
# src/vector_retriever.py
import os
import json
import pickle
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from src.access_control import UserContext, annotate_corpus_acl, can_access_document

try:
    import dashscope
    dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
except ImportError:
    dashscope = None


class VectorRetriever:
    """
    自适应双引擎语义向量检索器：
    - 优先加载 Chunk 级语义向量索引 (chunk_embeddings.npy + chunks_meta.json + chunk_lsa_model.pkl)
    - 支持毫秒级查询编码与运行时热增量追加 (add_docs) / 删除 (remove_by_parent_id)
    - 严格执行检索前 RBAC/ACL 硬过滤 (Pre-Retrieval Hard Filtering)
    """

    def __init__(
        self,
        cache_file: str = "data_cache/chunk_embeddings.npy",
        meta_file: str = "data_cache/chunks_meta.json",
        lsa_model_file: str = "data_cache/chunk_lsa_model.pkl",
        fallback_cache_file: str = "data_cache/doc_embeddings.npy",
        fallback_meta_file: str = "data_cache/doc_meta.json",
    ):
        self.cache_file = cache_file
        self.meta_file = meta_file
        self.lsa_model_file = lsa_model_file
        self.fallback_cache_file = fallback_cache_file
        self.fallback_meta_file = fallback_meta_file

        self.embeddings: Optional[np.ndarray] = None
        self.doc_metas: List[Dict[str, Any]] = []
        self.vectorizer = None
        self.svd = None
        self.dim: int = 0
        self.using_chunk_lsa: bool = False

    def load(self) -> bool:
        # 优先加载 Chunk 级语义索引
        if os.path.exists(self.cache_file) and os.path.exists(self.meta_file):
            self.embeddings = np.load(self.cache_file).astype(np.float32)
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.doc_metas = annotate_corpus_acl(json.load(f))
            self.dim = int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0
            if os.path.exists(self.lsa_model_file):
                try:
                    with open(self.lsa_model_file, "rb") as mf:
                        bundle = pickle.load(mf)
                    self.vectorizer = bundle.get("vectorizer")
                    self.svd = bundle.get("svd")
                    self.using_chunk_lsa = (self.vectorizer is not None and self.svd is not None)
                except Exception:
                    self.using_chunk_lsa = False
            return True

        # 向下兼容旧版整篇文档索引
        if os.path.exists(self.fallback_cache_file) and os.path.exists(self.fallback_meta_file):
            self.embeddings = np.load(self.fallback_cache_file).astype(np.float32)
            with open(self.fallback_meta_file, "r", encoding="utf-8") as f:
                self.doc_metas = annotate_corpus_acl(json.load(f))
            self.dim = int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0
            self.using_chunk_lsa = False
            return True

        return False

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """批量将文本编码为 L2 归一化语义向量（支持在线查询与新上传文档热入库）"""
        if not texts:
            return np.zeros((0, self.dim or 256), dtype=np.float32)

        if self.using_chunk_lsa and self.vectorizer is not None and self.svd is not None:
            mat = self.vectorizer.transform(texts)
            dense = self.svd.transform(mat).astype(np.float32)
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms = np.where(norms > 1e-12, norms, 1.0)
            return (dense / norms).astype(np.float32)

        # 兼容外部 DashScope（仅当未启用本地 LSA 且维度匹配时）
        vecs = [self.get_embedding(t) for t in texts]
        return np.array(vecs, dtype=np.float32)

    def get_embedding(self, text: str, user_ctx: Optional[UserContext] = None) -> List[float]:
        # 1. 本地 LSA 语义模型（与 chunk_embeddings.npy 维度严格一致，毫秒级零成本）
        if self.using_chunk_lsa and self.vectorizer is not None and self.svd is not None:
            vec = self.encode_texts([text])[0]
            return vec.tolist()

        # 2. 旧版 DashScope 模式（仅在配置了有效 Key 且回退旧索引时使用）
        if dashscope and os.environ.get("DASHSCOPE_API_KEY"):
            try:
                resp = dashscope.TextEmbedding.call(
                    model="text-embedding-v3",
                    input=[text[:2048]],
                )
                if resp.status_code == 200:
                    return resp.output["embeddings"][0]["embedding"]
            except Exception:
                pass

        # 3. 兜底投影（仅使用已授权文档）
        tokens = set(re.findall(r"[a-zA-Z0-9_\-]+|[一-龥]", (text or "").lower()))
        matched_vecs = []
        allowed_indices = []
        for i, doc in enumerate(self.doc_metas):
            if user_ctx is not None and not can_access_document(user_ctx, doc):
                continue
            allowed_indices.append(i)
            doc_toks = set(
                re.findall(r"[a-zA-Z0-9_\-]+|[一-龥]", (doc.get("title", "") + " " + doc.get("domain", "")).lower())
            )
            overlap = len(tokens & doc_toks)
            if overlap > 0:
                matched_vecs.append((self.embeddings[i], overlap))

        if matched_vecs:
            vec = np.zeros(self.embeddings.shape[1], dtype=np.float32)
            total_w = 0.0
            for v, w in matched_vecs[:10]:
                vec += v * w
                total_w += w
            if total_w > 0:
                vec /= total_w
            return vec.tolist()

        if allowed_indices:
            return np.mean(self.embeddings[allowed_indices[:20]], axis=0).tolist()
        return np.zeros(self.dim or 256, dtype=np.float32).tolist()

    def add_docs(self, new_chunks: List[Dict[str, Any]]) -> int:
        """实时热增量追加新 Chunk 的语义向量与元数据，无需重启服务"""
        if not new_chunks:
            return 0
        if self.embeddings is None:
            self.load()
        texts = [c.get("enriched_text") or c.get("text", "") for c in new_chunks]
        new_vecs = self.encode_texts(texts)
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = new_vecs
            self.dim = int(new_vecs.shape[1])
        else:
            self.embeddings = np.vstack([self.embeddings, new_vecs]).astype(np.float32)
        self.doc_metas.extend(new_chunks)
        return len(new_chunks)

    def remove_by_parent_id(self, parent_doc_id: str) -> int:
        """按父文档 ID 同步移除向量与元数据"""
        if not parent_doc_id or not self.doc_metas:
            return 0
        keep_indices = [
            i for i, d in enumerate(self.doc_metas)
            if (d.get("parent_doc_id") or d.get("uuid")) != parent_doc_id
        ]
        removed = len(self.doc_metas) - len(keep_indices)
        if removed > 0:
            self.doc_metas = [self.doc_metas[i] for i in keep_indices]
            if self.embeddings is not None:
                self.embeddings = self.embeddings[keep_indices]
        return removed

    def update_acl_by_parent_id(
        self,
        parent_doc_id: str,
        security_level: int,
        source_type: Optional[str] = None,
    ) -> int:
        updated = 0
        for d in self.doc_metas:
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
        if self.embeddings is None:
            self.load()
        if self.embeddings is None or len(self.doc_metas) == 0:
            return []

        src_filter_l = filter_source.lower().strip() if filter_source else None

        # 检索前构建授权候选索引掩码（Pre-Retrieval Hard Filtering）
        valid_indices = []
        for idx, doc in enumerate(self.doc_metas):
            if user_ctx is not None and not can_access_document(user_ctx, doc):
                continue
            if src_filter_l and (doc.get("source_type") or "").lower().strip() != src_filter_l:
                continue
            if filter_domain and doc.get("domain") != filter_domain:
                continue
            valid_indices.append(idx)

        if not valid_indices:
            return []

        q_vec = np.array(self.get_embedding(query, user_ctx=user_ctx), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        sub_embeddings = self.embeddings[valid_indices]
        sub_scores = np.dot(sub_embeddings, q_vec)
        order = np.argsort(sub_scores)[::-1][:top_k]

        results = []
        for local_pos in order:
            orig_idx = valid_indices[int(local_pos)]
            results.append((self.doc_metas[orig_idx], float(sub_scores[int(local_pos)])))
        return results
