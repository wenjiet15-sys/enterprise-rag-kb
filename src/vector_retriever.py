import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import re
from src.access_control import UserContext, annotate_corpus_acl, can_access_document

try:
    import dashscope
    dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')
except ImportError:
    dashscope = None

class VectorRetriever:
    def __init__(self, cache_file: str = 'data_cache/doc_embeddings.npy', meta_file: str = 'data_cache/doc_meta.json'):
        self.cache_file = cache_file
        self.meta_file = meta_file
        self.embeddings: np.ndarray = None
        self.doc_metas: List[Dict[str, Any]] = []

    def load(self):
        if os.path.exists(self.cache_file) and os.path.exists(self.meta_file):
            self.embeddings = np.load(self.cache_file)
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                self.doc_metas = annotate_corpus_acl(json.load(f))
            return True
        return False

    def get_embedding(self, text: str, user_ctx: Optional[UserContext] = None) -> List[float]:
        if dashscope:
            try:
                resp = dashscope.TextEmbedding.call(
                    model='text-embedding-v3',
                    input=[text[:2048]]
                )
                if resp.status_code == 200:
                    return resp.output['embeddings'][0]['embedding']
            except Exception:
                pass
        
        # Robust fallback: pseudo-semantic projection from authorized doc embeddings only
        tokens = set(re.findall(r'[a-zA-Z0-9_\-]+|[一-龥]', text.lower()))
        matched_vecs = []
        allowed_indices = []
        for i, doc in enumerate(self.doc_metas):
            if user_ctx is not None and not can_access_document(user_ctx, doc):
                continue
            allowed_indices.append(i)
            doc_toks = set(re.findall(r'[a-zA-Z0-9_\-]+|[一-龥]', (doc.get('title', '') + ' ' + doc.get('domain', '')).lower()))
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
        return np.zeros(self.embeddings.shape[1], dtype=np.float32).tolist()

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

        src_filter_l = filter_source.lower().strip() if filter_source else None

        # 检索前构建授权候选索引掩码（Pre-Retrieval Hard Filtering）
        valid_indices = []
        for idx, doc in enumerate(self.doc_metas):
            if user_ctx is not None and not can_access_document(user_ctx, doc):
                continue
            if src_filter_l and (doc.get('source_type') or '').lower().strip() != src_filter_l:
                continue
            if filter_domain and doc.get('domain') != filter_domain:
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
