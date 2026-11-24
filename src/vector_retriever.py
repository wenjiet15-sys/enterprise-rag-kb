import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import re

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
                self.doc_metas = json.load(f)
            return True
        return False

    def get_embedding(self, text: str) -> List[float]:
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
        
        # Robust fallback: pseudo-semantic projection from precomputed doc embeddings
        # Matches query tokens against title/domain of docs to construct a synthesized vector
        tokens = set(re.findall(r'[a-zA-Z0-9_\-]+|[一-龥]', text.lower()))
        matched_vecs = []
        for i, doc in enumerate(self.doc_metas):
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
            
        return np.mean(self.embeddings[:20], axis=0).tolist()

    def search(self, query: str, top_k: int = 10, filter_domain: str = None) -> List[Tuple[Dict[str, Any], float]]:
        if self.embeddings is None:
            self.load()

        q_vec = np.array(self.get_embedding(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = np.dot(self.embeddings, q_vec)

        top_indices = np.argsort(scores)[::-1]
        results = []
        for idx in top_indices:
            doc = self.doc_metas[idx]
            if filter_domain and doc.get('domain') != filter_domain:
                continue
            results.append((doc, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results
