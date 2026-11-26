# -*- coding: utf-8 -*-
# build_chunks_index.py
import os
import json
import time
import pickle
import statistics
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from src.access_control import annotate_corpus_acl
from src.chunker import SmartChunker


def build_chunks_and_vectors(
    doc_meta_path: str = "data_cache/doc_meta.json",
    chunks_meta_out: str = "data_cache/chunks_meta.json",
    chunk_emb_out: str = "data_cache/chunk_embeddings.npy",
    vectorizer_out: str = "data_cache/chunk_lsa_model.pkl",
    chunk_size: int = 800,
    overlap: int = 120,
    n_components: int = 256,
):
    t0 = time.time()
    with open(doc_meta_path, "r", encoding="utf-8") as f:
        docs = annotate_corpus_acl(json.load(f))

    chunker = SmartChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk_corpus(docs)
    t_chunk = time.time() - t0

    # 保存 Chunk 级元数据（含 ACL 标签与父子关系）
    with open(chunks_meta_out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    # 构建本地离线语义向量（词+字符级 TF-IDF -> TruncatedSVD 256维 L2 归一化，支持毫秒级增量推理）
    t_vec0 = time.time()
    texts = [c["enriched_text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=30000,
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_mat = vectorizer.fit_transform(texts)
    actual_dim = min(n_components, max(2, tfidf_mat.shape[1] - 1), max(2, tfidf_mat.shape[0] - 1))
    svd = TruncatedSVD(n_components=actual_dim, random_state=42)
    dense = svd.fit_transform(tfidf_mat)
    dense = normalize(dense, norm="l2", axis=1).astype(np.float32)

    np.save(chunk_emb_out, dense)
    with open(vectorizer_out, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "svd": svd, "dim": actual_dim}, f)
    t_vec = time.time() - t_vec0

    # 输出切片质量统计报告
    lens = [len(c["text"]) for c in chunks]
    lens_sorted = sorted(lens)
    p95 = lens_sorted[int(len(lens_sorted) * 0.95)]
    src_counts = Counter(c["source_type"] for c in chunks)
    lvl_counts = Counter(c["security_level"] for c in chunks)

    print("=" * 60)
    print(f"Smart Chunking Completed! {len(docs)} docs -> {len(chunks)} chunks")
    print(f"Chunking time: {t_chunk:.2f}s | Vector (TF-IDF+SVD-{actual_dim}) time: {t_vec:.2f}s")
    print(
        f"Chunk length stats: min={min(lens)}, median={int(statistics.median(lens))}, "
        f"avg={int(sum(lens)/len(lens))}, p95={p95}, max={max(lens)}"
    )
    print(f"Oversized (>1200 chars) chunks: {sum(1 for x in lens if x > 1200)} ({sum(1 for x in lens if x > 1200)/len(lens)*100:.2f}%)")
    print("Chunks per source:", dict(sorted(src_counts.items())))
    print("Chunks per security_level:", dict(sorted(lvl_counts.items())))
    print(f"Saved: {chunks_meta_out}, {chunk_emb_out} {dense.shape}, {vectorizer_out}")
    print("=" * 60)
    return chunks, dense


if __name__ == "__main__":
    build_chunks_and_vectors()
