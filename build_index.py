import os
import json
import time
from src.data_transformer import transform_document
from src.vector_retriever import VectorRetriever

def build_corpus():
    index_path = "data_cache/corpus_1000_index.json"
    sources_root = "EnterpriseRAG-Bench/generated_data/sources"
    meta_output = "data_cache/doc_meta.json"
    
    with open(index_path, 'r', encoding='utf-8') as f:
        corpus_index = json.load(f)

    print(f"Total documents to process: {len(corpus_index)}")
    processed_docs = []

    for uuid, rel_path in corpus_index.items():
        full_p = os.path.join(sources_root, rel_path)
        source_type = rel_path.split('/')[0]
        try:
            with open(full_p, 'r', encoding='utf-8') as f:
                doc_json = json.load(f)
            transformed = transform_document(doc_json, source_type, uuid)
            # Add relative path
            transformed['file_path'] = rel_path
            processed_docs.append(transformed)
        except Exception as e:
            print(f"Error reading {rel_path}: {e}")

    print(f"Successfully transformed {len(processed_docs)} documents.")
    with open(meta_output, 'w', encoding='utf-8') as f:
        json.dump(processed_docs, f, ensure_ascii=False)
    print(f"Saved transformed docs metadata to {meta_output}.")

    # Build vector embeddings
    vr = VectorRetriever(cache_file="data_cache/doc_embeddings.npy", meta_file=meta_output)
    vr.build_and_save(processed_docs, text_key='text', force=True)
    print("Vector embedding build completed!")

if __name__ == '__main__':
    build_corpus()
