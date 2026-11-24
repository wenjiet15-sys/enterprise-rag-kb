import re
import json

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize multiple whitespaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def transform_document(doc_json: dict, source_type: str, doc_uuid: str) -> dict:
    """
    Transforms heterogeneous source raw JSON into unified RAG representation.
    Extracts title, domain, author/owner, timestamp, clean text, and structured metadata.
    """
    title_field = doc_json.get('title_field_name', 'title')
    title = str(doc_json.get(title_field, '') or '')
    
    # Domain categorization (maps to enterprise governance, R&D, operations, sales/customer, HSE)
    domain_mapping = {
        'confluence': '内部规章与技术标准',
        'google_drive': '方案与规划文档',
        'jira': '研发缺陷与工程管理',
        'linear': '敏捷任务与需求管理',
        'github': '代码变更与架构规范',
        'slack': '日常协作与非结构化讨论',
        'gmail': '内外通信与业务通知',
        'hubspot': '客户关系与商业信息',
        'fireflies': '会议纪要与转录档案'
    }
    domain = domain_mapping.get(source_type, '企业综合信息')

    # Content extraction
    content_fields = doc_json.get('content_field_names', [])
    content_parts = []
    
    if not content_fields:
        # Fallback fields
        for cand in ['content', 'description', 'messages', 'notes', 'summary', 'transcript']:
            if cand in doc_json:
                content_fields.append(cand)

    for cf in content_fields:
        val = doc_json.get(cf)
        if not val:
            continue
        if isinstance(val, list):
            # Check if elements are dicts (like messages)
            item_strs = []
            for item in val:
                if isinstance(item, dict):
                    # Format dict nicely (e.g. sender: message)
                    sender = item.get('sender', item.get('author', item.get('user', '')))
                    body = item.get('body', item.get('text', item.get('message', '')))
                    ts = item.get('timestamp', item.get('date', ''))
                    prefix = f"[{sender}]" if sender else ""
                    if ts:
                        prefix = f"[{ts}] {prefix}"
                    item_strs.append(f"{prefix}: {body}".strip(": "))
                else:
                    item_strs.append(str(item))
            content_parts.append("\n".join(item_strs))
        elif isinstance(val, dict):
            content_parts.append(json.dumps(val, ensure_ascii=False))
        else:
            content_parts.append(str(val))

    full_text = clean_text("\n\n".join(content_parts))
    if not title:
        # Fallback title from channel or first line
        if 'channel' in doc_json:
            title = f"Slack Channel #{doc_json['channel']}"
        elif 'key' in doc_json:
            title = f"Issue {doc_json['key']}"
        else:
            title = full_text[:40].replace('\n', ' ') or f"Document {doc_uuid[:8]}"

    # Metadata extraction for filtering & provenance
    created_at = doc_json.get('created_at') or doc_json.get('recorded_at') or doc_json.get('first_email_at') or doc_json.get('first_message_ts') or ''
    author = doc_json.get('author') or doc_json.get('owner') or doc_json.get('creator') or doc_json.get('mailbox_owner') or doc_json.get('redwood_owner') or ''
    
    return {
        'uuid': doc_uuid,
        'source_type': source_type,
        'domain': domain,
        'title': title,
        'author': str(author),
        'created_at': str(created_at),
        'text': full_text,
        'raw_metadata': {k: str(v)[:200] for k, v in doc_json.items() if k not in content_fields and not isinstance(v, (dict, list))}
    }

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list:
    """
    Hierarchical / Recursive chunker preserving paragraphs and sentences.
    """
    if len(text) <= chunk_size:
        return [text] if text else []
    
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_len = 0
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(p) > chunk_size:
            # Split paragraph by sentences
            sentences = re.split(r'([。！？\.\?\!]\s*)', p)
            # Recombine sentence tokens
            reconstructed = []
            for i in range(0, len(sentences)-1, 2):
                reconstructed.append(sentences[i] + sentences[i+1])
            if len(sentences) % 2 == 1:
                reconstructed.append(sentences[-1])
                
            for sent in reconstructed:
                sent = sent.strip()
                if not sent:
                    continue
                if current_len + len(sent) > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Overlap
                    overlap_tokens = []
                    overlap_len = 0
                    for token in reversed(current_chunk):
                        if overlap_len + len(token) <= overlap:
                            overlap_tokens.insert(0, token)
                            overlap_len += len(token)
                        else:
                            break
                    current_chunk = overlap_tokens
                    current_len = overlap_len
                current_chunk.append(sent)
                current_len += len(sent)
        else:
            if current_len + len(p) > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_tokens = []
                overlap_len = 0
                for token in reversed(current_chunk):
                    if overlap_len + len(token) <= overlap:
                        overlap_tokens.insert(0, token)
                        overlap_len += len(token)
                    else:
                        break
                current_chunk = overlap_tokens
                current_len = overlap_len
            current_chunk.append(p)
            current_len += len(p)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
