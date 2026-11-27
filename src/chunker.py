# -*- coding: utf-8 -*-
# src/chunker.py
import re
from typing import List, Dict, Any, Tuple

from src.access_control import classify_document_acl


_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_SPEAKER_LINE_RE = re.compile(
    r"^(?:\[[^\]]{2,40}\]\s*(?:\[[^\]]{1,40}\])?\s*:|"
    r"[A-Za-z0-9_.\-\u4e00-\u9fa5]{1,32}\s*[\(\（][^\)\）]{1,30}[\)\）]\s*[:：]|"
    r"(?:From|To|Subject|Date|Speaker)\s*[:：]|"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s*[:：])"
)
_ENTITY_ID_RE = re.compile(r"\b(?:ENG|JIRA|LIN|INC|SEC)-\d+\b|\bPR[\s\-_#]*\d+\b", re.IGNORECASE)


def _split_atomic_blocks(text: str) -> List[Tuple[str, str]]:
    """
    将文本拆分为结构原子单元，保护 Markdown 代码块（```...```）与连续表格不被拦腰切碎。
    返回 [(block_type, block_text), ...]，其中 block_type ∈ {'code', 'table', 'para'}。
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: List[Tuple[str, str]] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 1. Fenced code block
        if stripped.startswith("```"):
            buf = [line]
            i += 1
            while i < n:
                buf.append(lines[i])
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                i += 1
            blocks.append(("code", "\n".join(buf).strip()))
            continue

        # 2. Markdown table block (lines containing | with at least 2 pipes or separator row)
        if stripped.startswith("|") or (stripped.count("|") >= 2 and i + 1 < n and "|" in lines[i + 1]):
            buf = [line]
            i += 1
            while i < n:
                s = lines[i].strip()
                if s and ("|" in s):
                    buf.append(lines[i])
                    i += 1
                else:
                    break
            blocks.append(("table", "\n".join(buf).strip()))
            continue

        # 3. Blank lines
        if not stripped:
            i += 1
            continue

        # 4. Regular paragraph (accumulate until blank line, code fence, table, or heading)
        buf = [line]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if not nxt or nxt.startswith("```") or nxt.startswith("|") or _HEADING_RE.match(nxt):
                break
            buf.append(lines[i])
            i += 1
        blocks.append(("para", "\n".join(buf).strip()))

    return [(bt, tx) for bt, tx in blocks if tx]


def _split_long_piece(piece: str, max_len: int, overlap: int) -> List[str]:
    """
    四级兜底切分：句子 -> 换行 -> 固定字符窗口，确保即使无标点超长日志/代码块也不会产生畸形超大块。
    """
    if len(piece) <= max_len:
        return [piece] if piece.strip() else []

    # Level 1: 按中英文句末标点切分
    raw_sents = re.split(r"([。！？\.\?\!]\s+|\n)", piece)
    units: List[str] = []
    buf = ""
    for tok in raw_sents:
        if not tok:
            continue
        buf += tok
        if re.match(r"^(?:[。！？\.\?\!]\s+|\n)$", tok):
            if buf.strip():
                units.append(buf.strip())
            buf = ""
    if buf.strip():
        units.append(buf.strip())

    # 若仍存在单句超过 max_len，则硬切字符窗口
    refined: List[str] = []
    step = max(100, max_len - min(overlap, max_len // 3))
    for u in units:
        if len(u) <= max_len:
            refined.append(u)
        else:
            pos = 0
            while pos < len(u):
                sub = u[pos : pos + max_len].strip()
                if sub:
                    refined.append(sub)
                if pos + max_len >= len(u):
                    break
                pos += step

    return _pack_units(refined, max_len=max_len, overlap=overlap, sep="\n")


def _pack_units(units: List[str], max_len: int, overlap: int, sep: str = "\n\n") -> List[str]:
    """将原子单元按 max_len 贪心打包，并从尾部回溯保留不超过 overlap 的完整重叠单元"""
    if not units:
        return []
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for u in units:
        u = u.strip()
        if not u:
            continue
        add_len = len(u) + (len(sep) if cur else 0)
        if cur and cur_len + add_len > max_len:
            chunks.append(sep.join(cur))
            # 尾部回溯构造 overlap
            ov: List[str] = []
            ov_len = 0
            for prev in reversed(cur):
                extra = len(prev) + (len(sep) if ov else 0)
                if ov_len + extra <= overlap:
                    ov.insert(0, prev)
                    ov_len += extra
                else:
                    break
            cur = ov
            cur_len = ov_len
        cur.append(u)
        cur_len += len(u) + (len(sep) if len(cur) > 1 else 0)

    if cur:
        chunks.append(sep.join(cur))
    return chunks


class SmartChunker:
    """
    企业级多源结构感知切片引擎：
    - 文档规范类 (confluence / google_drive / 上传文档): Markdown 标题层级切片 + 面包屑 section_path + 代码块/表格保护
    - 对话/转录类 (slack / gmail / fireflies): 按发言人/消息边界聚合切块，块间保留 1~2 条完整发言重叠
    - 工单/代码/商机类 (jira / linear / github / hubspot): 保留主实体头部上下文 + 评论流切片，全块继承单号标识
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 120):
        self.chunk_size = max(200, int(chunk_size))
        self.overlap = max(0, min(int(overlap), self.chunk_size // 2))

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将单篇文档切分为带完整 ACL 与父子关系的 Chunk 列表"""
        text = (doc.get("text") or "").strip()
        if not text:
            return []

        src = (doc.get("source_type") or "confluence").lower().strip()
        title = (doc.get("title") or "未命名文档").strip()
        parent_id = doc.get("uuid") or doc.get("parent_doc_id") or "doc_unknown"

        # 确保继承父文档的 ACL 安全标签
        sec_level = doc.get("security_level")
        acl_dept = doc.get("acl_dept")
        if sec_level is None or acl_dept is None:
            acl = classify_document_acl(doc)
            sec_level = acl["security_level"]
            acl_dept = acl["acl_dept"]

        if src in ("slack", "gmail", "fireflies"):
            raw_chunks = self._chunk_conversational(text, src=src)
        elif src in ("jira", "linear", "github", "hubspot"):
            raw_chunks = self._chunk_structured_ticket(text, title=title, src=src)
        else:
            raw_chunks = self._chunk_markdown_document(text, title=title)

        if not raw_chunks:
            raw_chunks = [("正文", text[: self.chunk_size])]

        total = len(raw_chunks)
        results: List[Dict[str, Any]] = []
        for idx, (section_path, chunk_body) in enumerate(raw_chunks):
            cid = f"{parent_id}#c{idx:03d}"
            sec_label = (section_path or "正文").strip()[:100]
            enriched = f"[来源: {src.upper()} | 文档: {title} | 章节: {sec_label}]\n{chunk_body}"
            results.append(
                {
                    "chunk_id": cid,
                    "uuid": cid,
                    "parent_doc_id": parent_id,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "section_path": sec_label,
                    "source_type": src,
                    "domain": doc.get("domain", "企业综合信息"),
                    "title": title,
                    "author": doc.get("author", ""),
                    "created_at": doc.get("created_at", ""),
                    "file_path": doc.get("file_path", ""),
                    "security_level": int(sec_level),
                    "acl_dept": acl_dept,
                    "is_uploaded": bool(doc.get("is_uploaded", False)),
                    "text": chunk_body,
                    "enriched_text": enriched,
                }
            )
        return results

    def chunk_corpus(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks: List[Dict[str, Any]] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    # -----------------------------------------------------------------
    # 策略 1：文档规范类（标题层级 + 代码块/表格原子保护）
    # -----------------------------------------------------------------
    def _chunk_markdown_document(self, text: str, title: str) -> List[Tuple[str, str]]:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        sections: List[Tuple[str, List[str]]] = []
        heading_stack: List[Tuple[int, str]] = []
        cur_path = "正文概览"
        cur_lines: List[str] = []
        in_code = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                cur_lines.append(line)
                continue
            if not in_code:
                m = _HEADING_RE.match(stripped)
                if m:
                    if cur_lines and "\n".join(cur_lines).strip():
                        sections.append((cur_path, cur_lines))
                        cur_lines = []
                    lvl = len(m.group(1))
                    h_text = m.group(2).strip()
                    while heading_stack and heading_stack[-1][0] >= lvl:
                        heading_stack.pop()
                    heading_stack.append((lvl, h_text))
                    cur_path = " > ".join(h for _, h in heading_stack)
                    continue
            cur_lines.append(line)

        if cur_lines and "\n".join(cur_lines).strip():
            sections.append((cur_path, cur_lines))

        if not sections:
            sections = [("正文", lines)]

        out: List[Tuple[str, str]] = []
        for sec_path, sec_lines in sections:
            sec_text = "\n".join(sec_lines).strip()
            if not sec_text:
                continue
            if len(sec_text) <= self.chunk_size:
                out.append((sec_path, sec_text))
                continue

            atomic_blocks = _split_atomic_blocks(sec_text)
            expanded_units: List[str] = []
            for btype, btext in atomic_blocks:
                # 表格与代码块允许轻微放宽 1.25x 尽量保持完整；超出再降级按行切
                limit = int(self.chunk_size * 1.25) if btype in ("code", "table") else self.chunk_size
                if len(btext) <= limit:
                    expanded_units.append(btext)
                else:
                    expanded_units.extend(_split_long_piece(btext, max_len=self.chunk_size, overlap=self.overlap))

            packed = _pack_units(expanded_units, max_len=self.chunk_size, overlap=self.overlap, sep="\n\n")
            for p in packed:
                out.append((sec_path, p))
        return out

    # -----------------------------------------------------------------
    # 策略 2：对话与会议转录类（按发言人/消息轮次边界切分）
    # -----------------------------------------------------------------
    def _chunk_conversational(self, text: str, src: str) -> List[Tuple[str, str]]:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        turns: List[str] = []
        cur: List[str] = []

        for line in lines:
            s = line.strip()
            if not s:
                if cur:
                    turns.append("\n".join(cur).strip())
                    cur = []
                continue
            if _SPEAKER_LINE_RE.match(s) and cur:
                turns.append("\n".join(cur).strip())
                cur = [line]
            else:
                cur.append(line)
        if cur:
            turns.append("\n".join(cur).strip())

        turns = [t for t in turns if t]
        max_len = int(self.chunk_size * 1.12)  # 对话/会议允许 ~900 字以保留完整轮次
        expanded: List[str] = []
        for t in turns:
            if len(t) <= max_len:
                expanded.append(t)
            else:
                expanded.extend(_split_long_piece(t, max_len=max_len, overlap=self.overlap))

        packed = _pack_units(expanded, max_len=max_len, overlap=max(self.overlap, 160), sep="\n\n")
        label_prefix = "会议纪要片段" if src == "fireflies" else ("邮件往来片段" if src == "gmail" else "频道讨论片段")
        return [(f"{label_prefix} #{i + 1}", p) for i, p in enumerate(packed)]

    # -----------------------------------------------------------------
    # 策略 3：工单 / PR / CRM 结构化类（保留主实体号 + 评论流切分）
    # -----------------------------------------------------------------
    def _chunk_structured_ticket(self, text: str, title: str, src: str) -> List[Tuple[str, str]]:
        entity_matches = _ENTITY_ID_RE.findall(f"{title} {text[:300]}")
        entity_tag = entity_matches[0].upper() if entity_matches else title[:32]
        sec_chunks = self._chunk_markdown_document(text, title=title)
        out: List[Tuple[str, str]] = []
        for idx, (sec_path, body) in enumerate(sec_chunks):
            tag = f"{entity_tag} · 概要" if idx == 0 else f"{entity_tag} · {sec_path}"
            if idx > 0 and entity_tag.lower() not in body.lower():
                body = f"[{entity_tag}] {body}"
            out.append((tag, body))
        return out
