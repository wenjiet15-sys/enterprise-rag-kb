# -*- coding: utf-8 -*-
# src/doc_ingestor.py
import io
import os
import re
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.access_control import UserContext, SOURCE_TYPE_LABELS
from src.chunker import SmartChunker


SOURCE_DOMAIN_MAP = {
    "confluence": "内部规章与技术标准",
    "google_drive": "方案与规划文档",
    "jira": "研发缺陷与工程管理",
    "linear": "敏捷任务与需求管理",
    "github": "代码变更与架构规范",
    "slack": "日常协作与非结构化讨论",
    "gmail": "内外通信与业务通知",
    "hubspot": "客户关系与商业信息",
    "fireflies": "会议纪要与转录档案",
}

SOURCE_DEPT_MAP = {
    "confluence": "公共",
    "google_drive": "公共",
    "jira": "研发",
    "linear": "研发",
    "github": "研发",
    "slack": "研发",
    "gmail": "商务",
    "hubspot": "商务",
    "fireflies": "管理",
}


def parse_document_bytes(file_name: str, file_bytes: bytes) -> Tuple[str, str]:
    """
    将上传的 .pdf / .docx / .md / .txt 解析为带结构标记（标题/表格）的纯文本。
    返回 (extracted_text, file_ext)。
    """
    ext = os.path.splitext(file_name or "")[1].lower().lstrip(".")
    if not file_bytes:
        return "", ext

    if ext in ("md", "txt", "markdown", "csv", "log", "json"):
        for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                return file_bytes.decode(enc).strip(), ext or "txt"
            except Exception:
                continue
        return file_bytes.decode("utf-8", errors="ignore").strip(), ext or "txt"

    if ext == "docx":
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        blocks: List[str] = []
        # 遍历段落并还原 Heading 层级为 Markdown 标题，供 SmartChunker 生成 section_path
        for p in doc.paragraphs:
            txt = (p.text or "").strip()
            if not txt:
                continue
            style_name = (p.style.name if p.style else "").lower()
            if "heading 1" in style_name or "标题 1" in style_name:
                blocks.append(f"# {txt}")
            elif "heading 2" in style_name or "标题 2" in style_name:
                blocks.append(f"## {txt}")
            elif "heading 3" in style_name or "标题 3" in style_name:
                blocks.append(f"### {txt}")
            elif "heading 4" in style_name or "标题 4" in style_name:
                blocks.append(f"#### {txt}")
            else:
                blocks.append(txt)

        # 保留 Word 表格为 Markdown 表格，避免切片丢失结构化数据
        for t_idx, table in enumerate(doc.tables, 1):
            rows_md = []
            for r_idx, row in enumerate(table.rows):
                cells = [re.sub(r"\s+", " ", (c.text or "").strip()).replace("|", "/") for c in row.cells]
                if not any(cells):
                    continue
                rows_md.append("| " + " | ".join(cells) + " |")
                if r_idx == 0:
                    rows_md.append("| " + " | ".join(["---"] * len(cells)) + " |")
            if rows_md:
                blocks.append(f"### 附表 {t_idx}\n" + "\n".join(rows_md))
        return "\n\n".join(blocks).strip(), "docx"

    if ext == "pdf":
        pages_out: List[str] = []
        # 优先使用 pdfplumber（对中文与表格更稳健），异常回退 pypdf
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for p_idx, page in enumerate(pdf.pages, 1):
                    p_txt = (page.extract_text() or "").strip()
                    if p_txt:
                        pages_out.append(f"## 第 {p_idx} 页\n{p_txt}")
        except Exception:
            pages_out = []

        if not pages_out:
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(file_bytes))
                for p_idx, page in enumerate(reader.pages, 1):
                    p_txt = (page.extract_text() or "").strip()
                    if p_txt:
                        pages_out.append(f"## 第 {p_idx} 页\n{p_txt}")
            except Exception as e:
                raise RuntimeError(f"PDF 解析失败: {e}")
        return "\n\n".join(pages_out).strip(), "pdf"

    raise ValueError(f"暂不支持的文件格式: .{ext}（支持 pdf / docx / md / txt）")


class DocumentIngestor:
    """
    在线文档上传、结构化切片、MySQL 持久化与索引热更新引擎：
    - uploaded_documents: 上传文档元信息与 ACL 密级
    - uploaded_chunks: 切片正文与带章节前缀的 enriched_text（重启自动恢复至内存索引）
    """

    def __init__(self, engine: Engine, chunk_size: int = 800, overlap: int = 120):
        self.engine = engine
        self.chunker = SmartChunker(chunk_size=chunk_size, overlap=overlap)
        self._init_tables()

    def _init_tables(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS uploaded_documents (
                        doc_id VARCHAR(64) PRIMARY KEY,
                        title VARCHAR(255) NOT NULL,
                        file_name VARCHAR(255) NOT NULL,
                        file_ext VARCHAR(16) NOT NULL,
                        source_type VARCHAR(64) NOT NULL,
                        domain VARCHAR(128) NOT NULL,
                        security_level INT NOT NULL DEFAULT 2,
                        acl_dept VARCHAR(64) NOT NULL DEFAULT '公共',
                        uploader_id VARCHAR(64) NOT NULL,
                        uploader_name VARCHAR(128) NOT NULL,
                        char_count INT NOT NULL DEFAULT 0,
                        chunk_count INT NOT NULL DEFAULT 0,
                        created_at DOUBLE NOT NULL,
                        updated_at DOUBLE NOT NULL,
                        INDEX idx_udocs_src_lvl (source_type, security_level),
                        INDEX idx_udocs_created (created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS uploaded_chunks (
                        chunk_id VARCHAR(96) PRIMARY KEY,
                        parent_doc_id VARCHAR(64) NOT NULL,
                        chunk_index INT NOT NULL,
                        total_chunks INT NOT NULL,
                        section_path VARCHAR(255) NOT NULL DEFAULT '正文',
                        text MEDIUMTEXT NOT NULL,
                        enriched_text MEDIUMTEXT NOT NULL,
                        INDEX idx_uchunks_parent (parent_doc_id, chunk_index),
                        CONSTRAINT fk_uchunks_doc
                            FOREIGN KEY (parent_doc_id) REFERENCES uploaded_documents(doc_id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
            )

    def ingest_uploaded_file(
        self,
        file_name: str,
        file_bytes: bytes,
        source_type: str,
        security_level: int,
        uploader: UserContext,
        pipeline,
        title: Optional[str] = None,
        auth_manager=None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """解析上传文件 -> 结构感知切片 -> MySQL 持久化 -> 热注入 BM25+Vector 索引 -> 记审计日志"""
        if not uploader or not uploader.is_active:
            return False, "未认证或账号已禁用，禁止上传文档。", {}

        src = (source_type or "confluence").lower().strip()
        if src not in SOURCE_TYPE_LABELS:
            return False, f"非法数据源：{src}", {}
        if not uploader.is_admin and src not in uploader.allowed_sources:
            return False, f"当前角色无权向受限数据源【{src}】上传文档。", {}

        sec_lvl = max(1, min(4, int(security_level or 2)))
        if not uploader.is_admin and sec_lvl > uploader.clearance_level:
            return False, f"当前密级上限为 L{uploader.clearance_level}，不能上传 L{sec_lvl} 密级文档。", {}

        try:
            raw_text, ext = parse_document_bytes(file_name, file_bytes)
        except Exception as e:
            return False, f"文档解析失败：{e}", {}

        if not raw_text or len(raw_text.strip()) < 5:
            return False, "未从文件中提取到有效文本内容（如果是扫描版图片 PDF，请先 OCR 转换）。", {}

        clean_title = (title or "").strip() or os.path.splitext(os.path.basename(file_name))[0]
        doc_id = f"up_{uuid.uuid4().hex[:12]}"
        domain = SOURCE_DOMAIN_MAP.get(src, "企业综合信息")
        acl_dept = SOURCE_DEPT_MAP.get(src, "公共")
        now = time.time()

        parent_doc = {
            "uuid": doc_id,
            "source_type": src,
            "domain": domain,
            "title": clean_title[:255],
            "author": uploader.display_name,
            "created_at": str(int(now)),
            "file_path": f"uploaded/{file_name}",
            "security_level": sec_lvl,
            "acl_dept": acl_dept,
            "is_uploaded": True,
            "text": raw_text,
        }
        chunks = self.chunker.chunk_document(parent_doc)
        if not chunks:
            return False, "切片结果为空，请检查文件内容。", {}

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO uploaded_documents (
                        doc_id, title, file_name, file_ext, source_type, domain,
                        security_level, acl_dept, uploader_id, uploader_name,
                        char_count, chunk_count, created_at, updated_at
                    ) VALUES (
                        :doc_id, :title, :file_name, :file_ext, :source_type, :domain,
                        :security_level, :acl_dept, :uploader_id, :uploader_name,
                        :char_count, :chunk_count, :now, :now
                    );
                    """
                ),
                {
                    "doc_id": doc_id,
                    "title": clean_title[:255],
                    "file_name": (file_name or "unknown")[:255],
                    "file_ext": ext[:16],
                    "source_type": src,
                    "domain": domain,
                    "security_level": sec_lvl,
                    "acl_dept": acl_dept,
                    "uploader_id": uploader.user_id,
                    "uploader_name": uploader.display_name[:128],
                    "char_count": len(raw_text),
                    "chunk_count": len(chunks),
                    "now": now,
                },
            )
            conn.execute(
                text(
                    """
                    INSERT INTO uploaded_chunks (
                        chunk_id, parent_doc_id, chunk_index, total_chunks,
                        section_path, text, enriched_text
                    ) VALUES (
                        :chunk_id, :parent_doc_id, :chunk_index, :total_chunks,
                        :section_path, :text, :enriched_text
                    );
                    """
                ),
                [
                    {
                        "chunk_id": ch["chunk_id"],
                        "parent_doc_id": doc_id,
                        "chunk_index": ch["chunk_index"],
                        "total_chunks": ch["total_chunks"],
                        "section_path": ch["section_path"][:255],
                        "text": ch["text"],
                        "enriched_text": ch["enriched_text"],
                    }
                    for ch in chunks
                ],
            )

        # 实时热更新双路索引（无需重启服务）
        if pipeline is not None:
            pipeline.add_chunks_to_index(chunks)

        if auth_manager is not None:
            auth_manager.log_audit(
                user=uploader,
                action="DOC_UPLOAD",
                status="ALLOW",
                query_text=clean_title,
                allowed_docs_count=len(chunks),
                detail=f"上传入库《{clean_title}》（.{ext}，数据源: {src}，密级: L{sec_lvl}，切片数: {len(chunks)}）",
            )

        return True, f"《{clean_title}》入库成功！已切分为 {len(chunks)} 个结构化片段并实时热更新至检索索引。", {
            "doc_id": doc_id,
            "title": clean_title,
            "chunk_count": len(chunks),
            "char_count": len(raw_text),
            "chunks_preview": chunks[:3],
        }

    def load_all_uploaded_chunks(self) -> List[Dict[str, Any]]:
        """启动恢复：从 MySQL 读取所有历史上传文档切片，自动挂载回内存索引"""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT
                        c.chunk_id, c.parent_doc_id, c.chunk_index, c.total_chunks,
                        c.section_path, c.text, c.enriched_text,
                        d.title, d.file_name, d.source_type, d.domain,
                        d.security_level, d.acl_dept, d.uploader_name, d.created_at
                    FROM uploaded_chunks c
                    JOIN uploaded_documents d ON c.parent_doc_id = d.doc_id
                    ORDER BY d.created_at ASC, c.chunk_index ASC;
                    """
                )
            ).mappings().all()

        chunks: List[Dict[str, Any]] = []
        for r in rows:
            chunks.append(
                {
                    "chunk_id": r["chunk_id"],
                    "uuid": r["chunk_id"],
                    "parent_doc_id": r["parent_doc_id"],
                    "chunk_index": int(r["chunk_index"]),
                    "total_chunks": int(r["total_chunks"]),
                    "section_path": r["section_path"],
                    "source_type": r["source_type"],
                    "domain": r["domain"],
                    "title": r["title"],
                    "author": r["uploader_name"],
                    "created_at": str(int(r["created_at"] or 0)),
                    "file_path": f"uploaded/{r['file_name']}",
                    "security_level": int(r["security_level"]),
                    "acl_dept": r["acl_dept"],
                    "is_uploaded": True,
                    "text": r["text"],
                    "enriched_text": r["enriched_text"],
                }
            )
        return chunks

    def list_uploaded_documents(self) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM uploaded_documents ORDER BY created_at DESC LIMIT 200;")
            ).mappings().all()
        return [dict(r) for r in rows]

    def update_document_acl(
        self,
        doc_id: str,
        security_level: int,
        source_type: str,
        pipeline,
        operator: Optional[UserContext] = None,
        auth_manager=None,
    ) -> Tuple[bool, str]:
        src = (source_type or "confluence").lower().strip()
        if src not in SOURCE_TYPE_LABELS:
            return False, f"非法数据源：{src}"
        sec_lvl = max(1, min(4, int(security_level)))
        domain = SOURCE_DOMAIN_MAP.get(src, "企业综合信息")
        acl_dept = SOURCE_DEPT_MAP.get(src, "公共")

        with self.engine.begin() as conn:
            res = conn.execute(
                text(
                    """
                    UPDATE uploaded_documents
                    SET security_level = :cl, source_type = :src, domain = :dom, acl_dept = :dept, updated_at = :ts
                    WHERE doc_id = :did;
                    """
                ),
                {"cl": sec_lvl, "src": src, "dom": domain, "dept": acl_dept, "ts": time.time(), "did": doc_id},
            )
            if res.rowcount == 0:
                return False, "文档不存在。"

        if pipeline is not None:
            pipeline.update_doc_acl_in_index(doc_id, security_level=sec_lvl, source_type=src)

        if auth_manager is not None:
            auth_manager.log_audit(
                user=operator,
                action="DOC_ACL_UPDATE",
                status="ALLOW",
                query_text=doc_id,
                detail=f"调整上传文档 {doc_id} 权限为数据源 {src} / 密级 L{sec_lvl}",
            )
        return True, f"已实时更新文档权限为 `{src}` · `L{sec_lvl}`。"

    def delete_document(
        self,
        doc_id: str,
        pipeline,
        operator: Optional[UserContext] = None,
        auth_manager=None,
    ) -> Tuple[bool, str]:
        with self.engine.begin() as conn:
            title_row = conn.execute(
                text("SELECT title FROM uploaded_documents WHERE doc_id = :did;"),
                {"did": doc_id},
            ).scalar()
            res = conn.execute(
                text("DELETE FROM uploaded_documents WHERE doc_id = :did;"),
                {"did": doc_id},
            )
            if res.rowcount == 0:
                return False, "文档不存在。"

        removed = pipeline.remove_doc_from_index(doc_id) if pipeline is not None else 0
        if auth_manager is not None:
            auth_manager.log_audit(
                user=operator,
                action="DOC_DELETE",
                status="ALLOW",
                query_text=str(title_row or doc_id),
                detail=f"删除上传文档《{title_row or doc_id}》，同步移除 {removed} 个索引切片",
            )
        return True, f"已删除文档《{title_row or doc_id}》并从实时索引移除 {removed} 个切片。"
