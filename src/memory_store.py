# -*- coding: utf-8 -*-
# src/memory_store.py
import os
import json
import time
from urllib.parse import quote_plus
from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.intent_router import SessionState


def _load_dotenv_file(env_path: str = ".env") -> None:
    """轻量读取项目根目录 .env 配置（已存在的系统环境变量优先级更高）"""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                val = v.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


def _trim_turn_extras(turn_extra: Optional[Dict[str, Any]], max_doc_chars: int = 400) -> Optional[Dict[str, Any]]:
    """落库前精简 extras_json 中检索文档正文，控制单行体积同时保留完整溯源信息"""
    if not turn_extra:
        return turn_extra
    trimmed = dict(turn_extra)
    raw_docs = trimmed.get("docs")
    if isinstance(raw_docs, list):
        compact_docs = []
        for d in raw_docs:
            if not isinstance(d, dict):
                continue
            doc_text = (d.get("text") or "").strip()
            if len(doc_text) > max_doc_chars:
                doc_text = doc_text[:max_doc_chars] + "..."
            compact_docs.append(
                {
                    "uuid": d.get("uuid"),
                    "source_type": d.get("source_type", "DOC"),
                    "title": d.get("title", "未命名"),
                    "domain": d.get("domain", "通用"),
                    "author": d.get("author", "未知"),
                    "security_level": d.get("security_level", 1),
                    "section_path": d.get("section_path", ""),
                    "context_window_range": d.get("context_window_range", ""),
                    "chunk_index": d.get("chunk_index"),
                    "total_chunks": d.get("total_chunks"),
                    "score": d.get("score", d.get("rrf_score", 0.0)),
                    "text": doc_text,
                }
            )
        trimmed["docs"] = compact_docs
    return trimmed


class MemoryStore:
    """
    基于 MySQL + SQLAlchemy 连接池 (PyMySQL) 的生产级记忆持久化层：
    - sessions: 多用户/多会话元信息、滚动摘要、状态机与算力遥测
    - messages: 增量追加写入的逐轮对话消息与精简溯源元数据 (extras_json)
    - long_term_memories: 跨会话长期记忆表（用户画像、项目背景、事实结论、自定义指令）
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        env_path: str = ".env",
    ):
        _load_dotenv_file(env_path)
        self.host = host or os.environ.get("MYSQL_HOST", "localhost")
        self.port = int(port or os.environ.get("MYSQL_PORT", "3306"))
        self.user = user or os.environ.get("MYSQL_USER", "root")
        self.password = password if password is not None else os.environ.get("MYSQL_PASSWORD", "root")
        self.database = database or os.environ.get("MYSQL_DB", "enterprise_rag_memory")

        self._ensure_database()
        self.engine: Engine = self._create_engine()
        self._init_tables()

    def _build_url(self, include_db: bool = True) -> str:
        u = quote_plus(self.user)
        p = quote_plus(self.password)
        db_part = f"/{quote_plus(self.database)}" if include_db else ""
        return f"mysql+pymysql://{u}:{p}@{self.host}:{self.port}{db_part}?charset=utf8mb4"

    def _ensure_database(self) -> None:
        """若目标数据库不存在，自动创建 utf8mb4 数据库"""
        server_engine = create_engine(self._build_url(include_db=False), pool_pre_ping=True)
        safe_db = self.database.replace("`", "")
        try:
            with server_engine.begin() as conn:
                conn.execute(
                    text(
                        f"CREATE DATABASE IF NOT EXISTS `{safe_db}` "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
                    )
                )
        finally:
            server_engine.dispose()

    def _create_engine(self) -> Engine:
        return create_engine(
            self._build_url(include_db=True),
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            future=True,
        )

    def _init_tables(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id VARCHAR(64) PRIMARY KEY,
                        user_id VARCHAR(64) NOT NULL DEFAULT 'default_user',
                        title VARCHAR(128) NOT NULL DEFAULT '新对话',
                        rolling_summary TEXT NOT NULL,
                        summarized_msg_count INT NOT NULL DEFAULT 0,
                        active_entity VARCHAR(128) NULL,
                        active_domain VARCHAR(128) NULL,
                        active_stage VARCHAR(128) NOT NULL DEFAULT '病症初诊与咨询',
                        traffic_counts_json VARCHAR(255) NOT NULL DEFAULT '{"1": 0, "2": 0, "3": 0}',
                        total_requests INT NOT NULL DEFAULT 0,
                        created_at DOUBLE NOT NULL,
                        last_updated DOUBLE NOT NULL,
                        INDEX idx_sessions_user_updated (user_id, last_updated)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(64) NOT NULL,
                        seq INT NOT NULL,
                        role VARCHAR(32) NOT NULL,
                        content MEDIUMTEXT NOT NULL,
                        timestamp DOUBLE NOT NULL,
                        level INT NOT NULL DEFAULT 2,
                        stage VARCHAR(128) NULL,
                        rewritten_query TEXT NULL,
                        original_query TEXT NULL,
                        model VARCHAR(128) NULL,
                        total_time DOUBLE NULL,
                        docs_count INT NOT NULL DEFAULT 0,
                        history_used TINYINT(1) NOT NULL DEFAULT 0,
                        extras_json MEDIUMTEXT NULL,
                        UNIQUE KEY uq_messages_session_seq (session_id, seq),
                        INDEX idx_messages_session_role (session_id, role),
                        CONSTRAINT fk_messages_session
                            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        user_id VARCHAR(64) NOT NULL DEFAULT 'default_user',
                        category VARCHAR(64) NOT NULL DEFAULT 'custom',
                        content TEXT NOT NULL,
                        keywords VARCHAR(512) NOT NULL DEFAULT '',
                        source_session_id VARCHAR(64) NULL,
                        importance INT NOT NULL DEFAULT 3,
                        hit_count INT NOT NULL DEFAULT 0,
                        created_at DOUBLE NOT NULL,
                        updated_at DOUBLE NOT NULL,
                        last_accessed_at DOUBLE NOT NULL,
                        INDEX idx_ltm_user_cat_imp (user_id, category, importance, updated_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
            )

    def health_check(self) -> Dict[str, Any]:
        """检查 MySQL 连通性与当前表记录数"""
        with self.engine.connect() as conn:
            ver = conn.execute(text("SELECT VERSION();")).scalar()
            s_cnt = conn.execute(text("SELECT COUNT(*) FROM sessions;")).scalar()
            m_cnt = conn.execute(text("SELECT COUNT(*) FROM messages;")).scalar()
            ltm_cnt = conn.execute(text("SELECT COUNT(*) FROM long_term_memories;")).scalar()
        return {
            "status": "ok",
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "mysql_version": ver,
            "sessions_count": int(s_cnt or 0),
            "messages_count": int(m_cnt or 0),
            "memories_count": int(ltm_cnt or 0),
        }

    # =================================================================
    # 会话与消息持久化 (MySQL 事务 + 增量追加写入)
    # =================================================================

    def save_session(
        self,
        session: SessionState,
        turn_extras: Optional[Dict[int, Dict[str, Any]]] = None,
        user_id: str = "default_user",
        force_rewrite_messages: bool = False,
    ) -> None:
        """
        事务保存会话元信息并增量追加新消息：
        - 正常多轮对话时仅 INSERT 新增的 seq 消息，不删除重写旧记录；
        - 若会话被清空重置（len(history) < db_count）或 force_rewrite_messages=True，则自动重建消息列表。
        """
        extras_map = turn_extras or {}
        now = time.time()
        session.last_updated = now
        traffic_json = json.dumps(
            {str(k): int(v) for k, v in session.traffic_counts.items()},
            ensure_ascii=False,
        )

        with self.engine.begin() as conn:
            # 防水平越权：若该 session_id 已存在且归属其他用户，拒绝覆盖写入
            owner_row = conn.execute(
                text("SELECT user_id FROM sessions WHERE session_id = :sid;"),
                {"sid": session.session_id},
            ).scalar()
            if owner_row is not None and owner_row != user_id:
                raise PermissionError(f"无权修改属于其他用户的会话: {session.session_id}")

            conn.execute(
                text(
                    """
                    INSERT INTO sessions (
                        session_id, user_id, title, rolling_summary, summarized_msg_count,
                        active_entity, active_domain, active_stage,
                        traffic_counts_json, total_requests, created_at, last_updated
                    ) VALUES (
                        :session_id, :user_id, :title, :rolling_summary, :summarized_msg_count,
                        :active_entity, :active_domain, :active_stage,
                        :traffic_counts_json, :total_requests, :created_at, :last_updated
                    )
                    ON DUPLICATE KEY UPDATE
                        title = VALUES(title),
                        rolling_summary = VALUES(rolling_summary),
                        summarized_msg_count = VALUES(summarized_msg_count),
                        active_entity = VALUES(active_entity),
                        active_domain = VALUES(active_domain),
                        active_stage = VALUES(active_stage),
                        traffic_counts_json = VALUES(traffic_counts_json),
                        total_requests = VALUES(total_requests),
                        last_updated = VALUES(last_updated);
                    """
                ),
                {
                    "session_id": session.session_id,
                    "user_id": user_id,
                    "title": (session.title or "新对话")[:128],
                    "rolling_summary": session.rolling_summary or "",
                    "summarized_msg_count": int(session.summarized_msg_count or 0),
                    "active_entity": session.active_entity,
                    "active_domain": session.active_domain,
                    "active_stage": (session.active_stage or "病症初诊与咨询")[:128],
                    "traffic_counts_json": traffic_json,
                    "total_requests": int(session.total_requests or 0),
                    "created_at": float(session.created_at or now),
                    "last_updated": float(session.last_updated),
                },
            )

            existing_count = int(
                conn.execute(
                    text("SELECT COUNT(*) FROM messages WHERE session_id = :sid;"),
                    {"sid": session.session_id},
                ).scalar()
                or 0
            )

            # 若发生清空/重置或显式要求重写，则清理旧消息并从 0 写入；否则增量追加
            start_seq = existing_count
            if force_rewrite_messages or len(session.history) < existing_count:
                conn.execute(
                    text("DELETE FROM messages WHERE session_id = :sid;"),
                    {"sid": session.session_id},
                )
                start_seq = 0

            # 计算每一条 user 消息对应的 1-based 轮次编号
            turn_num_by_seq: Dict[int, int] = {}
            t_idx = 0
            for idx, m in enumerate(session.history):
                if m.get("role") == "user":
                    t_idx += 1
                    turn_num_by_seq[idx] = t_idx

            rows_to_insert = []
            for seq in range(start_seq, len(session.history)):
                msg = session.history[seq]
                role = msg.get("role", "user")
                extra_json = None
                if role == "user":
                    t_num = turn_num_by_seq.get(seq)
                    if t_num is not None and t_num in extras_map:
                        compact = _trim_turn_extras(extras_map[t_num])
                        extra_json = json.dumps(compact, ensure_ascii=False, default=str)
                rows_to_insert.append(
                    {
                        "session_id": session.session_id,
                        "seq": seq,
                        "role": role,
                        "content": msg.get("content", ""),
                        "timestamp": float(msg.get("timestamp", now)),
                        "level": int(msg.get("level", 2) or 2),
                        "stage": msg.get("stage"),
                        "rewritten_query": msg.get("rewritten_query"),
                        "original_query": msg.get("original_query"),
                        "model": msg.get("model"),
                        "total_time": msg.get("total_time"),
                        "docs_count": int(msg.get("docs_count", 0) or 0),
                        "history_used": 1 if msg.get("history_used") else 0,
                        "extras_json": extra_json,
                    }
                )

            if rows_to_insert:
                conn.execute(
                    text(
                        """
                        INSERT INTO messages (
                            session_id, seq, role, content, timestamp, level, stage,
                            rewritten_query, original_query, model, total_time,
                            docs_count, history_used, extras_json
                        ) VALUES (
                            :session_id, :seq, :role, :content, :timestamp, :level, :stage,
                            :rewritten_query, :original_query, :model, :total_time,
                            :docs_count, :history_used, :extras_json
                        )
                        ON DUPLICATE KEY UPDATE
                            content = VALUES(content),
                            extras_json = VALUES(extras_json);
                        """
                    ),
                    rows_to_insert,
                )

    def load_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Tuple[SessionState, Dict[int, Dict[str, Any]]]]:
        """按 session_id 加载完整 SessionState 与每轮 turn_extras（传入 user_id 时强制校验归属，防水平越权）"""
        with self.engine.connect() as conn:
            if user_id is not None:
                s_row = conn.execute(
                    text("SELECT * FROM sessions WHERE session_id = :sid AND user_id = :uid;"),
                    {"sid": session_id, "uid": user_id},
                ).mappings().first()
            else:
                s_row = conn.execute(
                    text("SELECT * FROM sessions WHERE session_id = :sid;"),
                    {"sid": session_id},
                ).mappings().first()
            if not s_row:
                return None

            m_rows = conn.execute(
                text("SELECT * FROM messages WHERE session_id = :sid ORDER BY seq ASC;"),
                {"sid": session_id},
            ).mappings().all()

        history: List[Dict[str, Any]] = []
        turn_extras: Dict[int, Dict[str, Any]] = {}
        turn_num = 0

        for r in m_rows:
            role = r["role"]
            history.append(
                {
                    "role": role,
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "level": r["level"],
                    "stage": r["stage"],
                    "rewritten_query": r["rewritten_query"],
                    "original_query": r["original_query"],
                    "model": r["model"],
                    "total_time": r["total_time"],
                    "docs_count": r["docs_count"],
                    "history_used": bool(r["history_used"]),
                }
            )
            if role == "user":
                turn_num += 1
                if r["extras_json"]:
                    try:
                        turn_extras[turn_num] = json.loads(r["extras_json"])
                    except Exception:
                        turn_extras[turn_num] = {}

        try:
            tc_raw = json.loads(s_row["traffic_counts_json"] or "{}")
            traffic_counts = {int(k): int(v) for k, v in tc_raw.items()}
        except Exception:
            traffic_counts = {1: 0, 2: 0, 3: 0}

        session_dict = {
            "session_id": s_row["session_id"],
            "title": s_row["title"],
            "history": history,
            "rolling_summary": s_row["rolling_summary"],
            "summarized_msg_count": s_row["summarized_msg_count"],
            "active_entity": s_row["active_entity"],
            "active_domain": s_row["active_domain"],
            "active_stage": s_row["active_stage"],
            "created_at": s_row["created_at"],
            "last_updated": s_row["last_updated"],
            "traffic_counts": traffic_counts,
            "total_requests": s_row["total_requests"],
        }
        return SessionState.from_dict(session_dict), turn_extras

    def list_sessions(
        self,
        user_id: str = "default_user",
        limit: int = 30,
        keyword: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """按最近更新时间倒序列出历史会话摘要（可选关键词过滤标题/摘要）"""
        kw = (keyword or "").strip()
        with self.engine.connect() as conn:
            if kw:
                like_kw = f"%{kw}%"
                rows = conn.execute(
                    text(
                        """
                        SELECT
                            s.session_id,
                            s.title,
                            s.rolling_summary,
                            s.active_entity,
                            s.active_domain,
                            s.created_at,
                            s.last_updated,
                            SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS turns_count,
                            COALESCE(MAX(m.docs_count), 0) AS max_docs_count
                        FROM sessions s
                        LEFT JOIN messages m ON s.session_id = m.session_id
                        WHERE s.user_id = :uid
                          AND (s.title LIKE :kw OR s.rolling_summary LIKE :kw OR s.active_entity LIKE :kw)
                        GROUP BY s.session_id
                        ORDER BY s.last_updated DESC
                        LIMIT :lim;
                        """
                    ),
                    {"uid": user_id, "kw": like_kw, "lim": int(limit)},
                ).mappings().all()
            else:
                rows = conn.execute(
                    text(
                        """
                        SELECT
                            s.session_id,
                            s.title,
                            s.rolling_summary,
                            s.active_entity,
                            s.active_domain,
                            s.created_at,
                            s.last_updated,
                            SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS turns_count,
                            COALESCE(MAX(m.docs_count), 0) AS max_docs_count
                        FROM sessions s
                        LEFT JOIN messages m ON s.session_id = m.session_id
                        WHERE s.user_id = :uid
                        GROUP BY s.session_id
                        ORDER BY s.last_updated DESC
                        LIMIT :lim;
                        """
                    ),
                    {"uid": user_id, "lim": int(limit)},
                ).mappings().all()

        results = []
        for r in rows:
            results.append(
                {
                    "session_id": r["session_id"],
                    "title": r["title"] or "新对话",
                    "turns_count": int(r["turns_count"] or 0),
                    "has_rag": int(r["max_docs_count"] or 0) > 0,
                    "has_summary": bool((r["rolling_summary"] or "").strip()),
                    "active_entity": r["active_entity"],
                    "active_domain": r["active_domain"],
                    "created_at": r["created_at"],
                    "last_updated": r["last_updated"],
                }
            )
        return results

    def rename_session(self, session_id: str, new_title: str, user_id: Optional[str] = None) -> bool:
        clean_title = (new_title or "").strip()[:120] or "未命名对话"
        with self.engine.begin() as conn:
            if user_id is not None:
                res = conn.execute(
                    text("UPDATE sessions SET title = :title, last_updated = :ts WHERE session_id = :sid AND user_id = :uid;"),
                    {"title": clean_title, "ts": time.time(), "sid": session_id, "uid": user_id},
                )
            else:
                res = conn.execute(
                    text("UPDATE sessions SET title = :title, last_updated = :ts WHERE session_id = :sid;"),
                    {"title": clean_title, "ts": time.time(), "sid": session_id},
                )
            return res.rowcount > 0

    def delete_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        with self.engine.begin() as conn:
            if user_id is not None:
                res = conn.execute(
                    text("DELETE FROM sessions WHERE session_id = :sid AND user_id = :uid;"),
                    {"sid": session_id, "uid": user_id},
                )
            else:
                res = conn.execute(
                    text("DELETE FROM sessions WHERE session_id = :sid;"),
                    {"sid": session_id},
                )
            return res.rowcount > 0

    # =================================================================
    # 跨会话长期记忆 CRUD（供步骤 3 长期记忆引擎与步骤 4 管理面板调用）
    # =================================================================

    def add_memory(
        self,
        content: str,
        category: str = "custom",
        keywords: str = "",
        source_session_id: Optional[str] = None,
        importance: int = 3,
        user_id: str = "default_user",
    ) -> int:
        now = time.time()
        clean_content = (content or "").strip()
        if not clean_content:
            return -1
        imp = max(1, min(5, int(importance or 3)))
        with self.engine.begin() as conn:
            res = conn.execute(
                text(
                    """
                    INSERT INTO long_term_memories (
                        user_id, category, content, keywords, source_session_id,
                        importance, hit_count, created_at, updated_at, last_accessed_at
                    ) VALUES (
                        :uid, :cat, :content, :kw, :sid,
                        :imp, 0, :now, :now, :now
                    );
                    """
                ),
                {
                    "uid": user_id,
                    "cat": category[:64],
                    "content": clean_content,
                    "kw": (keywords or "").strip()[:512],
                    "sid": source_session_id,
                    "imp": imp,
                    "now": now,
                },
            )
            return int(res.lastrowid)

    def update_memory(
        self,
        memory_id: int,
        content: Optional[str] = None,
        category: Optional[str] = None,
        keywords: Optional[str] = None,
        importance: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        fields = []
        params: Dict[str, Any] = {"mid": int(memory_id), "ts": time.time()}
        if content is not None and content.strip():
            fields.append("content = :content")
            params["content"] = content.strip()
        if category is not None:
            fields.append("category = :category")
            params["category"] = category[:64]
        if keywords is not None:
            fields.append("keywords = :keywords")
            params["keywords"] = keywords.strip()[:512]
        if importance is not None:
            fields.append("importance = :importance")
            params["importance"] = max(1, min(5, int(importance)))
        if not fields:
            return False

        fields.append("updated_at = :ts")
        where_clause = "id = :mid"
        if user_id is not None:
            where_clause += " AND user_id = :uid"
            params["uid"] = user_id

        with self.engine.begin() as conn:
            res = conn.execute(
                text(f"UPDATE long_term_memories SET {', '.join(fields)} WHERE {where_clause};"),
                params,
            )
            return res.rowcount > 0

    def delete_memory(self, memory_id: int, user_id: Optional[str] = None) -> bool:
        with self.engine.begin() as conn:
            if user_id is not None:
                res = conn.execute(
                    text("DELETE FROM long_term_memories WHERE id = :mid AND user_id = :uid;"),
                    {"mid": int(memory_id), "uid": user_id},
                )
            else:
                res = conn.execute(
                    text("DELETE FROM long_term_memories WHERE id = :mid;"),
                    {"mid": int(memory_id)},
                )
            return res.rowcount > 0

    def clear_memories(self, category: Optional[str] = None, user_id: str = "default_user") -> int:
        with self.engine.begin() as conn:
            if category:
                res = conn.execute(
                    text("DELETE FROM long_term_memories WHERE user_id = :uid AND category = :cat;"),
                    {"uid": user_id, "cat": category},
                )
            else:
                res = conn.execute(
                    text("DELETE FROM long_term_memories WHERE user_id = :uid;"),
                    {"uid": user_id},
                )
            return int(res.rowcount)

    def list_memories(
        self,
        category: Optional[str] = None,
        limit: int = 200,
        user_id: str = "default_user",
    ) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            if category:
                rows = conn.execute(
                    text(
                        """
                        SELECT * FROM long_term_memories
                        WHERE user_id = :uid AND category = :cat
                        ORDER BY importance DESC, updated_at DESC
                        LIMIT :lim;
                        """
                    ),
                    {"uid": user_id, "cat": category, "lim": int(limit)},
                ).mappings().all()
            else:
                rows = conn.execute(
                    text(
                        """
                        SELECT * FROM long_term_memories
                        WHERE user_id = :uid
                        ORDER BY importance DESC, updated_at DESC
                        LIMIT :lim;
                        """
                    ),
                    {"uid": user_id, "lim": int(limit)},
                ).mappings().all()
        return [dict(r) for r in rows]

    def touch_memories(self, memory_ids: List[int], user_id: Optional[str] = None) -> None:
        if not memory_ids:
            return
        now = time.time()
        with self.engine.begin() as conn:
            if user_id is not None:
                conn.execute(
                    text(
                        "UPDATE long_term_memories "
                        "SET hit_count = hit_count + 1, last_accessed_at = :now "
                        "WHERE id = :mid AND user_id = :uid;"
                    ),
                    [{"now": now, "mid": int(mid), "uid": user_id} for mid in memory_ids],
                )
            else:
                conn.execute(
                    text(
                        "UPDATE long_term_memories "
                        "SET hit_count = hit_count + 1, last_accessed_at = :now "
                        "WHERE id = :mid;"
                    ),
                    [{"now": now, "mid": int(mid)} for mid in memory_ids],
                )
