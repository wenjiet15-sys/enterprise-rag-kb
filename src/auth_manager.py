# -*- coding: utf-8 -*-
# src/auth_manager.py
import hmac
import time
import uuid
import hashlib
import secrets
from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.access_control import UserContext, ROLE_POLICIES


PBKDF2_ITERATIONS = 120_000


def hash_password(password: str) -> str:
    """使用 PBKDF2-HMAC-SHA256 + 16 字节随机盐生成安全密码摘要"""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """常量时间比对密码哈希，防止时序攻击"""
    if not password or not stored_hash:
        return False
    try:
        algo, iter_str, salt, expected_hex = stored_hash.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iter_str),
        )
        return hmac.compare_digest(dk.hex(), expected_hex)
    except Exception:
        return False


# 预置 4 个演示测试账号（首次初始化表为空时自动写入，密码以加盐哈希落库）
DEFAULT_SEED_USERS = [
    ("u_admin", "admin", "系统管理员", "admin123", "信息安全与平台部", "admin", 4),
    ("u_dev_zhang", "dev_zhang", "张工 (研发)", "dev123", "敏捷研发工程部", "engineer", 2),
    ("u_sales_li", "sales_li", "李经理 (商务)", "sales123", "商业化与客户成功部", "sales", 2),
    ("u_intern_wang", "intern_wang", "小王 (实习生)", "intern123", "研发实习组", "intern", 1),
]


class AuthManager:
    """
    基于现有 MySQL 连接池的企业认证与安全审计管理器：
    - users 表：员工账号、角色、部门、密级、加盐密码哈希
    - audit_logs 表：登录、RAG 检索、越权拦截 (DENY)、文档导出与权限变更审计
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self._init_tables()
        self._seed_default_users()

    def _init_tables(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id VARCHAR(64) PRIMARY KEY,
                        username VARCHAR(64) NOT NULL UNIQUE,
                        display_name VARCHAR(128) NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        department VARCHAR(128) NOT NULL DEFAULT '通用部门',
                        role VARCHAR(32) NOT NULL DEFAULT 'intern',
                        clearance_level INT NOT NULL DEFAULT 1,
                        is_active TINYINT(1) NOT NULL DEFAULT 1,
                        created_at DOUBLE NOT NULL,
                        last_login_at DOUBLE NULL,
                        INDEX idx_users_role_active (role, is_active)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        user_id VARCHAR(64) NOT NULL,
                        username VARCHAR(64) NOT NULL DEFAULT '',
                        role VARCHAR(32) NOT NULL DEFAULT '',
                        action VARCHAR(64) NOT NULL,
                        query_text TEXT NULL,
                        allowed_docs_count INT NOT NULL DEFAULT 0,
                        blocked_docs_count INT NOT NULL DEFAULT 0,
                        status VARCHAR(16) NOT NULL DEFAULT 'ALLOW',
                        detail TEXT NULL,
                        created_at DOUBLE NOT NULL,
                        INDEX idx_audit_created (created_at),
                        INDEX idx_audit_user_status (user_id, status, created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
            )

    def _seed_default_users(self) -> None:
        now = time.time()
        with self.engine.begin() as conn:
            cnt = int(conn.execute(text("SELECT COUNT(*) FROM users;")).scalar() or 0)
            if cnt > 0:
                return
            for uid, uname, dname, raw_pwd, dept, role, clearance in DEFAULT_SEED_USERS:
                conn.execute(
                    text(
                        """
                        INSERT INTO users (
                            user_id, username, display_name, password_hash,
                            department, role, clearance_level, is_active, created_at
                        ) VALUES (
                            :uid, :uname, :dname, :phash,
                            :dept, :role, :cl, 1, :now
                        );
                        """
                    ),
                    {
                        "uid": uid,
                        "uname": uname,
                        "dname": dname,
                        "phash": hash_password(raw_pwd),
                        "dept": dept,
                        "role": role,
                        "cl": clearance,
                        "now": now,
                    },
                )

    @staticmethod
    def _row_to_user_context(row: Dict[str, Any]) -> UserContext:
        return UserContext(
            user_id=row["user_id"],
            username=row["username"],
            display_name=row["display_name"],
            department=row["department"],
            role=row["role"],
            clearance_level=int(row["clearance_level"]),
            is_active=bool(row["is_active"]),
        )

    # =================================================================
    # 认证与用户管理
    # =================================================================

    def authenticate(self, username: str, password: str) -> Tuple[Optional[UserContext], str]:
        """校验账号密码，成功返回 (UserContext, '') 并记录审计日志"""
        uname = (username or "").strip()
        if not uname or not password:
            return None, "请输入用户名和密码。"

        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM users WHERE username = :uname;"),
                {"uname": uname},
            ).mappings().first()

        if not row:
            self.log_audit(
                user=None,
                action="LOGIN_FAIL",
                status="DENY",
                detail=f"未知账号尝试登录: {uname}",
                fallback_username=uname,
            )
            return None, "用户名或密码错误。"

        if not bool(row["is_active"]):
            self.log_audit(
                user=self._row_to_user_context(row),
                action="LOGIN_FAIL",
                status="DENY",
                detail="账号已被管理员禁用",
            )
            return None, "该账号已被禁用，请联系系统管理员。"

        if not verify_password(password, row["password_hash"]):
            self.log_audit(
                user=self._row_to_user_context(row),
                action="LOGIN_FAIL",
                status="DENY",
                detail="密码校验失败",
            )
            return None, "用户名或密码错误。"

        user_ctx = self._row_to_user_context(row)
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE users SET last_login_at = :ts WHERE user_id = :uid;"),
                {"ts": time.time(), "uid": user_ctx.user_id},
            )
        self.log_audit(
            user=user_ctx,
            action="LOGIN_OK",
            status="ALLOW",
            detail=f"登录成功（角色：{user_ctx.role_name}，密级：{user_ctx.clearance_label}）",
        )
        return user_ctx, ""

    def get_user_by_id(self, user_id: str) -> Optional[UserContext]:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM users WHERE user_id = :uid;"),
                {"uid": user_id},
            ).mappings().first()
        return self._row_to_user_context(row) if row else None

    def list_users(self) -> List[Dict[str, Any]]:
        """返回脱敏后的员工账号列表（永不返回 password_hash）"""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT user_id, username, display_name, department,
                           role, clearance_level, is_active, created_at, last_login_at
                    FROM users
                    ORDER BY created_at ASC;
                    """
                )
            ).mappings().all()
        return [dict(r) for r in rows]

    def create_user(
        self,
        username: str,
        display_name: str,
        password: str,
        department: str,
        role: str,
        clearance_level: Optional[int] = None,
        operator: Optional[UserContext] = None,
    ) -> Tuple[bool, str]:
        uname = (username or "").strip()
        dname = (display_name or "").strip() or uname
        if len(uname) < 2:
            return False, "用户名长度不能少于 2 个字符。"
        if len(password or "") < 6:
            return False, "初始密码长度不能少于 6 位。"
        if role not in ROLE_POLICIES:
            return False, f"非法角色：{role}"

        cl = int(clearance_level or ROLE_POLICIES[role]["default_clearance"])
        cl = max(1, min(4, cl))
        uid = f"u_{uuid.uuid4().hex[:10]}"

        try:
            with self.engine.begin() as conn:
                exists = conn.execute(
                    text("SELECT 1 FROM users WHERE username = :uname;"),
                    {"uname": uname},
                ).scalar()
                if exists:
                    return False, f"用户名 `{uname}` 已存在。"
                conn.execute(
                    text(
                        """
                        INSERT INTO users (
                            user_id, username, display_name, password_hash,
                            department, role, clearance_level, is_active, created_at
                        ) VALUES (
                            :uid, :uname, :dname, :phash,
                            :dept, :role, :cl, 1, :now
                        );
                        """
                    ),
                    {
                        "uid": uid,
                        "uname": uname,
                        "dname": dname,
                        "phash": hash_password(password),
                        "dept": (department or "通用部门").strip(),
                        "role": role,
                        "cl": cl,
                        "now": time.time(),
                    },
                )
            self.log_audit(
                user=operator,
                action="ADMIN_CREATE_USER",
                status="ALLOW",
                detail=f"新增账号 {uname}（角色: {role}, 密级: L{cl}）",
            )
            return True, f"已创建账号 `{uname}`。"
        except Exception as e:
            return False, f"创建失败：{e}"

    def update_user_access(
        self,
        target_user_id: str,
        role: str,
        clearance_level: int,
        department: str,
        is_active: bool,
        new_password: Optional[str] = None,
        operator: Optional[UserContext] = None,
    ) -> Tuple[bool, str]:
        if role not in ROLE_POLICIES:
            return False, f"非法角色：{role}"
        cl = max(1, min(4, int(clearance_level)))

        fields = [
            "role = :role",
            "clearance_level = :cl",
            "department = :dept",
            "is_active = :active",
        ]
        params: Dict[str, Any] = {
            "uid": target_user_id,
            "role": role,
            "cl": cl,
            "dept": (department or "通用部门").strip(),
            "active": 1 if is_active else 0,
        }
        if new_password and new_password.strip():
            if len(new_password.strip()) < 6:
                return False, "重置密码长度不能少于 6 位。"
            fields.append("password_hash = :phash")
            params["phash"] = hash_password(new_password.strip())

        with self.engine.begin() as conn:
            res = conn.execute(
                text(f"UPDATE users SET {', '.join(fields)} WHERE user_id = :uid;"),
                params,
            )
            if res.rowcount == 0:
                return False, "目标账号不存在。"

        self.log_audit(
            user=operator,
            action="ADMIN_UPDATE_USER",
            status="ALLOW",
            detail=f"更新账号 {target_user_id} 权限（角色: {role}, 密级: L{cl}, 启用: {is_active}）",
        )
        return True, "账号权限已更新。"

    # =================================================================
    # 安全审计日志
    # =================================================================

    def log_audit(
        self,
        user: Optional[UserContext],
        action: str,
        status: str = "ALLOW",
        query_text: str = "",
        allowed_docs_count: int = 0,
        blocked_docs_count: int = 0,
        detail: str = "",
        fallback_username: str = "",
    ) -> None:
        uid = user.user_id if user else "anonymous"
        uname = user.username if user else (fallback_username or "anonymous")
        role = user.role if user else "guest"
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO audit_logs (
                            user_id, username, role, action, query_text,
                            allowed_docs_count, blocked_docs_count, status, detail, created_at
                        ) VALUES (
                            :uid, :uname, :role, :action, :qtext,
                            :allowed_cnt, :blocked_cnt, :status, :detail, :now
                        );
                        """
                    ),
                    {
                        "uid": uid,
                        "uname": uname,
                        "role": role,
                        "action": action[:64],
                        "qtext": (query_text or "")[:1000],
                        "allowed_cnt": int(allowed_docs_count or 0),
                        "blocked_cnt": int(blocked_docs_count or 0),
                        "status": status[:16],
                        "detail": (detail or "")[:1000],
                        "now": time.time(),
                    },
                )
        except Exception:
            pass

    def list_audit_logs(
        self,
        limit: int = 100,
        username: Optional[str] = None,
        status: Optional[str] = None,
        action: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conds = ["1=1"]
        params: Dict[str, Any] = {"lim": int(limit)}
        if username and username.strip():
            conds.append("username LIKE :uname")
            params["uname"] = f"%{username.strip()}%"
        if status and status != "ALL":
            conds.append("status = :status")
            params["status"] = status
        if action and action != "ALL":
            conds.append("action = :action")
            params["action"] = action

        sql = (
            "SELECT * FROM audit_logs "
            f"WHERE {' AND '.join(conds)} "
            "ORDER BY created_at DESC LIMIT :lim;"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]
