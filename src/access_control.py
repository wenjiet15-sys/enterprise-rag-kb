# -*- coding: utf-8 -*-
# src/access_control.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set


CLEARANCE_LABELS: Dict[int, str] = {
    1: "L1 公开",
    2: "L2 内部",
    3: "L3 机密",
    4: "L4 绝密",
}

SOURCE_TYPE_LABELS: Dict[str, str] = {
    "confluence": "Confluence (内部规章与标准)",
    "slack": "Slack (即时技术协同)",
    "jira": "Jira (研发缺陷与工单)",
    "google_drive": "Google Drive (方案与规划文档)",
    "gmail": "Gmail (内外通信与邮件)",
    "linear": "Linear (敏捷任务与需求)",
    "hubspot": "HubSpot (客户关系与商业报价)",
    "github": "GitHub (代码变更与架构PR)",
    "fireflies": "Fireflies (会议纪要与转录)",
}

ALL_SOURCE_TYPES: Set[str] = set(SOURCE_TYPE_LABELS.keys())

ROLE_POLICIES: Dict[str, Dict[str, Any]] = {
    "admin": {
        "role_name": "系统管理员",
        "default_clearance": 4,
        "allowed_sources": set(ALL_SOURCE_TYPES),
        "allowed_pages": [
            "💬 主对话",
            "🧠 记忆管理",
            "🛡️ 权限与审计",
            "📑 文档自动化生成",
            "📊 评测基准与算力消融",
            "🗂️ 多源知识库画像",
        ],
        "can_use_level3": True,
        "can_export_doc": True,
        "description": "拥有全部 9 大数据源 L1-L4 访问权限、账号角色管理与全员安全审计权限。",
    },
    "engineer": {
        "role_name": "研发工程师",
        "default_clearance": 2,
        "allowed_sources": {"confluence", "github", "jira", "linear", "slack", "google_drive"},
        "allowed_pages": [
            "💬 主对话",
            "🧠 记忆管理",
            "📑 文档自动化生成",
            "📊 评测基准与算力消融",
            "🗂️ 多源知识库画像",
        ],
        "can_use_level3": True,
        "can_export_doc": True,
        "description": "可查阅研发工程与运维规范数据源；禁止访问 HubSpot 客户商业报价、Gmail 及高密级会议纪要。",
    },
    "sales": {
        "role_name": "商务/销售",
        "default_clearance": 2,
        "allowed_sources": {"hubspot", "gmail", "confluence", "google_drive"},
        "allowed_pages": [
            "💬 主对话",
            "🧠 记忆管理",
            "📑 文档自动化生成",
            "🗂️ 多源知识库画像",
        ],
        "can_use_level3": False,
        "can_export_doc": True,
        "description": "可查阅 HubSpot 客户治理、邮件与公开方案文档；禁止访问 GitHub 代码、Jira 缺陷及研发任务。",
    },
    "intern": {
        "role_name": "实习生/访客",
        "default_clearance": 1,
        "allowed_sources": {"confluence"},
        "allowed_pages": [
            "💬 主对话",
            "🧠 记忆管理",
        ],
        "can_use_level3": False,
        "can_export_doc": False,
        "description": "仅可查阅 L1 公开级 Confluence 规章文档；禁止访问研发/商务数据源、重型会诊与文档导出。",
    },
}


@dataclass
class UserContext:
    """已认证员工的安全上下文（随请求透传至检索层、记忆层与审计层）"""
    user_id: str
    username: str
    display_name: str
    department: str
    role: str
    clearance_level: int = 1
    is_active: bool = True
    allowed_sources: Set[str] = field(default_factory=set)

    def __post_init__(self):
        policy = ROLE_POLICIES.get(self.role, ROLE_POLICIES["intern"])
        if not self.allowed_sources:
            self.allowed_sources = set(policy["allowed_sources"])
        self.clearance_level = max(1, min(4, int(self.clearance_level or policy["default_clearance"])))

    @property
    def role_name(self) -> str:
        return ROLE_POLICIES.get(self.role, ROLE_POLICIES["intern"])["role_name"]

    @property
    def clearance_label(self) -> str:
        return CLEARANCE_LABELS.get(self.clearance_level, f"L{self.clearance_level}")

    @property
    def is_admin(self) -> bool:
        return self.role == "admin" and self.is_active

    @property
    def allowed_pages(self) -> List[str]:
        return list(ROLE_POLICIES.get(self.role, ROLE_POLICIES["intern"])["allowed_pages"])

    @property
    def can_use_level3(self) -> bool:
        return bool(self.is_active and ROLE_POLICIES.get(self.role, ROLE_POLICIES["intern"])["can_use_level3"])

    @property
    def can_export_doc(self) -> bool:
        return bool(self.is_active and ROLE_POLICIES.get(self.role, ROLE_POLICIES["intern"])["can_export_doc"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "department": self.department,
            "role": self.role,
            "clearance_level": self.clearance_level,
            "is_active": self.is_active,
            "allowed_sources": sorted(self.allowed_sources),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserContext":
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            display_name=data.get("display_name", data["username"]),
            department=data.get("department", "通用部门"),
            role=data.get("role", "intern"),
            clearance_level=int(data.get("clearance_level", 1)),
            is_active=bool(data.get("is_active", True)),
            allowed_sources=set(data.get("allowed_sources") or []),
        )


# =====================================================================
# 文档级 ACL 安全分类器（基于 9 大真实数据源 + 敏感主题规则）
# =====================================================================

_L3_KEYWORDS = (
    "salary", "payroll", "compensation", "layoff", "acquisition", "m&a",
    "board meeting", "executive", "confidential", "secret", "private key",
    "nda", "pricing", "contract", "budget", "financial", "soc2", "audit",
    "vulnerability", "cve-", "zero-day", "credential", "薪资", "报价", "合同", "绝密", "机密",
)

_L2_CONFLUENCE_KEYWORDS = (
    "incident", "postmortem", "security", "failover", "cluster", "production",
    "migration", "dr ", "disaster", "runbook", "outage", "容灾", "故障", "生产",
)


def classify_document_acl(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    为每篇文档计算确定性的安全标签：
    - security_level: 1(公开) / 2(内部) / 3(机密) / 4(绝密)
    - acl_dept: 归属业务线（公共 / 研发 / 商务 / 管理）
    """
    src = (doc.get("source_type") or "confluence").lower().strip()
    title = (doc.get("title") or "").lower()
    snippet = (doc.get("text") or "")[:600].lower()
    blob = f"{title} {snippet}"

    has_l3 = any(k in blob for k in _L3_KEYWORDS)

    if src == "confluence":
        dept = "公共"
        if any(k in blob for k in ("salary", "payroll", "board meeting", "acquisition", "confidential")):
            level = 3
        elif any(k in blob for k in _L2_CONFLUENCE_KEYWORDS):
            level = 2
        else:
            level = 1
    elif src in ("github", "jira", "linear"):
        dept = "研发"
        level = 3 if any(k in blob for k in ("vulnerability", "cve-", "credential", "private key", "secret")) else 2
    elif src == "hubspot":
        dept = "商务"
        level = 3 if any(k in blob for k in ("pricing", "contract", "nda", "budget", "confidential")) else 2
    elif src in ("fireflies", "gmail"):
        dept = "管理" if src == "fireflies" else "商务"
        level = 3 if has_l3 else 2
    elif src in ("slack", "google_drive"):
        dept = "研发" if src == "slack" else "公共"
        if has_l3:
            level = 3
        elif any(k in title for k in ("general", "handbook", "onboarding", "faq")):
            level = 1
        else:
            level = 2
    else:
        dept = "公共"
        level = 2

    return {
        "security_level": level,
        "acl_dept": dept,
        "source_type": src,
    }


def annotate_corpus_acl(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """启动初始化阶段为全量语料原地补充 security_level 与 acl_dept 字段"""
    for doc in docs:
        if "security_level" not in doc or "acl_dept" not in doc:
            acl = classify_document_acl(doc)
            doc["security_level"] = acl["security_level"]
            doc["acl_dept"] = acl["acl_dept"]
    return docs


def can_access_document(user: Optional[UserContext], doc: Dict[str, Any]) -> bool:
    """
    检索前硬过滤核心判定函数（Fail-Closed：未认证或禁用账号一律拒绝）：
    - 管理员放行全部文档；
    - 其他角色同时校验：source_type 在白名单内 AND doc.security_level <= user.clearance_level。
    """
    if user is None or not user.is_active:
        return False
    if user.is_admin:
        return True

    src = (doc.get("source_type") or "").lower().strip()
    if src not in user.allowed_sources:
        return False

    doc_level = doc.get("security_level")
    if doc_level is None:
        doc_level = classify_document_acl(doc)["security_level"]
    return int(doc_level) <= int(user.clearance_level)


def can_access_engineering_ticket(user: Optional[UserContext]) -> bool:
    """判断用户是否有权访问 Jira/GitHub 工单直查与研发缺陷上下文"""
    if user is None or not user.is_active:
        return False
    if user.is_admin:
        return True
    return ("jira" in user.allowed_sources or "github" in user.allowed_sources) and user.clearance_level >= 2
