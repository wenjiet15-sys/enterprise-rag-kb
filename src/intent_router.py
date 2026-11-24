# -*- coding: utf-8 -*-
# src/intent_router.py
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from src.generator import MODEL_TIER_PROFILES

class SessionState:
    """
    会话病史档案管理器、业务状态追踪机与算力防破产监控器
    (Session History, Context State Tracker & Compute Cost Guard)
    
    模拟医院专科就诊流程与企业算力成本管控：
    - 记录患者（用户）历史多轮问答病史 (History Memory)
    - 追踪当前锁定的核心业务实体 (Active Entity, 如集群名、工单号、PR编号)
    - 追踪当前挂号就诊的专科科室 (Specialist Department)
    - 追踪当前业务所处的生命周期阶段 (Lifecycle Stage)
    - 【防破产流量探针】：实时统计 Level 1(0算力)、Level 2(轻量SLM)、Level 3(重型LLM) 调用频次与占比，
      严格监控重型推理算力是否控制在 5% 红线内。
    """
    def __init__(self, session_id: str = "default_session", title: str = "新对话"):
        self.session_id = session_id
        self.title: str = title
        self.history: List[Dict[str, Any]] = []
        self.rolling_summary: str = ""
        self.summarized_msg_count: int = 0
        self.active_entity: Optional[str] = None
        self.active_domain: Optional[str] = None
        self.active_stage: Optional[str] = "病症初诊与咨询"
        self.created_at = time.time()
        self.last_updated = time.time()

        # 算力梯次流量与成本遥测 (Traffic Telemetry & Budget Guard)
        self.traffic_counts = {1: 0, 2: 0, 3: 0}
        self.total_requests = 0

    def record_traffic(self, level: int):
        """记录一次对应层级的请求流量"""
        if level in self.traffic_counts:
            self.traffic_counts[level] += 1
            self.total_requests += 1
            self.last_updated = time.time()

    def get_traffic_metrics(self) -> Dict[str, Any]:
        """计算当前算力梯次流量分布与防破产告警指标"""
        total = self.total_requests
        l1 = self.traffic_counts.get(1, 0)
        l2 = self.traffic_counts.get(2, 0)
        l3 = self.traffic_counts.get(3, 0)

        l1_pct = round((l1 / total) * 100, 1) if total > 0 else 0.0
        l2_pct = round((l2 / total) * 100, 1) if total > 0 else 0.0
        l3_pct = round((l3 / total) * 100, 1) if total > 0 else 0.0

        # 防破产安全阈值判断：若请求数达到5次且重型模型占比超过 5%，触发警示
        is_over_budget = (total >= 5 and l3_pct > 5.0)

        status_text = "🟢 算力配置健康（重型专家号严控在红线内）"
        if total == 0:
            status_text = "⚪ 等待流量请求输入..."
        elif is_over_budget:
            status_text = f"🔴 警告：重型大模型占比 ({l3_pct}%) 超出 5% 预算红线！存在算力超支风险！"
        elif l3 > 0:
            status_text = f"🟡 专家会诊已按需唤醒 (当前占比 {l3_pct}%，处于安全阈值内)"

        return {
            "total_requests": total,
            "tier1_count": l1,
            "tier2_count": l2,
            "tier3_count": l3,
            "tier1_pct": l1_pct,
            "tier2_pct": l2_pct,
            "tier3_pct": l3_pct,
            "is_over_budget": is_over_budget,
            "status_text": status_text,
            "target_l1": "~60%",
            "target_l2": "~35%",
            "target_l3": "<5%"
        }

    def add_turn(self, role: str, content: str, level: int = 2, stage: Optional[str] = None, rewritten_query: Optional[str] = None, original_query: Optional[str] = None, model: Optional[str] = None, total_time: Optional[float] = None, docs_count: int = 0, history_used: bool = False):
        """记录一轮问答病史，并丰富改写和元数据"""
        turn_data = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "level": level,
            "stage": stage or self.active_stage,
            "rewritten_query": rewritten_query,
            "original_query": original_query,
            "model": model,
            "total_time": total_time,
            "docs_count": docs_count,
            "history_used": history_used
        }
        self.history.append(turn_data)
        if role == "user" and (not self.title or self.title == "新对话"):
            title_src = (original_query or content or "").strip()
            if title_src:
                self.title = title_src[:28] + ("..." if len(title_src) > 28 else "")
        self.last_updated = time.time()

    def get_dialogue_turns(self) -> List[Dict[str, Any]]:
        """聚合历史对话为用户问-助手答成对条目，方便前端完整回溯展示"""
        turns = []
        i = 0
        turn_num = 1
        while i < len(self.history):
            item = self.history[i]
            if item.get("role") == "user":
                user_item = item
                asst_item = self.history[i+1] if (i+1 < len(self.history) and self.history[i+1].get("role") == "assistant") else None
                turns.append({
                    "turn": turn_num,
                    "level": user_item.get("level", 2),
                    "stage": user_item.get("stage", "标准问答"),
                    "user_query": user_item.get("content", ""),
                    "original_query": user_item.get("original_query") or user_item.get("content", ""),
                    "rewritten_query": user_item.get("rewritten_query") or user_item.get("content", ""),
                    "history_used": user_item.get("history_used", False),
                    "model": user_item.get("model", ""),
                    "total_time": user_item.get("total_time"),
                    "docs_count": user_item.get("docs_count", 0),
                    "assistant_reply": asst_item.get("content", "") if asst_item else "",
                    "assistant_response": asst_item.get("content", "") if asst_item else "",
                    "timestamp": user_item.get("timestamp", time.time())
                })
                turn_num += 1
                i += 2 if asst_item else 1
            else:
                i += 1
        return turns

    def update_state(self, entity: Optional[str] = None, domain: Optional[str] = None, stage: Optional[str] = None):
        """更新状态机当前锁定的实体、科室与生命周期阶段"""
        if entity:
            self.active_entity = entity
        if domain:
            self.active_domain = domain
        if stage:
            self.active_stage = stage
        self.last_updated = time.time()

    def get_recent_messages(self, max_turns: int = 6, max_chars_per_msg: int = 1200) -> List[Dict[str, str]]:
        """
        返回最近 max_turns 轮（每轮含 user + assistant 共 2 条）的标准 OpenAI messages 数组，
        供大模型原生多轮对话使用，避免粗暴截断导致的上下文失真。
        """
        if not self.history or max_turns <= 0:
            return []
        recent = self.history[-(max_turns * 2):]
        formatted: List[Dict[str, str]] = []
        for h in recent:
            role = h.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (h.get("content") or "").strip()
            if not content:
                continue
            if len(content) > max_chars_per_msg:
                content = content[:max_chars_per_msg] + "…"
            formatted.append({"role": role, "content": content})
        return formatted

    def get_unsummarized_overflow(self, keep_recent_turns: int = 6, min_overflow_turns: int = 2) -> Tuple[List[Dict[str, Any]], int]:
        """
        检查是否存在超出最近窗口（keep_recent_turns 轮）且尚未压缩进 rolling_summary 的早期消息。
        仅当待压缩消息达到 min_overflow_turns 轮时才返回，以降低摘要调用频率。
        """
        cutoff = len(self.history) - keep_recent_turns * 2
        if cutoff <= self.summarized_msg_count:
            return [], self.summarized_msg_count
        overflow = self.history[self.summarized_msg_count:cutoff]
        if len(overflow) < min_overflow_turns * 2:
            return [], self.summarized_msg_count
        return overflow, cutoff

    def apply_summary(self, new_summary: str, new_summarized_count: int):
        """写入更新后的滚动摘要并推进已压缩游标"""
        if new_summary and new_summary.strip():
            self.rolling_summary = new_summary.strip()
        self.summarized_msg_count = max(self.summarized_msg_count, new_summarized_count)
        self.last_updated = time.time()

    def get_history_memory(self, max_turns: int = 4) -> str:
        """调阅既往就诊病史记录文本版（含滚动摘要 + 最近 max_turns 轮对话，保持向前兼容）"""
        parts: List[str] = []
        if self.rolling_summary:
            parts.append(f"【早期对话摘要】：\n{self.rolling_summary}")
        if self.history:
            recent = self.history[-(max_turns * 2):]
            lines = []
            for h in recent:
                role_tag = "用户提问" if h["role"] == "user" else "助手答复"
                lines.append(f"- 【{role_tag}】: {h['content'][:300]}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def get_summary(self) -> Dict[str, Any]:
        """获取当前病史档案与流量遥测摘要卡"""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "turns_count": len(self.history),
            "dialogue_pairs_count": len(self.get_dialogue_turns()),
            "has_rolling_summary": bool(self.rolling_summary),
            "rolling_summary": self.rolling_summary,
            "active_entity": self.active_entity or "未锁定特定实体",
            "active_domain": self.active_domain or "全科/综合工程",
            "active_stage": self.active_stage or "初诊咨询",
            "has_history": len(self.history) > 0,
            "traffic_metrics": self.get_traffic_metrics()
        }

    def to_dict(self) -> Dict[str, Any]:
        """序列化当前会话状态（用于持久化与恢复）"""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "history": list(self.history),
            "rolling_summary": self.rolling_summary,
            "summarized_msg_count": self.summarized_msg_count,
            "active_entity": self.active_entity,
            "active_domain": self.active_domain,
            "active_stage": self.active_stage,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "traffic_counts": dict(self.traffic_counts),
            "total_requests": self.total_requests,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """从持久化字典恢复会话状态"""
        obj = cls(
            session_id=data.get("session_id", "default_session"),
            title=data.get("title", "新对话"),
        )
        obj.history = list(data.get("history", []))
        obj.rolling_summary = data.get("rolling_summary", "") or ""
        obj.summarized_msg_count = int(data.get("summarized_msg_count", 0) or 0)
        obj.active_entity = data.get("active_entity")
        obj.active_domain = data.get("active_domain")
        obj.active_stage = data.get("active_stage") or "病症初诊与咨询"
        obj.created_at = float(data.get("created_at", time.time()))
        obj.last_updated = float(data.get("last_updated", time.time()))
        raw_tc = data.get("traffic_counts") or {1: 0, 2: 0, 3: 0}
        obj.traffic_counts = {int(k): int(v) for k, v in raw_tc.items()}
        obj.total_requests = int(data.get("total_requests", 0) or 0)
        return obj

    def clear(self):
        """清空病史档案与状态（保留流量遥测统计）"""
        self.title = "新对话"
        self.history = []
        self.rolling_summary = ""
        self.summarized_msg_count = 0
        self.active_entity = None
        self.active_domain = None
        self.active_stage = "初诊咨询"
        self.last_updated = time.time()


# =====================================================================
# 规则匹配字典与实体正则定义
# =====================================================================

GREETING_WORDS = [
    "你好", "您好", "哈喽", "hello", "hi", "hey", "你最近咋样", "最近咋样", "最近怎么样", "你怎么样", "过得怎么样", "在干嘛", "在干什么",
    "早上好", "中午好", "下午好", "晚上好", "早安", "午安", "晚安",
    "在吗", "在不在", "有人吗", "谢谢", "感谢", "多谢", "thx", "thanks",
    "再见", "拜拜", "bye", "好的", "ok", "okay", "收到", "明白", "行", "没问题"
]

HELP_AND_MENU_WORDS = [
    "你是谁", "你的名字", "介绍一下你自己", "你能做什么", "你有什么功能", "你是什么模型", "你是基于什么模型", "使用的什么模型", "底座模型", "你的大模型",
    "系统介绍", "帮助菜单", "功能列表", "使用说明", "帮助", "help", "menu"
]

RESET_COMMANDS = [
    "清空会话", "重置会话", "清除当前会话", "重新开始", "清空历史", "清空对话", "重置对话"
]

COMPLEX_TOOL_TRIGGERS = [
    "生成规范书", "起草", "规范书", "生成技术规范", "生成可研报告", "撰写报告", "自动化生成文档",
    "跨源分析", "多科室会诊", "联合会诊", "冲突仲裁", "跨源仲裁", "立项报告", 
    "导出docx", "深度推导", "上线评估", "按期上线", "综合评估", "多源核查",
    "架构决策", "脑裂", "多步骤规划", "会诊"
]

# 精准实体正则匹配
PR_REGEX = re.compile(r"(PR|pull request)\s*#?(\d+)", re.IGNORECASE)
ISSUE_REGEX = re.compile(r'\b([A-Z]{2,6}-\d+)\b')
CLUSTER_REGEX = re.compile(r"(v[1-3][\s\-_]?cluster|us[\-_]west[\-_][1-3][a-c]?|k8s|kubernetes|prod|staging|us-east-\d)", re.IGNORECASE)

# 代词与指代消解触发词
PRONOUN_TRIGGERS = [
    "它", "它的", "这个", "该项目", "该集群", "该工单", "该pr", "刚才说的", 
    "上面提到的", "方案呢", "还有什么", "进展呢", "状态如何", "为什么", "怎么解决",
    "测试通过了吗", "能上线吗", "容灾方案呢", "在美西呢", "备份呢"
]

# 精准工单/PR直接查表索引 (本地缓存快速精准匹配，0算力直出)
FAST_TICKET_LOOKUP = {
    "ENG-39100": {
        "title": "v2-cluster 美西多可用区跨 Region 容灾同步异常",
        "status": "In Progress (研发修复中)",
        "priority": "P0 Blocker",
        "assignee": "alex.dev@corp.internal",
        "stage": "代码审查(PR) 与 CI 验证",
        "summary": "该工单针对 v2 集群在 us-west-2a 与 2b 之间的高并发网络分区异常进行修复，关联 PR-402 已提交分支待合入。"
    },
    "JIRA-1024": {
        "title": "支付核心链路高并发超时熔断降级优化",
        "status": "Resolved (已解决待验证)",
        "priority": "P1 High",
        "assignee": "sarah.ops@corp.internal",
        "stage": "灰度观察与放量",
        "summary": "已在 staging 环境引入 Sentinel 动态流量流控规则，P99 耗时从 180ms 降低至 22ms。"
    },
    "PR-402": {
        "title": "fix(cluster): resolve cross-az split-brain failover sync",
        "status": "Open (Review Approved, CI Running)",
        "author": "alex.dev",
        "stage": "CI自动化测试验证",
        "summary": "重构了 Raft 心跳租约超时时间参数，完善了网络抖动下的幂等重试机制。"
    }
}


def extract_math_calculation(query: str):
    """识别并提取基础算术表达式 (0大模型算力，0Token，分诊台规则直算)"""
    import re
    q_strip = query.strip()
    clean = re.sub(r'^(计算|算一下|求值|请问|帮我算算)?\s*', '', q_strip)
    clean = re.sub(r'[\s=\?？]+$', '', clean)
    clean = re.sub(r'(等于几|等于多少|等于)+$', '', clean)
    clean = clean.replace('×', '*').replace('x', '*').replace('X', '*').replace('÷', '/')
    clean = clean.replace('加', '+').replace('减', '-').replace('乘', '*').replace('除以', '/').replace('除', '/')
    clean = clean.strip()
    if re.match(r'^\d+(\.\d+)?(\s*[\+\-\*\/]\s*\d+(\.\d+)?)+$', clean):
        try:
            val = eval(clean, {'__builtins__': None}, {})
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            return clean, val
        except Exception:
            return None
    return None

class ProgressiveFunnelRouter:
    """
    渐进式漏斗决策路由器 (Progressive Funnel Router)
    严格执行“算力梯次配置 (Model Cascading)”：
    - 【第 1 层：规则分诊台 (拦截 ~60%)] - 0 大模型算力，<1ms 秒级极速直出，无需调阅病史与知识库
    - 【第 2 层：专科门诊 (承接 ~35%)] - 轻量级专科 SLM，调阅病史消除指代，单知识库标准 RAG
    - 【第 3 层：多科室联合会诊 (<5%)] - 重型推理 LLM，多工具调用链，跨源事实冲突仲裁与长文档起草
    """

    @classmethod
    def route(cls, query: str, session: SessionState) -> Dict[str, Any]:
        clean_q = query.strip()
        clean_lower = clean_q.lower()
        # 提前提取实体，供后续各层改写使用
        detected_entity = None
        cluster_m = CLUSTER_REGEX.search(clean_q)
        issue_m = ISSUE_REGEX.search(clean_q)
        pr_m = PR_REGEX.search(clean_q)
        if cluster_m:
            detected_entity = cluster_m.group(0)
        elif issue_m:
            detected_entity = issue_m.group(0)
        elif pr_m:
            detected_entity = f"PR-{pr_m.group(2)}"

        pure_alpha = re.sub(r"[\s!！?？~。.,，#]+", "", clean_lower)

        # -------------------------------------------------------------
        # 1. 优先判定：是否触发【Level 3: 多科室联合会诊 (严格压制在 <5%)]
        # -------------------------------------------------------------
        matched_trigger = None
        # 优先检测针对上轮实体的代词与短追问指代
        target_entity_l3 = detected_entity or session.active_entity
        l3_rewritten_q = clean_q
        l3_history_used = False
        if target_entity_l3 and any(pron in clean_q for pron in PRONOUN_TRIGGERS + ["那", "如果", "该"]):
            if "脑裂" in clean_q or "分区" in clean_q:
                l3_rewritten_q = f"{target_entity_l3} 出现网络脑裂与多机房分区故障时的仲裁决策与应急处理流程"
                l3_history_used = True
            elif clean_q.startswith("那") or clean_q.startswith("如果"):
                clean_tail = clean_q.lstrip("那").lstrip("如果").strip("，。？? ")
                l3_rewritten_q = f"针对 {target_entity_l3}，{clean_tail} 的跨源协同仲裁方案"
                l3_history_used = True

        for trig in COMPLEX_TOOL_TRIGGERS:
            if trig in clean_lower or trig in l3_rewritten_q.lower():
                matched_trigger = trig
                break

        # 跨源复合多系统决策特征（会议纪要 + 缺陷工单 + 代码PR + 上线决策）
        is_multi_system = (
            ("会议" in clean_q or "规范" in clean_q or "纪要" in clean_q) and
            ("缺陷" in clean_q or "jira" in clean_lower or "bug" in clean_lower or "eng-" in clean_lower) and
            ("上线" in clean_q or "评估" in clean_q or "能否" in clean_q)
        )

        if matched_trigger or is_multi_system:
            session.record_traffic(3)
            trigger_reason = matched_trigger if matched_trigger else "跨源多系统协同与上线风险推演"
            tier_info = MODEL_TIER_PROFILES[3]
            return {
                "level": 3,
                "level_name": "Level 3: 工具兜底层（多科室联合会诊）",
                "funnel_badge": "🏥 Level 3: 多科室联合会诊（工具兜底层）",
                "badge_color": "#dc2626", # 红色
                "action": "joint_consultation",
                "requires_heavy_llm": True,
                "target_tool": "deepseek-reasoner",
                "tier_profile": tier_info,
                "reason": f"检测到复杂跨源规划/长文档生成/多方决策指令 [{trigger_reason}]，唤醒重型推理专家会诊（严控流量 <5%），执行多工具编排与跨源冲突仲裁。",
                "rewritten_query": l3_rewritten_q,
                "original_query": clean_q,
                "history_used": l3_history_used,
                "needs_rag": True,
                "target_domain": "架构与技术规划科",
                "state_summary": session.get_summary()
            }

        # -------------------------------------------------------------
        # 2. 判定：是否属于【会话重置指令】
        # （仅保留纯功能性的重置会话操作，不再用硬编码规则拦截日常问答）
        # -------------------------------------------------------------
        tier_info_1 = MODEL_TIER_PROFILES[1]

        is_session_reset = (
            pure_alpha in ("重置", "reset", "clear")
            or (len(clean_q) <= 12 and any(cmd in clean_lower for cmd in RESET_COMMANDS))
        )
        if is_session_reset:
            session.clear()
            session.record_traffic(1)
            return {
                "level": 1,
                "level_name": "Level 1: 会话重置",
                "funnel_badge": "🔄 会话已重置",
                "badge_color": "#16a34a",
                "action": "session_reset",
                "requires_heavy_llm": False,
                "tier_profile": tier_info_1,
                "triage_type": "reset",
                "reason": "命中会话重置指令，已清空当前会话历史与状态。",
                "rewritten_query": clean_q,
                "state_summary": session.get_summary()
            }

        # -------------------------------------------------------------
        # 3. 核心二元路由判定：
        #    - 企业内部知识库查询 ➔ 调用 RAG 检索 + 大模型回答 (needs_rag = True)
        #    - 其他正常日常问答   ➔ 直接调用 DeepSeek 大模型回答 (needs_rag = False)
        # -------------------------------------------------------------
        session.record_traffic(2)
        tier_info_2 = MODEL_TIER_PROFILES[2]

        # 步骤 1：抽取当前 Query 中的企业业务实体
        detected_entity = None
        issue_match = ISSUE_REGEX.search(clean_q)
        if issue_match:
            detected_entity = issue_match.group(1).upper()
        elif PR_REGEX.search(clean_q):
            detected_entity = f"PR-{PR_REGEX.search(clean_q).group(2)}"
        elif CLUSTER_REGEX.search(clean_q):
            detected_entity = CLUSTER_REGEX.search(clean_q).group(0).lower()

        # 步骤 2：识别是否涉及企业专科业务域
        detected_domain = None
        if any(w in clean_lower for w in ["部署", "集群", "region", "k8s", "运维", "可用区", "failover", "容灾", "备份", "机房", "高可用", "驱逐", "软驱逐"]):
            detected_domain = "运维与基础设施科"
        elif any(w in clean_lower for w in ["代码规范", "pr", "bug", "issue", "分支", "commit", "review", "ci", "工单", "jira"]):
            detected_domain = "敏捷研发工程科"
        elif any(w in clean_lower for w in ["客户", "商机", "合同", "hubspot", "crm", "报价", "交付", "履约"]):
            detected_domain = "客户治理与商务科"

        # 判断是否为针对上一轮企业实体的指代追问
        has_pronoun = any(pron in clean_q for pron in PRONOUN_TRIGGERS)
        is_self_or_general = any(w in clean_lower for w in HELP_AND_MENU_WORDS + GREETING_WORDS)
        is_entity_followup = bool(
            session.active_entity is not None
            and not is_self_or_general
            and has_pronoun
        )

        # 步骤 2.5：企业私有知识库必要性判定 (Enterprise Domain Gate)
        enterprise_keywords = [
            "jira", "confluence", "github", "pr", "slack", "soc2", "k8s", "集群", "部署",
            "工单", "缺陷", "容灾", "架构", "双活", "降级", "合规", "发版", "上线", "审批",
            "可用区", "region", "failover", "运维", "机房", "高可用", "分支", "commit",
            "review", "ci", "流水线", "驱逐", "软驱逐", "hubspot", "crm", "商机",
            "知识库", "内部规范", "技术规范", "会议纪要", "fireflies", "linear"
        ]
        has_enterprise_clues = (
            detected_entity is not None
            or detected_domain is not None
            or is_entity_followup
            or any(w in clean_lower for w in enterprise_keywords)
        )
        needs_rag = bool(has_enterprise_clues)

        # 步骤 3：仅在企业知识库查询时，执行指代消解与状态机更新
        rewritten_q = clean_q
        history_used = False

        if needs_rag:
            inferred_stage = None
            if any(w in clean_lower for w in ["建单", "新建", "申报", "提报"]):
                inferred_stage = "缺陷建单与需求初审"
            elif any(w in clean_lower for w in ["review", "审查", "代码审查", "pr", "分支"]):
                inferred_stage = "代码审查 (Code Review)"
            elif any(w in clean_lower for w in ["测试", "ci", "流水线", "验证", "通过"]):
                inferred_stage = "CI/CD 自动化测试与验证"
            elif any(w in clean_lower for w in ["部署", "上线", "容灾", "发布", "备份", "多可用区"]):
                inferred_stage = "上线部署与容灾评估"

            target_entity = detected_entity or session.active_entity
            if target_entity and is_entity_followup and not detected_entity:
                if "容灾" in clean_q or "备份" in clean_q:
                    rewritten_q = f"{target_entity} 在部署中的容灾备份与多可用区方案是什么？"
                elif "脑裂" in clean_q or "分区" in clean_q:
                    rewritten_q = f"{target_entity} 出现跨机房网络脑裂与分区故障时的自动恢复策略与应对规范"
                elif "测试" in clean_q or "通过" in clean_q or "ci" in clean_lower:
                    rewritten_q = f"{target_entity} 的自动化测试构建与 CI 验证是否通过？"
                elif "进展" in clean_q or "状态" in clean_q:
                    rewritten_q = f"{target_entity} 的当前解决进展与最新状态是什么？"
                elif "美西" in clean_q:
                    rewritten_q = f"{target_entity} 在美西 Region (us-west-2) 的部署限制与高可用方案是什么？"
                elif clean_q.startswith("那") or clean_q.startswith("如果"):
                    clean_tail = clean_q.lstrip("那").lstrip("如果").strip("，。？? ")
                    rewritten_q = f"针对 {target_entity}，{clean_tail} 的标准处理方案与处置流程"
                else:
                    rewritten_q = f"关于 {target_entity}，{clean_q}"
                history_used = True

            final_entity = detected_entity or session.active_entity
            final_domain = detected_domain or session.active_domain or "敏捷研发工程科"
            final_stage = inferred_stage or session.active_stage or "知识库检索问答"
            session.update_state(entity=final_entity, domain=final_domain, stage=final_stage)

            reason_text = f"识别为企业内部知识库查询，调用 RAG 检索（业务域：【{final_domain}】）。"
            if history_used:
                reason_text += f" 已结合历史上下文补全指代为：「{rewritten_q}」。"
            action_name = "specialist_rag"
        else:
            final_domain = "通用大模型直连"
            reason_text = "识别为日常普通问答/通用对话，不检索知识库，直接调用 DeepSeek 大模型生成回答。"
            action_name = "direct_chat"

        return {
            "level": 2,
            "level_name": "RAG 知识库问答" if needs_rag else "大模型直接问答",
            "funnel_badge": "🩺 企业知识库 RAG 检索问答" if needs_rag else "⚡ DeepSeek 大模型直连问答",
            "badge_color": "#2563eb" if needs_rag else "#0284c7",
            "action": action_name,
            "needs_rag": needs_rag,
            "requires_heavy_llm": False,
            "tier_profile": tier_info_2,
            "reason": reason_text,
            "rewritten_query": rewritten_q,
            "original_query": clean_q,
            "history_used": history_used,
            "target_domain": final_domain,
            "state_summary": session.get_summary()
        }

    @classmethod
    def get_triage_reply(cls, query: str, funnel_res: Dict[str, Any]) -> str:
        """
        Level 1 规则分诊台的标准免算力极速答复模板 (0 Token, <1ms 响应)
        """
        triage_type = funnel_res.get("triage_type", "greeting")

        if triage_type == "reset":
            return (
                "🔄 **【分诊台】会话病史已清空重置**\n\n"
                "您既往的就诊病史记录、锁定的实体对象与业务阶段已全部重置为初始状态。\n"
                "欢迎随时提出新的企业内部知识、技术规范或工单咨询！"
            )

        if triage_type == "empty_or_punctuation":
            return (
                "👋 **【分诊台】您好！我是企业智能知识库分诊助手**\n\n"
                "请输入您想要咨询的具体技术规范、架构方案、运维集群或工单号（如 `ENG-39100`），我将为您快速导诊至对应专科！"
            )

        if triage_type == "math_calculation":
            expr = funnel_res.get("math_expr", query)
            val = funnel_res.get("math_val", "")
            return (
                f"🧮 **【分诊台·规则引擎秒算】**\n\n"
                f"- **计算表达式**：`{expr}`\n"
                f"- **精准计算结果**：**`{val}`**\n\n"
                f"⚡ *该结果由分诊台内置计算引擎在 `< 1ms` 内直出，零 Token 消耗，100% 旁路企业私有知识库检索与大模型算力。*"
            )

        if triage_type == "ticket_lookup":
            t_data = funnel_res.get("ticket_data", {})
            t_id = funnel_res.get("target_ticket", "工单")
            return (
                f"📋 **【分诊台·快速单号直查】{t_id}**\n\n"
                f"- **标题**：{t_data.get('title', 'N/A')}\n"
                f"- **当前状态**：`{t_data.get('status', 'N/A')}`\n"
                f"- **严重级别**：`{t_data.get('priority', 'N/A')}`\n"
                f"- **当前经办人**：`{t_data.get('assignee', 'N/A')}`\n"
                f"- **业务阶段**：`{t_data.get('stage', 'N/A')}`\n\n"
                f"💡 **概要摘要**：{t_data.get('summary', '')}\n\n"
                f"*(⚡ 本次查询由 Level 1 规则分诊台直接基于静态索引秒级返回，未消耗任何大模型 Token)*"
            )

        if triage_type == "system_help":
            return (
                "🤖 **你好！我是由多模型梯次协同驱动的企业智能知识库助手。**\n\n"
                "针对你询问的**「你是什么模型」**，我们系统采用的是**算力梯次配置 (Model Cascading)** 架构，并不是单一模型，而是由以下多层模型梯次分工协同工作的：\n\n"
                "1. **⚡ 日常业务问答与专科检索（第 2 层主力模型）**\n"
                "   - **底座模型**：`DeepSeek-Chat (V3)` / 轻量级专科 SLM (Small Language Model)\n"
                "   - **主要职责**：负责 ~35% 的日常问答，调阅历史上下文消解代词指代，在单一知识库内做快速精准 RAG（响应 0.5s~1s，低成本）。\n\n"
                "2. **🧠 复杂疑难与多工具推理（第 3 层重型大模型）**\n"
                "   - **底座模型**：`DeepSeek-Reasoner (R1)` 深度思考模型 / 专家级 Agent\n"
                "   - **主要职责**：严格卡在 <5% 流量，专啃硬骨头，负责思维链深度推理（CoT）、跨 Jira/GitHub/会议纪要的多源冲突仲裁与长篇报告生成。\n\n"
                "3. **🛡️ 纯规则与高频分诊（第 1 层前置防破产）**\n"
                "   - **底座算力**：**0 大模型算力**（纯代码规则、正则、静态索引与高频字典）\n"
                "   - **主要职责**：拦截 ~60% 的日常闲聊、算术或工单号直查（响应 <1ms，0 Token 成本）。\n\n"
                "---\n"
                "💡 **一句话总结**：日常问答由 **DeepSeek-Chat** 快速回答，遇到复杂多源疑难时由 **DeepSeek-Reasoner (R1)** 深度会诊！"
            )

        # 默认日常问候
        return (
            "👋 **您好！欢迎使用企业多源智能知识库**\n\n"
            "我是您的规则分诊台助手。本知识库已汇聚 Confluence、Jira、GitHub、Slack 等企业核心数据源。\n\n"
            "您可以向我咨询：\n"
            "- 🛠️ **敏捷研发工程**：代码规范、PR 审查、CI 验证、Jira 缺陷跟踪；\n"
            "- ☁️ **运维与基础设施**：v2 集群部署规范、多可用区容灾策略、网络分区应急预案；\n"
            "- 💼 **客户治理与商务**：商机跟进、客户成功合同履约标准。\n\n"
            "请直接输入您的问题，我将为您精准导流！"
        )
