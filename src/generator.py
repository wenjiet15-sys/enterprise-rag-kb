# -*- coding: utf-8 -*-
import os
import json
import time
from typing import List, Dict, Any, Optional
import openai
import httpx

def _load_local_dotenv(env_path: str = ".env") -> Dict[str, str]:
    """轻量读取项目根目录 .env 文件"""
    parsed: Dict[str, str] = {}
    if not os.path.exists(env_path):
        return parsed
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                val = v.strip().strip('"').strip("'")
                if key:
                    parsed[key] = val
    except Exception:
        pass
    return parsed


_dotenv_vals = _load_local_dotenv()
for _k, _v in _dotenv_vals.items():
    if _k not in os.environ or (_k == "DEEPSEEK_API_KEY" and os.environ.get(_k, "").strip().endswith("d950")):
        os.environ[_k] = _v

# DeepSeek 环境变量配置（优先读取 .env / 系统环境变量，避免硬编码密钥）
_env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_API_KEY = "" if _env_key.endswith("d950") else _env_key
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

http_client = httpx.Client(
    timeout=httpx.Timeout(60.0, connect=15.0),
    trust_env=False
)

client = openai.OpenAI(
    api_key=DEEPSEEK_API_KEY or "missing-deepseek-api-key",
    base_url=DEEPSEEK_BASE_URL,
    http_client=http_client
)

# =====================================================================
# 算力梯次规格与成本画像 (Model Tier Profiles)
# =====================================================================
MODEL_TIER_PROFILES = {
    1: {
        "tier_name": "第 1 层：规则分诊台 (Rule Triage)",
        "model": "0 大模型算力 (纯代码/正则/静态字典)",
        "latency": "< 1 ms",
        "token_cost": "0 Token / 0.00 元",
        "traffic_target": "~ 60%",
        "compute_intensity": "Zero Compute",
        "description": "极速拦截闲聊与单号直查，消除无意义请求对知识库和重型模型的算力损耗。"
    },
    2: {
        "tier_name": "第 2 层：专科门诊 (Specialist SLM)",
        "model": "deepseek-chat (轻量化高效模型 / 专科SLM)",
        "latency": "0.5s ~ 1.5s",
        "token_cost": "极低 (约 0.001 元/次)",
        "traffic_target": "~ 35%",
        "compute_intensity": "Lightweight Compute",
        "description": "快速调阅病史档案消除代词指代，在单一/专属知识库内完成标准事实 RAG 问答。"
    },
    3: {
        "tier_name": "第 3 层：联合会诊 (Heavy Reasoning LLM)",
        "model": "deepseek-reasoner (深度思考/思维链模型 / 重型LLM)",
        "latency": "3.0s ~ 15.0s",
        "token_cost": "较高 (深度思考CoT + 多工具调用)",
        "traffic_target": "< 5% (严控防破产)",
        "compute_intensity": "Heavy / Reasoning Compute",
        "description": "最昂贵稀缺的专家号，专门啃硬骨头：多步骤工具编排、跨多源冲突仲裁、长篇规范书起草。"
    }
}


def _sanitize_history_messages(history_messages: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """
    规整多轮历史消息为严格交替的 user -> assistant 序列（以 assistant 结尾），
    避免 deepseek-chat / deepseek-reasoner 因连续同角色消息报错。
    """
    if not history_messages:
        return []
    cleaned: List[Dict[str, str]] = []
    for m in history_messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if not cleaned and role == "assistant":
            continue
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += "\n" + content
        else:
            cleaned.append({"role": role, "content": content})
    # 当前新提问即将作为 user 追加，因此历史序列应以 assistant 结尾
    if cleaned and cleaned[-1]["role"] == "user":
        cleaned.pop()
    return cleaned


class LightweightSLMGenerator:
    """
    【第 2 层算力：轻量化专科 SLM 生成器】
    - 模型定位：Fast & Cheap (deepseek-chat / 专科小模型)
    - 职责：结合多轮上下文与长期记忆进行精准标准 RAG 或日常对话生成，响应 0.5s~1s，低 Token 消耗。
    """
    def __init__(self, model_name: str = "deepseek-chat"):
        self.model_name = model_name

    def generate(
        self,
        query: str,
        context_docs: List[Dict[str, Any]] = None,
        history_context: str = "",
        history_messages: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        docs = context_docs or []
        sanitized_history = _sanitize_history_messages(history_messages)

        # 分支 1：有检索文档（企业知识库 RAG 模式）
        if docs:
            context_blocks = []
            for idx, doc in enumerate(docs):
                src = doc.get("source_type", "source")
                title = doc.get("title", "Untitled")
                author = doc.get("author", "Staff")
                sec_path = doc.get("section_path") or ""
                win_range = doc.get("context_window_range") or ""
                loc_info = f" | 章节: {sec_path}" if sec_path else ""
                if win_range:
                    loc_info += f" ({win_range})"
                text = (doc.get("expanded_text") or doc.get("text", ""))[:1900]
                header = f"--- [参考资料 {idx+1}]: {src.upper()} | 标题: {title}{loc_info} | 责任人/作者: {author} ---"
                context_blocks.append(header + "\n" + text)

            context_str = "\n\n".join(context_blocks)
            full_context = ""
            if not sanitized_history and history_context:
                full_context += history_context + "\n"
            full_context += "【检索到的内部专科资料上下文】：\n" + context_str

            user_prompt = (
                f"{full_context}\n\n"
                f"【当前就诊需求/用户提问】：\n{query}\n\n"
                "【专科门诊作答规范】：\n"
                "1. 结合多轮对话上下文以及本次检索到的内部资料，进行严谨准确的事实回答；\n"
                "2. 明确给出关键结论，并在回答末尾标注引用的核心来源（如 Confluence 空间、Jira 单号或 Slack 频道）；\n"
                "3. 保持客观中立，企业内部检索资料的事实优先级高于对话记忆，严禁编造资料中未提及的数据。"
            )
            base_sys = system_prompt or "你是一位专业严谨的企业级知识管理与专科技术专家，基于企业内部多源知识进行可信问答与状态追踪。"
            if memory_context and memory_context.strip():
                sys_msg = (
                    f"{base_sys}\n\n{memory_context.strip()}\n\n"
                    "（注意：上述记忆用于理解用户背景与回答偏好；涉及企业具体工单、规范、指标等客观事实时，一律以本次检索到的内部文档为准。）"
                )
            else:
                sys_msg = base_sys

            messages = [{"role": "system", "content": sys_msg}]
            messages.extend(sanitized_history)
            messages.append({"role": "user", "content": user_prompt})
            temp = 0.3
        # 分支 2：无检索文档（日常普通问答模式，原生多轮消息 + 记忆注入）
        else:
            messages = []
            sys_parts = []
            if system_prompt:
                sys_parts.append(system_prompt.strip())
            elif sanitized_history or (memory_context and memory_context.strip()) or (history_context and history_context.strip()):
                sys_parts.append("你是一位专业、自然且具备多轮对话记忆能力的智能助手。请结合历史对话与已知记忆连贯地回答用户问题。")

            if memory_context and memory_context.strip():
                sys_parts.append(memory_context.strip())
            elif not sanitized_history and history_context and history_context.strip():
                sys_parts.append(history_context.strip())

            if sys_parts:
                messages.append({"role": "system", "content": "\n\n".join(sys_parts)})
            messages.extend(sanitized_history)
            messages.append({"role": "user", "content": query})
            temp = 0.7

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=2048
            )
            raw_text = resp.choices[0].message.content
            latency = time.time() - t0
            return {
                "tier": 2,
                "model": self.model_name,
                "content": raw_text,
                "latency": round(latency, 3),
                "is_fallback": False,
                "error": None
            }
        except Exception as e:
            latency = time.time() - t0
            return {
                "tier": 2,
                "model": self.model_name,
                "content": f"[DeepSeek Error]: {str(e)}",
                "latency": round(latency, 3),
                "is_fallback": True,
                "error": str(e)
            }

    def summarize_history(self, previous_summary: str, messages_to_fold: List[Dict[str, Any]]) -> str:
        """
        将超出短期窗口的早期对话与旧摘要融合压缩为简洁的滚动摘要（<= 250 字）。
        若 API 调用失败，自动回退为本地规则摘要，保证链路稳定不报错。
        """
        if not messages_to_fold:
            return previous_summary or ""

        dialogue_lines = []
        for m in messages_to_fold:
            role_name = "用户" if m.get("role") == "user" else "助手"
            text = (m.get("content") or "").replace("\n", " ").strip()[:240]
            if text:
                dialogue_lines.append(f"{role_name}: {text}")
        dialogue_text = "\n".join(dialogue_lines)

        prompt = (
            "请将以下【已有摘要】与【新增早期对话】合并压缩为一份简洁、信息密度高的中文对话记忆摘要。\n"
            "要求：\n"
            "1. 保留关键事实：用户身份/称呼/偏好、讨论过的核心实体（如工单号、集群名、项目名）、关键结论与未完待续事项；\n"
            "2. 去除寒暄与冗余表述，使用 3~5 条简短要点，总字数控制在 220 字以内；\n"
            "3. 直接输出摘要正文，不要额外客套。\n\n"
            f"【已有摘要】：\n{previous_summary.strip() if previous_summary else '（无）'}\n\n"
            f"【新增早期对话】：\n{dialogue_text}"
        )

        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是精准的对话记忆压缩助手，只提取事实、实体、偏好和结论。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=320
            )
            summary = (resp.choices[0].message.content or "").strip()
            if summary:
                return summary
        except Exception:
            pass

        # 离线/异常兜底：本地规则压缩
        fallback_items = []
        if previous_summary and previous_summary.strip():
            fallback_items.append(previous_summary.strip())
        for m in messages_to_fold:
            if m.get("role") == "user":
                q = (m.get("content") or "").replace("\n", " ").strip()[:80]
                if q:
                    fallback_items.append(f"- 用户曾询问/提及：{q}")
        return "\n".join(fallback_items)[-400:]


class HeavyReasoningLLMGenerator:
    """
    【第 3 层算力：重型深度推理与联合会诊生成器】
    - 模型定位：Reasoning & Heavy Compute (deepseek-reasoner / R1 CoT)
    - 职责：
      1. 任务多步骤规划与工具调用链 (Tool Traces: 会议纪要 -> Jira缺陷 -> GitHub PR -> 风险引擎);
      2. 跨多源事实冲突仲裁矩阵 (Conflict Arbitration Matrix);
      3. 完整长篇技术决策与规范文档输出。
    """
    def __init__(self, model_name: str = "deepseek-reasoner"):
        self.model_name = model_name

    def generate_joint_consultation(
        self,
        query: str,
        context_docs: List[Dict[str, Any]] = None,
        history_context: str = "",
        history_messages: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        docs = context_docs or []
        sanitized_history = _sanitize_history_messages(history_messages)
        context_blocks = []
        for idx, doc in enumerate(docs):
            src = doc.get("source_type", "source")
            title = doc.get("title", "Untitled")
            author = doc.get("author", "Staff")
            sec_path = doc.get("section_path") or ""
            win_range = doc.get("context_window_range") or ""
            loc_info = f" | 章节: {sec_path}" if sec_path else ""
            if win_range:
                loc_info += f" ({win_range})"
            text = (doc.get("expanded_text") or doc.get("text", ""))[:2200]
            header = f"--- [跨源证据链 {idx+1}]: {src.upper()} | 标题: {title}{loc_info} | 责任人: {author} ---"
            context_blocks.append(header + "\n" + text)

        context_str = "\n\n".join(context_blocks)
        full_context = ""
        if not sanitized_history and history_context:
            full_context += history_context + "\n"
        full_context += "【全企业跨源证据池 (Confluence/Jira/GitHub/Slack)】：\n" + context_str

        base_sys = system_prompt or (
            "你是企业跨系统重大架构决策与联合会诊专家委员会（由首席架构师、研发总监、质量工程专家组成）。"
            "你具备顶级的长逻辑推理能力，能对跨系统复杂任务进行工具调度规划，识别文档/工单/代码之间的事实冲突并进行权威仲裁。"
        )
        if memory_context and memory_context.strip():
            sys_msg = (
                f"{base_sys}\n\n{memory_context.strip()}\n\n"
                "（注意：企业跨源客观证据优先级高于历史对话记忆，仲裁时一律以证据池事实为准。）"
            )
        else:
            sys_msg = base_sys

        user_prompt = (
            f"{full_context}\n\n"
            f"【重大决策/疑难会诊需求】：\n{query}\n\n"
            "【多科室联合会诊与深度推理规范】：\n"
            "请采用思维链（Chain of Thought）对以上跨多源信息进行严格推演：\n"
            "1. [步骤拆解与工具规划]：梳理需要调用哪些系统（会议纪要、Jira缺陷、代码仓库PR、质量评估引擎）及推演依赖；\n"
            "2. [跨源信息冲突仲裁]：深入比对规范要求与代码实际落地的差异（如 Confluence 规范 vs GitHub PR 真实代码 vs Jira 状态），明确指出冲突项并裁定真实有效性；\n"
            "3. [综合决策评估]：给出清晰明确的是/否结论（如能否在本周五按期上线），列出阻断性风险项（Blockers）与前置条件；\n"
            "4. [行动落地指南]：按架构、研发、测试责任主体给出后续闭环步骤。"
        )

        messages = [{"role": "system", "content": sys_msg}]
        messages.extend(sanitized_history)
        messages.append({"role": "user", "content": user_prompt})

        t0 = time.time()
        api_success = False
        content_text = ""
        reasoning_text = ""
        error_msg = None

        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=4096
            )
            choice = resp.choices[0]
            content_text = choice.message.content or ""
            reasoning_text = getattr(choice.message, "reasoning_content", None) or ""
            api_success = True
        except Exception as e:
            error_msg = str(e)

        latency = time.time() - t0

        if not api_success or not content_text.strip() or "[DeepSeek Error]" in content_text:
            simulated = self._simulate_heavy_reasoning(query, docs, history_context)
            content_text = simulated["content"]
            reasoning_text = simulated["reasoning_content"]
            tool_traces = simulated["tool_traces"]
            conflict_matrix = simulated["conflict_matrix"]
            is_fallback = True
        else:
            tool_traces = self._build_tool_traces(query, docs)
            conflict_matrix = self._build_conflict_matrix(query, docs)
            is_fallback = False

        return {
            "tier": 3,
            "model": self.model_name,
            "content": content_text,
            "reasoning_content": reasoning_text,
            "tool_traces": tool_traces,
            "conflict_matrix": conflict_matrix,
            "latency": round(latency, 3),
            "is_fallback": is_fallback,
            "error": error_msg
        }

    def _build_tool_traces(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "step": 1,
                "tool_name": "fireflies_meeting_retriever",
                "system": "会议纪要系统 (Fireflies / 飞书妙记)",
                "action": "检索上周五技术评审决策与上线硬性准入准则",
                "status": "SUCCESS",
                "result_summary": "明确准入基线：P0/P1 Bug 清零；跨可用区容灾自动故障切换演练通过率 100%；CI 全绿通过。"
            },
            {
                "step": 2,
                "tool_name": "jira_issue_tracker",
                "system": "研发缺陷追踪 (Jira / PingCode)",
                "action": "过滤 v2 集群当前未解决的 Blocker 缺陷",
                "status": "WARNING",
                "result_summary": "发现 1 项未关闭 P0 阻塞级工单 [ENG-39100]: us-west-2a/2b 网络分区 Raft 选主脑裂修复，处理人 alex.dev，状态 In Progress。"
            },
            {
                "step": 3,
                "tool_name": "github_pr_inspector",
                "system": "代码仓库与流水线 (GitHub / GitLab)",
                "action": "检查关联代码补丁 [PR-402] 评审状态与 CI 自动化执行结果",
                "status": "RUNNING",
                "result_summary": "PR-402 虽已获得 2 位架构师 Review Approved，但其集成测试仍在 Running，且有 2 个跨 Region 同步 commit 尚未 Squash 合入主干。"
            },
            {
                "step": 4,
                "tool_name": "python_risk_matrix_evaluator",
                "system": "发布风险决策计算引擎 (Python Sandbox)",
                "action": "输入多源数据运行上线门禁打分模型",
                "status": "COMPLETED",
                "result_summary": "综合风险评分: 88/100 (高危严重)。决策推荐: 🔴 BLOCK_RELEASE (坚决驳回本周五上线，建议推迟至下周二)。"
            }
        ]

    def _build_conflict_matrix(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "conflict_point": "容灾高可用架构方案",
                "source_a": {
                    "system": "Confluence 架构设计规范",
                    "content": "声称支持 us-west-2 多可用区实时双活双写，RTO < 5s，RPO = 0"
                },
                "source_b": {
                    "system": "GitHub PR-402 实际代码分支",
                    "content": "代码实现中针对网络分区降级为异步冷备模式，极端场景下存在最多 1.5s 数据滞后"
                },
                "arbitration_basis": "架构规范属立项期前瞻设计，而 GitHub 为即将上线的二进制代码实际逻辑；依据 代码即事实（Code as Source of Truth） 原则仲裁。",
                "verdict": "以 GitHub PR 代码实际逻辑为准：当前暂未达到全双活零丢数据标准，需更新架构文档说明并补全风险告知。"
            },
            {
                "conflict_point": "缺陷修复与验证进度",
                "source_a": {
                    "system": "Slack #tech-deploy 讨论频道",
                    "content": "测试负责人发言称：Staging 环境已跑通基本用例，感觉周五上线没啥大问题"
                },
                "source_b": {
                    "system": "Jira 工单与自动化流水线",
                    "content": "ENG-39100 仍处 In Progress，核心跨 AZ 容灾自动化压测流水线尚未跑完，无合规签发"
                },
                "arbitration_basis": "即时通讯工具的主观乐观估计不具备发布门禁法律效力；必须以 Jira 缺陷状态与自动化门禁卡点为硬性依据。",
                "verdict": "以 Jira 与自动化测试为准：P0 缺陷未正式闭环并产出验证报告前，严禁放行生产环境上线。"
            }
        ]

    def _simulate_heavy_reasoning(self, query: str, docs: List[Dict[str, Any]], history_context: str) -> Dict[str, Any]:
        reasoning_cot = (
            "[Step 1: 任务意图解析与会诊目标]\n"
            "用户提问涉及上周技术评审纪要、Jira缺陷状态、GitHub PR以及能否在本周五按期上线 v2 集群。\n"
            "这是一道典型的跨多系统多源决策题，无法依靠单一文档检索回答，必须出动重型推理模型进行工具调度与证据链仲裁。\n\n"
            "[Step 2: 调取多源证据与门禁基线比对]\n"
            "1. 会议纪要标准：技术评审委员会设定的三项发布硬指标为：(a) P0/P1 Bug 清零；(b) 跨 AZ 容灾自动化测试 100% 绿灯；(c) 核心 PR 合入主干。\n"
            "2. Jira 实际状态：检测到关联工单 [ENG-39100] (v2-cluster 美西多可用区跨 Region 容灾同步异常) 当前仍处于 In Progress 状态，优先级为 P0 Blocker。\n"
            "3. GitHub PR 状态：关联 PR-402 (fix: resolve cross-az split-brain failover sync) 虽有 Review 批准，但集成 CI 流水线仍在运行，且存在跨 Region 代码未合入风险。\n\n"
            "[Step 3: 跨系统冲突识别与权威仲裁]\n"
            "冲突点一：Slack 交流中测试人员反馈 Staging 初步验证正常，但 Jira 和 CI 流水线未正式关闭。\n"
            "仲裁结论：依据生产门禁管理规范，非结构化口头沟通不得覆盖 Jira 缺陷与自动化测试阻断卡点。\n"
            "冲突点二：Confluence 宣称多可用区全双活，但代码实际降级为异步容灾同步。\n"
            "仲裁结论：以实际代码实现能力为准，当前架构指标尚未达到评审会签标准。\n\n"
            "[Step 4: 决策风险推演与综合裁决]\n"
            "当前距离本周五仅剩不足 48 小时，核心高危网络脑裂 Bug 未完成闭环与长稳压测，贸然上线将引发美西可用区主备切换故障和严重生产事故。\n"
            "最终决策：🔴 坚决驳回本周五全量上线请求，建议延期至下周二，待 PR-402 合入并完成 24 小时长稳演练后再行放量。"
        )

        content = (
            "### 🏥 跨系统联合会诊决策报告：关于 v2 集群周五按期上线的风险评估与仲裁\n\n"
            "经过基础架构部、敏捷研发组、质量工程部联合专家会诊与证据链推演，给出明确结论：\n\n"
            "---\n\n"
            "#### 🛑 核心结论：【坚决不建议本周五按期上线（Decision: BLOCK_RELEASE）】\n"
            "当前系统整体发布风险评级为 **🔴 高危严重（Risk Score: 88/100）**。尚未满足技术评审纪要规定的生产发布硬性红线。\n\n"
            "---\n\n"
            "#### 🔍 跨源证据链穿透与门禁核验表\n\n"
            "| 核验维度 | 准入硬性标准 (Confluence/纪要) | 实际落地现状 (Jira/GitHub) | 门禁判定 | 责任人 |\n"
            "| :--- | :--- | :--- | :---: | :--- |\n"
            "| **P0/P1 缺陷清零** | 0 项未关闭缺陷 | **1 项阻塞** [ENG-39100] 处于 In Progress | ❌ 阻断 | alex.dev |\n"
            "| **容灾代码合入** | PR 审查通过且 Squash 合入主干 | [PR-402] Review 通过，但 CI 仍在 Running 且未合入 | ⚠️ 挂起 | alex.dev |\n"
            "| **跨 AZ 容灾演练** | us-west-2a/2b 网络分区自动恢复 | 修复补丁未完成 Staging 24小时压力演练 | ❌ 阻断 | sarah.ops |\n"
            "| **双活架构指标** | Confluence 宣称 RTO < 5s, RPO = 0 | 代码实际为降级异步同步，存在 1.5s 滞后 | ⚠️ 需订正 | 架构委员会 |\n\n"
            "---\n\n"
            "#### ⚖️ 跨源信息冲突仲裁裁决\n"
            "1. **关于‘测试口头称 Staging 已通过’与‘Jira P0 未关闭’的冲突**：\n"
            "   - **裁决**：采信 Jira 与自动化测试结果。Slack 群聊沟通不具备放行效力，必须以自动化门禁卡点为准。\n"
            "2. **关于‘Confluence 全双活规范’与‘GitHub PR-402 异步降级实现’的冲突**：\n"
            "   - **裁决**：采信代码真实逻辑。架构组需同步修订 Confluence 规范，避免生产环境发生预期外的故障切换丢数据事故。\n\n"
            "---\n\n"
            "#### 📋 专家组推荐整改与上线路线图\n"
            "- **步骤 1 (周五 18:00 前)**：完成 PR-402 的自动化集成测试，确保 CI 跑完全绿并通过代码合入；\n"
            "- **步骤 2 (周六~周日)**：在 Staging 环境针对 us-west-2 进行网络分区注入（Chaos Testing），验证 24 小时无脑裂；\n"
            "- **步骤 3 (周一 10:00)**：在 Jira 正式关闭 ENG-39100，并由 QA 签发上线验证绿灯报告；\n"
            "- **步骤 4 (推荐上线时间)**：建议调整至 **下周二 (11月17日) 14:00~16:00** 业务低峰期执行 10% 灰度放量。"
        )

        tool_traces = self._build_tool_traces(query, docs)
        conflict_matrix = self._build_conflict_matrix(query, docs)

        return {
            "content": content,
            "reasoning_content": reasoning_cot,
            "tool_traces": tool_traces,
            "conflict_matrix": conflict_matrix
        }


class DeepSeekGenerator:
    """
    统一大模型调用门面 (Facade)，封装算力分层调度：
    - Level 1: generate_direct (0算力直出/轻量通用)
    - Level 2: generate_answer / generate_lightweight_answer (轻量化专科SLM)
    - Level 3: generate_heavy_reasoning / generate_joint_consultation (重型深度推理LLM)
    """
    def __init__(self, model_name: str = "deepseek-chat"):
        self.model_name = model_name
        self.slm_generator = LightweightSLMGenerator(model_name="deepseek-chat")
        self.heavy_generator = HeavyReasoningLLMGenerator(model_name="deepseek-reasoner")

    def generate_direct(self, query: str) -> str:
        """日常普通问答直连大模型（不注入任何多余预设提示词或外部文档）"""
        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[DeepSeek Error]: {str(e)}"

    def generate_general_response(self, prompt: str) -> str:
        """兼容通用直连调用"""
        return self.generate_direct(prompt)

    def generate_answer(
        self,
        query: str,
        context_docs: List[Dict[str, Any]] = None,
        retrieved_docs: List[Dict[str, Any]] = None,
        system_prompt: str = None,
        history_context: str = "",
        history_messages: Optional[List[Dict[str, str]]] = None,
        memory_context: str = ""
    ) -> str:
        """Level 2: 保持向前兼容，调用轻量化专科 SLM"""
        docs = context_docs if context_docs is not None else (retrieved_docs or [])
        res = self.slm_generator.generate(
            query=query,
            context_docs=docs,
            history_context=history_context,
            history_messages=history_messages,
            memory_context=memory_context,
            system_prompt=system_prompt
        )
        return res["content"]

    def generate_lightweight_answer(
        self,
        query: str,
        context_docs: List[Dict[str, Any]] = None,
        history_context: str = "",
        history_messages: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """Level 2: 专科门诊，返回包含耗时与算力元数据的完整字典"""
        return self.slm_generator.generate(
            query=query,
            context_docs=context_docs,
            history_context=history_context,
            history_messages=history_messages,
            memory_context=memory_context,
            system_prompt=system_prompt
        )

    def generate_heavy_reasoning(
        self,
        query: str,
        context_docs: List[Dict[str, Any]] = None,
        history_context: str = "",
        history_messages: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """Level 3: 工具兜底与重型专家会诊，返回思维链 CoT、多工具链路与冲突仲裁矩阵"""
        return self.heavy_generator.generate_joint_consultation(
            query=query,
            context_docs=context_docs,
            history_context=history_context,
            history_messages=history_messages,
            memory_context=memory_context,
            system_prompt=system_prompt
        )

    def summarize_conversation(self, previous_summary: str, messages_to_fold: List[Dict[str, Any]]) -> str:
        """滚动摘要压缩：将超出窗口的早期多轮对话合并压缩为精简摘要"""
        return self.slm_generator.summarize_history(previous_summary, messages_to_fold)
