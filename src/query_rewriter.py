# -*- coding: utf-8 -*-
# src/query_rewriter.py
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import src.generator as _gen_mod
from src.intent_router import SessionState, GREETING_WORDS, HELP_AND_MENU_WORDS


# 指代词与多轮省略承接信号词
COREF_FOLLOWUP_CUES = (
    "它", "它的", "它们", "这个", "那个", "这些", "那些", "该", "其",
    "刚才", "上面", "前面", "上文", "之前", "提到的", "说的",
    "为什么", "怎么解决", "怎么办", "怎么处理", "详细说说", "展开", "举例",
    "继续", "还有呢", "还有什么", "那如果", "如果", "对比", "区别",
    "方案呢", "进展呢", "状态如何", "报价呢", "价格呢", "sla", "参数呢",
)

# 强显式代词（用于区分“明显指代追问”与“可能切换新话题的独立短句”）
EXPLICIT_PRONOUN_CUES = (
    "它", "它的", "它们", "这个", "那个", "这些", "那些", "该", "其",
    "刚才", "上面", "前面", "上文", "之前", "提到的", "说的", "进展呢", "方案呢", "报价呢", "价格呢",
)

# 否定与强限定词（防止改写时意外丢失核心约束）
CONSTRAINT_CUES = ("不能", "禁止", "不允许", "不可", "除了", "仅限", "只允许", "不得", "不要", "not ", "only ")

# 模型误把“改写问题”写成“直接作答”的前缀信号
ANSWER_LIKE_PREFIXES = ("根据上下文", "根据前序", "根据对话", "答案是", "答：", "结论是", "综上所述", "可以确认")

# 关键结构化实体提取正则（工单、PR、云区域、版本号、安全密级）
_ENTITY_PATTERNS = [
    re.compile(r"\b(?:eng|jira|lin)-\d+\b", re.IGNORECASE),
    re.compile(r"\bpr[\s\-_#]*\d+\b", re.IGNORECASE),
    re.compile(r"\bus-(?:west|east|central)-\d[a-z]?\b", re.IGNORECASE),
    re.compile(r"\bv\d+(?:\.\d+)*\b", re.IGNORECASE),
]

# 离线/API 异常兜底：常见中文企业术语 -> 英文等价检索词映射
_BILINGUAL_FALLBACK_MAP = [
    ("美西", "us-west"),
    ("美东", "us-east"),
    ("集群", "cluster"),
    ("部署", "deploy deployment allowed regions"),
    ("区域", "regions"),
    ("可用区", "availability zone multi-az"),
    ("容灾", "failover disaster recovery dr"),
    ("高可用", "high availability ha"),
    ("脑裂", "split-brain"),
    ("工单", "jira issue ticket"),
    ("缺陷", "bug blocker issue"),
    ("代码审查", "pull request code review"),
    ("客户", "customer account hubspot"),
    ("报价", "pricing quote contract"),
    ("商机", "deal opportunity pipeline"),
    ("会议纪要", "meeting transcript action items"),
    ("延迟", "latency slo"),
    ("监控", "monitoring alert dashboard"),
    ("权限", "rbac permission access control"),
]

_HAS_ZH_RE = re.compile(r"[\u4e00-\u9fa5]")
_ZH_FILLERS = set("的了吗呢啊呀吧哦嘛把被给在和与跟对从向就都也还又很太非常请问帮我查一下看一下了解")


class QueryRewriter:
    """
    三阶门控 + 置信度护栏智能用户问题改写引擎：
    - 第 1 阶：0 成本直通（简单寒暄/独立清晰闲聊直接跳过，0 Token 开销）
    - 第 2 阶：多轮指代消解与省略补全（结合最近 3 轮对话 + 滚动摘要 + 锁定实体）
    - 第 3 阶：RAG 跨语与专业术语扩展（中文查英文知识库时自动生成 1~2 个英文等价检索词 Multi-Query）
    - 置信度防漂移护栏：确定性实体保真/反幻觉检查 + LLM自评与规则保真融合打分 + 双阈值分级回退
    安全承诺：仅读取当前用户自己的历史会话，绝不接触知识库文档；API 异常时自动降级本地规则兜底。
    """

    def __init__(self, model_name: str = "deepseek-chat"):
        self.model_name = model_name

    @staticmethod
    def _extract_key_entities(text: str) -> List[str]:
        """提取关键结构化实体并归一化（如 PR 402 -> pr-402）"""
        if not text:
            return []
        found: List[str] = []
        seen = set()
        for pat in _ENTITY_PATTERNS:
            for m in pat.findall(text):
                norm = re.sub(r"[\s_#]+", "-", m.strip().lower())
                norm = re.sub(r"^pr(\d+)$", r"pr-\1", norm)
                if norm not in seen:
                    seen.add(norm)
                    found.append(norm)
        return found

    @staticmethod
    def _extract_anchor_tokens(text: str) -> List[str]:
        """提取原句中去除指代词与虚词后的实义锚点词（英文词 + 中文实义字符）"""
        clean = (text or "").lower()
        for cue in sorted(EXPLICIT_PRONOUN_CUES, key=len, reverse=True):
            clean = clean.replace(cue, " ")
        en_words = re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", clean)
        zh_chars = [ch for ch in re.findall(r"[\u4e00-\u9fa5]", clean) if ch not in _ZH_FILLERS]
        return list(dict.fromkeys(en_words + zh_chars))

    @classmethod
    def _check_hard_guardrails(
        cls,
        original_query: str,
        rewritten_query: str,
        history_block: str,
    ) -> Tuple[bool, str]:
        """
        第一道防线：确定性硬护栏（0ms 拦截明显漂移或幻觉）。
        返回 (是否通过, 违规原因)。
        """
        orig = (original_query or "").strip()
        rew = (rewritten_query or "").strip()
        if not rew:
            return False, "改写结果为空"

        # 1. 长度膨胀检查
        max_allowed_len = max(int(len(orig) * 3.2), len(orig) + 48)
        if len(rew) > max_allowed_len:
            return False, f"改写过度膨胀（{len(rew)} > {max_allowed_len} 字）"

        # 2. 问句形态检查（防止 LLM 把改写问题写成直接回答）
        if any(rew.startswith(p) for p in ANSWER_LIKE_PREFIXES):
            return False, "改写偏离问句形态（出现作答口吻）"

        # 3. 实体保真检查（原句明确提到的关键编号/版本/区域不可丢失）
        orig_ents = cls._extract_key_entities(orig)
        rew_ents = set(cls._extract_key_entities(rew))
        missing_ents = [e for e in orig_ents if e not in rew_ents]
        if missing_ents:
            return False, f"丢失原句关键实体: {', '.join(missing_ents)}"

        # 4. 反幻觉溯源检查（改写新引入的实体必须在原句或前序对话中出现过）
        ctx_ents = set(orig_ents) | set(cls._extract_key_entities(history_block))
        hallucinated = [e for e in rew_ents if e not in ctx_ents]
        if hallucinated:
            return False, f"引入上下文未出现的新实体（疑似幻觉）: {', '.join(hallucinated)}"

        return True, ""

    @classmethod
    def _compute_rule_fidelity_score(
        cls,
        original_query: str,
        rewritten_query: str,
        history_block: str,
        mode: str,
    ) -> float:
        """
        第二道防线规则侧：计算规则语义保真分 rule_score ∈ [0.0, 1.0]，用于校准 LLM 自评分数。
        """
        orig = (original_query or "").strip()
        rew = (rewritten_query or "").strip()
        if not rew or rew.lower() == orig.lower():
            return 1.0

        rew_l = rew.lower()
        orig_l = orig.lower()

        # 1. 实义锚点词留存率
        anchors = cls._extract_anchor_tokens(orig)
        if not anchors:
            retention = 0.85
        else:
            hit = sum(1 for tok in anchors if tok in rew_l)
            retention = hit / len(anchors)
            # 极短追问（如“状态呢？”）天然锚点少，给合理下限
            if len(anchors) <= 3:
                retention = max(retention, 0.72)

        score = 0.35 + 0.65 * retention

        # 2. 话题切换误绑定惩罚：原句不含显式代词且长度 >= 10，却强行塞入了历史实体
        has_explicit_pronoun = any(cue in orig_l for cue in EXPLICIT_PRONOUN_CUES)
        orig_ents = set(cls._extract_key_entities(orig))
        rew_ents = set(cls._extract_key_entities(rew))
        injected_hist_ents = rew_ents - orig_ents
        if injected_hist_ents and not has_explicit_pronoun and len(orig) >= 10:
            score -= 0.38

        # 3. 否定/强限定词保真检查
        orig_constraints = [c for c in CONSTRAINT_CUES if c in orig_l]
        if orig_constraints and not any(c in rew_l for c in orig_constraints):
            score -= 0.28

        # 4. 过度改写惩罚（仅跨语模式却大幅改写中文主句）
        if mode == "cross_lingual_rag" and retention < 0.65:
            score -= 0.20

        return round(max(0.0, min(1.0, score)), 3)

    @classmethod
    def _filter_expansion_queries(
        cls,
        extra_queries: List[str],
        original_query: str,
        history_block: str,
    ) -> List[str]:
        """过滤异常或含幻觉实体的扩展检索词"""
        allowed_ents = set(cls._extract_key_entities(original_query)) | set(cls._extract_key_entities(history_block))
        valid: List[str] = []
        for q in extra_queries:
            clean = (q or "").strip()
            if not clean or len(clean) < 2 or len(clean) > 150:
                continue
            if any(clean.startswith(p) for p in ANSWER_LIKE_PREFIXES):
                continue
            q_ents = set(cls._extract_key_entities(clean))
            if any(e not in allowed_ents for e in q_ents):
                continue
            valid.append(clean)
        return valid[:2]

    @staticmethod
    def should_rewrite(
        query: str,
        session: Optional[SessionState],
        needs_rag: bool = False,
        enable_multi_query: bool = True,
    ) -> Tuple[bool, str]:
        """
        三阶门控判定：返回 (是否需要调用改写, 触发模式)。
        触发模式 ∈ {'skip', 'coreference', 'cross_lingual_rag', 'both'}。
        """
        clean_q = (query or "").strip()
        if len(clean_q) < 2:
            return False, "skip"

        clean_l = clean_q.lower()
        pure = re.sub(r"[\s!！?？~。.,，#]+", "", clean_l)
        if pure in GREETING_WORDS or pure in HELP_AND_MENU_WORDS:
            return False, "skip"

        has_history = bool(session is not None and len(session.history) > 0)
        has_coref_cue = any(cue in clean_l for cue in COREF_FOLLOWUP_CUES)
        is_short_followup = has_history and len(clean_q) <= 22

        need_coref = has_history and (has_coref_cue or is_short_followup)
        # 当进入 RAG 且包含中文字符（便于命中库里大量英文技术文档）时触发跨语扩展
        need_rag_expand = bool(needs_rag and enable_multi_query and _HAS_ZH_RE.search(clean_q))

        if need_coref and need_rag_expand:
            return True, "both"
        if need_coref:
            return True, "coreference"
        if need_rag_expand:
            return True, "cross_lingual_rag"
        return False, "skip"

    def rewrite(
        self,
        query: str,
        session: Optional[SessionState] = None,
        needs_rag: bool = False,
        rule_rewritten_query: Optional[str] = None,
        enable_multi_query: bool = True,
        confidence_threshold: float = 0.70,
    ) -> Dict[str, Any]:
        """
        执行三阶门控问题改写 + 置信度防漂移过滤，返回：
        {
            "rewritten_query": str,            # 最终生效的主问句
            "candidate_rewritten_query": str,  # LLM 原始候选改写句（用于拦截审计/前端展示）
            "search_queries": List[str],       # 用于 RAG 检索的查询列表（含主问题 + 0~2 个扩展检索词）
            "query_weights": List[float],      # 与 search_queries 对齐的 RRF 加权系数
            "rewrite_used": bool,              # 是否实际生效了改写或扩展
            "rewrite_reason": str,             # 改写说明标签
            "trigger_mode": str,               # skip / coreference / cross_lingual_rag / both
            "confidence": float,               # 融合置信度 [0.0, 1.0]
            "rewrite_status": str,             # skipped / accepted / soft_fallback / rejected
            "reject_reason": str,              # 若发生回退或拦截，记录具体原因
        }
        """
        clean_q = (query or "").strip()
        base_rewritten = (rule_rewritten_query or clean_q).strip() or clean_q
        threshold = max(0.35, min(0.98, float(confidence_threshold)))
        soft_threshold = max(0.45, round(threshold - 0.25, 2))

        should_run, mode = self.should_rewrite(
            query=clean_q,
            session=session,
            needs_rag=needs_rag,
            enable_multi_query=enable_multi_query,
        )
        if not should_run:
            rule_changed = (base_rewritten != clean_q)
            return {
                "rewritten_query": base_rewritten,
                "candidate_rewritten_query": base_rewritten,
                "search_queries": [base_rewritten],
                "query_weights": [1.0],
                "rewrite_used": rule_changed,
                "rewrite_reason": "规则指代补全" if rule_changed else "",
                "trigger_mode": "skip",
                "confidence": 1.0,
                "rewrite_status": "accepted" if rule_changed else "skipped",
                "reject_reason": "",
            }

        # 构造当前用户自己的最近 3 轮精简历史上下文（绝不读取知识库原文）
        history_lines: List[str] = []
        if session is not None:
            if session.rolling_summary:
                history_lines.append(f"[早期对话摘要] {session.rolling_summary[:260]}")
            if session.active_entity:
                history_lines.append(f"[当前讨论核心实体] {session.active_entity}")
            recent_msgs = session.get_recent_messages(max_turns=3, max_chars_per_msg=280)
            for m in recent_msgs:
                r_tag = "用户" if m["role"] == "user" else "助手"
                history_lines.append(f"{r_tag}: {m['content']}")
        history_block = "\n".join(history_lines) if history_lines else "（无前序对话）"

        llm_res = self._call_llm_rewriter(
            query=clean_q,
            history_block=history_block,
            mode=mode,
            need_multi_query=bool(needs_rag and enable_multi_query),
        )

        if llm_res is None:
            # 离线或 API 异常降级兜底
            fallback_extras = self._fallback_bilingual_queries(base_rewritten) if (needs_rag and enable_multi_query) else []
            s_queries = self._dedup_queries([base_rewritten] + fallback_extras)
            q_weights = [1.0] + [0.75] * (len(s_queries) - 1)
            changed = (base_rewritten != clean_q) or (len(s_queries) > 1)
            return {
                "rewritten_query": base_rewritten,
                "candidate_rewritten_query": base_rewritten,
                "search_queries": s_queries,
                "query_weights": q_weights,
                "rewrite_used": changed,
                "rewrite_reason": "规则兜底改写/跨语扩展" if changed else "",
                "trigger_mode": mode,
                "confidence": 0.85 if changed else 1.0,
                "rewrite_status": "accepted" if changed else "skipped",
                "reject_reason": "",
            }

        candidate_rew = (llm_res.get("rewritten_query") or "").strip() or clean_q
        raw_extras = llm_res.get("search_queries") or []
        if not isinstance(raw_extras, list):
            raw_extras = []
        extra_clean = (
            self._filter_expansion_queries([str(x) for x in raw_extras], clean_q, history_block)
            if (needs_rag and enable_multi_query)
            else []
        )

        is_topic_shift = bool(llm_res.get("is_topic_shift", False))
        try:
            llm_conf = float(llm_res.get("confidence", 0.85))
        except Exception:
            llm_conf = 0.85
        llm_conf = max(0.0, min(1.0, llm_conf))

        llm_reason = (llm_res.get("rewrite_reason") or "").strip()

        # 第一道防线：确定性硬护栏
        guard_ok, guard_reason = self._check_hard_guardrails(clean_q, candidate_rew, history_block)

        # 第二道防线：规则侧语义保真分 + 融合置信度
        rule_score = self._compute_rule_fidelity_score(clean_q, candidate_rew, history_block, mode)
        fused_conf = round(0.55 * llm_conf + 0.45 * rule_score, 3)

        main_changed = (candidate_rew.lower() != clean_q.lower())

        # 若 LLM 明确识别为话题切换，且仍强行修改了主句，强制拦回原句
        if is_topic_shift and main_changed:
            guard_ok = False
            guard_reason = "检测到用户已切换新话题，禁止绑定上轮实体"
            fused_conf = min(fused_conf, 0.35)

        if not guard_ok:
            fused_conf = 0.0 if "话题" not in guard_reason else fused_conf

        # 第三道防线：双阈值分级决策
        # 情况 1：主句未改写（例如纯中文跨语扩展），只需校验扩展词与整体置信度
        if not main_changed:
            if extra_clean and fused_conf >= soft_threshold:
                s_queries = self._dedup_queries([clean_q] + extra_clean)
                q_weights = [1.0] + [0.75] * (len(s_queries) - 1)
                return {
                    "rewritten_query": clean_q,
                    "candidate_rewritten_query": candidate_rew,
                    "search_queries": s_queries,
                    "query_weights": q_weights,
                    "rewrite_used": len(s_queries) > 1,
                    "rewrite_reason": llm_reason or "跨语检索词扩展",
                    "trigger_mode": mode,
                    "confidence": fused_conf,
                    "rewrite_status": "accepted" if len(s_queries) > 1 else "skipped",
                    "reject_reason": "",
                }
            # 置信度过低或无有效扩展词 -> 跳过扩展
            return {
                "rewritten_query": clean_q,
                "candidate_rewritten_query": candidate_rew,
                "search_queries": [clean_q],
                "query_weights": [1.0],
                "rewrite_used": False,
                "rewrite_reason": "",
                "trigger_mode": mode,
                "confidence": fused_conf,
                "rewrite_status": "rejected" if extra_clean else "skipped",
                "reject_reason": f"扩展置信度 {fused_conf:.2f} 低于下限 {soft_threshold:.2f}" if extra_clean else "",
            }

        # 情况 2：主句发生了改写，严格执行 硬拦截 / 软回退 / 高置信采纳
        if not guard_ok or fused_conf < soft_threshold:
            reject_msg = guard_reason or f"融合置信度 {fused_conf:.2f} 低于安全下限 {soft_threshold:.2f}"
            # 硬拦截回退原问题（若为中文跨语检索仍可保留本地安全兜底词，避免检索完全失明）
            s_queries = [clean_q]
            q_weights = [1.0]
            if needs_rag and enable_multi_query and _HAS_ZH_RE.search(clean_q):
                safe_fb = self._fallback_bilingual_queries(clean_q)
                if safe_fb:
                    s_queries = self._dedup_queries([clean_q] + safe_fb)
                    q_weights = [1.0] + [0.65] * (len(s_queries) - 1)
            return {
                "rewritten_query": clean_q,
                "candidate_rewritten_query": candidate_rew,
                "search_queries": s_queries,
                "query_weights": q_weights,
                "rewrite_used": False,
                "rewrite_reason": f"改写已拦截并回退原问句（{reject_msg}）",
                "trigger_mode": mode,
                "confidence": fused_conf,
                "rewrite_status": "rejected",
                "reject_reason": reject_msg,
            }

        if fused_conf < threshold:
            # 中置信度软回退：主句保留原问题防跑偏，改写句仅作低权重(0.5x)补充检索词
            s_queries = self._dedup_queries([clean_q, candidate_rew] + extra_clean[:1])
            q_weights = [1.0] + [0.50] * (len(s_queries) - 1)
            soft_msg = f"置信度 {fused_conf:.2f} 低于阈值 {threshold:.2f}，保留原问句并以 0.5x 权重辅助检索"
            return {
                "rewritten_query": clean_q,
                "candidate_rewritten_query": candidate_rew,
                "search_queries": s_queries,
                "query_weights": q_weights,
                "rewrite_used": len(s_queries) > 1,
                "rewrite_reason": soft_msg,
                "trigger_mode": mode,
                "confidence": fused_conf,
                "rewrite_status": "soft_fallback",
                "reject_reason": soft_msg,
            }

        # 高置信度：正式采纳改写主句 + 跨语扩展词
        s_queries = self._dedup_queries([candidate_rew] + extra_clean)
        q_weights = [1.0] + [0.75] * (len(s_queries) - 1)
        reason = llm_reason
        if not reason:
            if mode == "both":
                reason = "指代消解 + 跨语检索扩展"
            elif mode == "coreference":
                reason = "多轮指代消解"
            else:
                reason = "跨语检索词扩展"

        return {
            "rewritten_query": candidate_rew,
            "candidate_rewritten_query": candidate_rew,
            "search_queries": s_queries,
            "query_weights": q_weights,
            "rewrite_used": True,
            "rewrite_reason": reason,
            "trigger_mode": mode,
            "confidence": fused_conf,
            "rewrite_status": "accepted",
            "reject_reason": "",
        }

    def _call_llm_rewriter(
        self,
        query: str,
        history_block: str,
        mode: str,
        need_multi_query: bool,
    ) -> Optional[Dict[str, Any]]:
        expand_instr = (
            '- "search_queries": 数组，提供 1~2 个等价的英文或专业术语检索查询（因为企业知识库内包含大量英文技术文档与工单；若原句已是纯英文清晰问题可返回空数组 []）。\n'
            if need_multi_query
            else '- "search_queries": 空数组 []。\n'
        )
        prompt = (
            "你是企业智能知识库的【用户问题改写器】，职责是结合前序对话，将用户最新提问改写为语义完整、无指代歧义的独立问句。\n"
            "要求：\n"
            "1. 严格保留用户原意与所有编号/版本号/限定词，绝对不要自行编造前序对话中未出现过的实体、数字或结论；\n"
            "2. 若当前提问包含代词（它、这个、刚才说的）或明显省略了主语，请从【前序对话】中补全具体对象；\n"
            "3. 若用户已明显切换到了全新话题，请将 is_topic_shift 设为 true，且 rewritten_query 严格保持用户原句，切勿强行绑定上轮实体；\n"
            "4. 请客观评估你对本次改写完全忠实于用户原意的置信度 confidence（0.0~1.0，越确定语义无偏移越接近 1.0）；\n"
            "5. 严格输出合法 JSON 对象，不要输出任何额外解释文字。格式：\n"
            "{\n"
            '  "rewritten_query": "补全后的完整问句（保持用户原语言）",\n'
            f"{expand_instr}"
            '  "is_topic_shift": false,\n'
            '  "confidence": 0.92,\n'
            '  "rewrite_reason": "简短说明（如：指代消解 / 省略补全 / 跨语扩展 / 无需改写）"\n'
            "}\n\n"
            f"【前序对话】：\n{history_block}\n\n"
            f"【用户最新提问】：\n{query}"
        )

        try:
            resp = _gen_mod.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是精准的搜索问题改写引擎，只输出严格 JSON 对象。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=260,
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _fallback_bilingual_queries(query: str) -> List[str]:
        """离线兜底：将中文企业高频词映射为英文等价检索词"""
        q = (query or "").strip()
        en_tokens = re.findall(r"[A-Za-z0-9_\-]{2,}", q)
        mapped = []
        for zh, en in _BILINGUAL_FALLBACK_MAP:
            if zh in q:
                mapped.append(en)
        if not mapped:
            return []
        combined = " ".join(dict.fromkeys(en_tokens + mapped))
        return [combined] if combined and combined.lower() != q.lower() else []

    @staticmethod
    def _dedup_queries(queries: List[str]) -> List[str]:
        seen = set()
        out = []
        for q in queries:
            clean = (q or "").strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                out.append(clean)
        return out[:3]
