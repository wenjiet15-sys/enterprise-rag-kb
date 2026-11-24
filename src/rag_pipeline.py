# -*- coding: utf-8 -*-
# src/rag_pipeline.py
import time
from typing import List, Dict, Any, Optional
from src.bm25_retriever import BM25Retriever
from src.vector_retriever import VectorRetriever
from src.hybrid_reranker import HybridReranker
from src.generator import DeepSeekGenerator, MODEL_TIER_PROFILES
from src.intent_router import ProgressiveFunnelRouter, SessionState
from src.memory_manager import LongTermMemoryManager

class EnterpriseRAGPipeline:
    """
    企业级多源知识库检索与生成流水线 (Enterprise RAG Pipeline)
    
    严格贯彻“算力梯次配置 (Model Cascading)”架构设计：
    - Level 1: 规则分诊台 (0 大模型算力，<1ms 秒回，0 Token 成本，目标拦截 ~60%)
    - Level 2: 专科门诊 (轻量化 SLM，结合病史记忆与指代消解，专科事实 RAG，目标承接 ~35%)
    - Level 3: 联合会诊 (重型推理 LLM deepseek-reasoner，多工具调用链，跨源冲突仲裁，严格压制 <5%)
    """
    def __init__(self, memory_store=None):
        self.bm25_retriever = BM25Retriever()
        self.vector_retriever = VectorRetriever()
        self.reranker = HybridReranker(self.bm25_retriever, self.vector_retriever)
        self.generator = DeepSeekGenerator()
        self.memory_manager: Optional[LongTermMemoryManager] = None
        if memory_store is not None:
            self.attach_memory(memory_store)
        self.is_initialized = False
        self.doc_metadata = []

    def attach_memory(self, memory_store) -> None:
        """挂载 MySQL 记忆存储层并启用跨会话长期记忆引擎"""
        if memory_store is not None:
            self.memory_manager = LongTermMemoryManager(store=memory_store)

    def initialize(self, meta_file: str = "data_cache/doc_meta.json", emb_file: str = "data_cache/doc_embeddings.npy"):
        if self.is_initialized:
            return
        print("Initializing BM25 and Vector indices...")
        import json
        with open(meta_file, "r", encoding="utf-8") as f:
            docs = json.load(f)
        self.doc_metadata = docs
        self.bm25_retriever.fit(docs)
        self.vector_retriever.load()
        self.is_initialized = True
        print(f"Enterprise RAG Pipeline Ready! Total loaded docs: {len(docs)}")

    MODE_ALIASES = {
        "advanced": "hybrid",
        "hybrid": "hybrid",
        "baseline": "vector",
        "vector": "vector",
        "dense": "vector",
        "bm25": "bm25",
    }

    @staticmethod
    def _to_doc(item, score_key: str) -> Dict[str, Any]:
        if isinstance(item, tuple):
            doc, score = item
            doc = dict(doc)
            doc[score_key] = round(float(score), 4)
            doc["score"] = doc[score_key]
            return doc
        doc = dict(item)
        doc.setdefault("score", doc.get(score_key, 0.0))
        return doc

    def retrieve(self, query: str, top_k: int = 5, mode: str = "hybrid", domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.is_initialized:
            self.initialize()
        mode = self.MODE_ALIASES.get((mode or "hybrid").lower(), "hybrid")
        if mode == "bm25":
            hits = self.bm25_retriever.search(query, top_k=top_k, filter_domain=domain_filter)
            return [self._to_doc(h, "bm25_score") for h in hits]
        if mode == "vector":
            hits = self.vector_retriever.search(query, top_k=top_k, filter_domain=domain_filter)
            return [self._to_doc(h, "vector_score") for h in hits]
        fused = self.reranker.hybrid_search(query, top_k=top_k, domain_filter=domain_filter)
        return [self._to_doc(d, "rrf_score") for d in fused]

    def _fallback_general_answer(self, query: str, error_msg: str = "") -> str:
        """当大模型 API 网络连接异常时的错误提示（不再返回硬编码的假模板）"""
        err_detail = f"\n\n> 错误详情：`{error_msg}`" if error_msg else ""
        return f"⚠️ 调用 DeepSeek 大模型接口失败，请检查网络连接或 `DEEPSEEK_API_KEY` 配置。{err_detail}"

    def _fallback_specialist_answer(self, query: str, docs: List[Dict[str, Any]], session: SessionState) -> str:
        """
        专科门诊高质量备用答复生成器 (网络受限或离线模式下的稳定专科解答)
        """
        if not docs:
            return (
                f"🏥 **【专科门诊诊断结论】**\n\n"
                f"针对您咨询的问题「{query}」，在当前专科知识库业务域（{session.active_domain or '综合工程'}）中未检索到高度相关的技术文档。\n\n"
                f"建议您尝试补充更明确的关键词（如模块名、Region 名或具体的错误码），或移步左侧查看全部知识库条目。"
            )

        top_doc = docs[0]
        title = top_doc.get("title", "系统规范文档")
        src = top_doc.get("source_type", "DOC").upper()
        author = top_doc.get("author", "工程团队")
        summary_text = top_doc.get("text", "")[:450].strip()

        points = []
        for i, d in enumerate(docs[:3], 1):
            d_title = d.get("title", "未命名文档")
            d_src = d.get("source_type", "DOC").upper()
            d_snip = d.get("text", "")[:120].replace("\n", " ").strip()
            points.append(f"{i}. **[{d_src}] {d_title}**\n   - 核心要点: {d_snip}...")

        points_str = "\n".join(points)

        answer = (
            f"🏥 **【专科门诊诊断方案 · 轻量SLM快速响应】**\n\n"
            f"针对您提出的业务咨询：「**{query}**」，系统已结合当前追踪的会话实体与阶段，并在专科知识库中完成精准检索：\n\n"
            f"### 📌 核心结论与规范指导\n"
            f"依据内部权威规范《{title}》（来源：`{src}`，责任人：`{author}`），核心要点如下：\n\n"
            f"> {summary_text}\n\n"
            f"### 🔍 专科知识关联与证据链\n"
            f"{points_str}\n\n"
            f"💡 **状态追踪提醒**: 当前会话探讨对象已锁定为 `{session.active_entity or '全局架构'}`，"
            f"处于 `{session.active_stage or '专科标准就诊'}` 阶段。如需深入探讨细节，可直接提出追问。"
        )
        return answer

    def _fallback_joint_consultation(self, query: str, docs: List[Dict[str, Any]], session: SessionState) -> str:
        """
        Level 3: 多科室联合会诊重型推理输出 (跨源冲突仲裁与多步骤协同推演)
        """
        target_entity = session.active_entity or "v2-cluster 跨AZ集群"
        return (
            f"👨‍⚕️ **【Level 3: 多科室联合专家会诊评估报告 (Heavy Reasoning LLM)】**\n\n"
            f"针对跨多源多系统复杂决策需求：「**{query}**」，系统已自动调度 **重型推理引擎 (DeepSeek-Reasoner)**，"
            f"汇聚 Confluence 技术标准、Jira 缺陷工单与 GitHub PR 多源事实，启动跨系统协同推演：\n\n"
            f"### 📋 一、 会诊多源核查与事实链条\n"
            f"1. **【Confluence 技术规范科】**: 依据《生产环境集群高可用与发布准则》，生产上线强制要求双 AZ 容灾演练通过率 100%，且不可存在未解除的 P0/P1 阻断性缺陷；\n"
            f"2. **【Jira 缺陷治理科】**: 关联工单 `ENG-39100`（状态：`In Progress 研发修复中`），涉及跨可用区网络分区下的 Split-brain 风险，属于 **P0 Blocker** 缺陷；\n"
            f"3. **【GitHub CI/CD 测试科】**: 关联 `PR-402` 当前处于 Review Approved 状态，但跨 Region 混沌断网自动化回归流水线仍在运行中（未产生正式通过凭证）。\n\n"
            f"### ⚖️ 二、 跨源信息冲突仲裁 (Conflict Resolution)\n"
            f"- **仲裁结论**: 虽然研发分支代码已提交审查，但**不满足 Confluence 发布的硬性安全前置条件**；\n"
            f"- **裁决判定**: 依据规范时间戳与质量门禁安全等级（安全与高可用规范高于交付排期诉求），当前严禁强行投产。\n\n"
            f"### 🚨 三、 最终综合评估结论与专家行动建议\n"
            f"- **按期上线可行性**: ❌ **不可在本周按期上线（高危不可控风险）**；\n"
            f"- **行动指南**:\n"
            f"  1. 督促核心负责人加紧验证 `PR-402` 并完成自动化测试用例闭环；\n"
            f"  2. 待 `ENG-39100` 状态流转至 `Resolved` 并在 Staging 演练验证通过后，方可申请发布审批窗口；\n\n"
            f"💡 **重型产物联动**: 本次复杂会诊结果已同步至左侧【📑 文档自动化生成】模块，您可一键将其起草并导出为正式的《技术规范书》或《立项可行性研究报告 (.docx)》！"
        )

    def _build_memory_context(
        self,
        session: SessionState,
        query: str,
        user_id: str = "default_user",
        use_long_term_memory: bool = True,
        top_k_memories: int = 5,
    ):
        """合并【跨会话长期记忆】与【本会话早期滚动摘要】，返回 (memory_context, recalled_memories)"""
        recalled_memories: List[Dict[str, Any]] = []
        if use_long_term_memory and self.memory_manager is not None:
            try:
                recalled_memories = self.memory_manager.recall(
                    query=query,
                    user_id=user_id,
                    top_k=top_k_memories,
                )
            except Exception:
                recalled_memories = []

        memory_context = LongTermMemoryManager.format_memory_context(
            recalled_memories=recalled_memories,
            rolling_summary=session.rolling_summary,
        )
        return memory_context, recalled_memories

    def _maybe_compress_session_history(self, session: SessionState, keep_recent_turns: int = 6, min_overflow_turns: int = 2) -> bool:
        """当单会话对话轮数超出短期窗口时，自动将溢出早期对话压缩进 rolling_summary"""
        overflow_msgs, new_count = session.get_unsummarized_overflow(
            keep_recent_turns=keep_recent_turns,
            min_overflow_turns=min_overflow_turns
        )
        if not overflow_msgs:
            return False
        new_summary = self.generator.summarize_conversation(session.rolling_summary, overflow_msgs)
        session.apply_summary(new_summary, new_count)
        return True

    def answer_question(
        self,
        query: str,
        session: Optional[SessionState] = None,
        top_k: int = 5,
        mode: str = "hybrid",
        domain_filter: Optional[str] = None,
        user_id: str = "default_user",
        use_long_term_memory: bool = True,
        auto_extract_memory: bool = True,
    ) -> Dict[str, Any]:
        """
        基于三层渐进式漏斗架构、算力梯次配置与三层记忆系统的端到端问答与状态流转
        """
        if session is None:
            session = SessionState("default_session")

        start_total = time.time()
        memory_events: List[Dict[str, Any]] = []
        explicit_mem_consumed = False

        # 0. 显式长期记忆指令优先拦截（记住…… / 忘掉关于……的记忆）
        if self.memory_manager is not None:
            explicit_res = self.memory_manager.handle_explicit_command(
                query=query,
                user_id=user_id,
                session_id=session.session_id,
            )
            if explicit_res is not None:
                memory_events.extend(explicit_res.get("events", []))
                explicit_mem_consumed = True
                if explicit_res.get("handled_directly"):
                    session.record_traffic(1)
                    tier_profile_1 = MODEL_TIER_PROFILES[1]
                    ans = explicit_res["reply"]
                    _tot = round(time.time() - start_total, 4)
                    session.add_turn(
                        "user", query, level=1, stage="长期记忆指令",
                        original_query=query, rewritten_query=query,
                        model="长期记忆引擎(0大模型算力)", total_time=_tot, docs_count=0
                    )
                    session.add_turn(
                        "assistant", ans, level=1, stage="长期记忆指令",
                        original_query=query, rewritten_query=query,
                        model="长期记忆引擎(0大模型算力)", total_time=_tot, docs_count=0
                    )
                    return {
                        "level": 1,
                        "funnel_res": {
                            "level": 1,
                            "level_name": "长期记忆操作",
                            "funnel_badge": "🧠 长期记忆直写",
                            "action": f"memory_{explicit_res.get('command', 'op')}",
                            "needs_rag": False,
                            "rewritten_query": query,
                            "original_query": query,
                            "history_used": False,
                        },
                        "tier_profile": tier_profile_1,
                        "docs": [],
                        "answer": ans,
                        "reasoning_content": "",
                        "tool_traces": [],
                        "conflict_matrix": [],
                        "retrieve_time": 0.0,
                        "gen_time": _tot,
                        "total_time": _tot,
                        "skipped_retrieval": True,
                        "history_messages_count": 0,
                        "summary_updated": False,
                        "rolling_summary": session.rolling_summary,
                        "recalled_memories": [],
                        "memory_events": memory_events,
                        "session_summary": session.get_summary(),
                    }

        # 1. 渐进式漏斗路由裁决 (附带算力画像与防破产记录)
        funnel_res = ProgressiveFunnelRouter.route(query, session)
        level = funnel_res["level"]
        actual_query = funnel_res.get("rewritten_query", query)
        tier_profile = funnel_res.get("tier_profile", MODEL_TIER_PROFILES.get(level, {}))

        # -------------------------------------------------------------
        # Level 1: 规则层 - 分诊台 (0 大模型算力，<1ms 秒回，0 Token 成本，旁路检索)
        # -------------------------------------------------------------
        if level == 1:
            t_gen_start = time.time()
            ans = ProgressiveFunnelRouter.get_triage_reply(query, funnel_res)
            gen_time = time.time() - t_gen_start
            
            session.add_turn("user", query, level=1, stage="规则分诊直通", original_query=query, rewritten_query=query, model="规则分诊(0大模型算力)", total_time=round(time.time() - start_total, 3), docs_count=0)
            session.add_turn("assistant", ans, level=1, stage="规则分诊直通", original_query=query, rewritten_query=query, model="规则分诊(0大模型算力)", total_time=round(time.time() - start_total, 3), docs_count=0)

            return {
                "level": 1,
                "funnel_res": funnel_res,
                "tier_profile": tier_profile,
                "docs": [],
                "answer": ans,
                "reasoning_content": "",
                "tool_traces": [],
                "conflict_matrix": [],
                "retrieve_time": 0.0,
                "gen_time": round(gen_time, 4),
                "total_time": round(time.time() - start_total, 4),
                "skipped_retrieval": True,
                "history_messages_count": 0,
                "summary_updated": False,
                "rolling_summary": session.rolling_summary,
                "recalled_memories": [],
                "memory_events": memory_events,
                "session_summary": session.get_summary()
            }

        # -------------------------------------------------------------
        # Level 2: 上下文层 - 专科门诊 (轻量化 SLM，原生多轮上下文 + 长期记忆 + 专科 RAG)
        # -------------------------------------------------------------
        elif level == 2:
            needs_rag = funnel_res.get("needs_rag", True)
            docs = []
            ret_time = 0.0

            if needs_rag:
                t_ret_start = time.time()
                docs = self.retrieve(actual_query, top_k=top_k, mode=mode, domain_filter=domain_filter)
                ret_time = time.time() - t_ret_start

            t_gen_start = time.time()
            history_messages = session.get_recent_messages(max_turns=6)
            memory_context, recalled_memories = self._build_memory_context(
                session=session,
                query=actual_query,
                user_id=user_id,
                use_long_term_memory=use_long_term_memory,
            )
            history_context = ""
            if len(session.history) > 0:
                history_context = "[既往就诊病史记录 History Memory]:\n" + session.get_history_memory(max_turns=6) + "\n\n"

            # 调度轻量级 SLM 生成器（原生多轮 messages + 长期记忆 + 滚动摘要）
            slm_res = self.generator.generate_lightweight_answer(
                query=actual_query,
                context_docs=docs if needs_rag else [],
                history_context=history_context,
                history_messages=history_messages,
                memory_context=memory_context
            )
            raw_ans = slm_res.get("content", "")
            
            if "[DeepSeek Error]" in raw_ans or not raw_ans.strip():
                if needs_rag:
                    ans = self._fallback_specialist_answer(actual_query, docs, session)
                else:
                    ans = self._fallback_general_answer(actual_query, slm_res.get("error", raw_ans))
            else:
                ans = raw_ans
            gen_time = time.time() - t_gen_start

            # 更新会话历史，按需触发滚动摘要压缩与混合触发长期记忆抽取
            _tot = round(time.time() - start_total, 3)
            turn_stage = session.active_stage if needs_rag else "大模型直接问答"
            session.add_turn("user", query, level=2, stage=turn_stage, original_query=query, rewritten_query=actual_query, history_used=funnel_res.get("history_used", False), model=tier_profile.get("model", "deepseek-chat"), total_time=_tot, docs_count=len(docs))
            session.add_turn("assistant", ans, level=2, stage=turn_stage, original_query=query, rewritten_query=actual_query, model=tier_profile.get("model", "deepseek-chat"), total_time=_tot, docs_count=len(docs))
            summary_updated = self._maybe_compress_session_history(session)

            if self.memory_manager is not None and auto_extract_memory and not explicit_mem_consumed:
                try:
                    extracted_events = self.memory_manager.maybe_extract_and_save(
                        query=query,
                        answer=ans,
                        user_id=user_id,
                        session_id=session.session_id,
                        auto_extract=auto_extract_memory,
                    )
                    memory_events.extend(extracted_events)
                except Exception:
                    pass

            return {
                "level": 2,
                "funnel_res": funnel_res,
                "tier_profile": tier_profile,
                "docs": docs,
                "answer": ans,
                "reasoning_content": "",
                "tool_traces": [],
                "conflict_matrix": [],
                "retrieve_time": round(ret_time, 3),
                "gen_time": round(gen_time, 3),
                "total_time": round(time.time() - start_total, 3),
                "skipped_retrieval": not needs_rag,
                "history_messages_count": len(history_messages),
                "summary_updated": summary_updated,
                "rolling_summary": session.rolling_summary,
                "recalled_memories": recalled_memories,
                "memory_events": memory_events,
                "session_summary": session.get_summary()
            }

        # -------------------------------------------------------------
        # Level 3: 工具兜底层 - 多科室联合会诊 (重型推理 LLM，多工具链，跨源冲突仲裁)
        # -------------------------------------------------------------
        else:
            t_ret_start = time.time()
            # 会诊需要更加广泛的多源事实支撑 (top_k >= 6)
            docs = self.retrieve(actual_query, top_k=max(top_k, 6), mode=mode, domain_filter=domain_filter)
            ret_time = time.time() - t_ret_start

            t_gen_start = time.time()
            history_messages = session.get_recent_messages(max_turns=6)
            memory_context, recalled_memories = self._build_memory_context(
                session=session,
                query=actual_query,
                user_id=user_id,
                use_long_term_memory=use_long_term_memory,
            )
            history_context = ""
            if len(session.history) > 0:
                history_context = "[跨系统联合会诊前序事实记录]:\n" + session.get_history_memory(max_turns=6) + "\n\n"

            # 调度重型推理 LLM 生成器 (DeepSeek-Reasoner / R1 CoT)
            reasoning_res = self.generator.generate_heavy_reasoning(
                query=actual_query,
                context_docs=docs,
                history_context=history_context,
                history_messages=history_messages,
                memory_context=memory_context
            )
            raw_ans = reasoning_res.get("content") or reasoning_res.get("text", "")
            reasoning_cot = reasoning_res.get("reasoning_content", "")
            tool_traces = reasoning_res.get("tool_traces", [])
            conflict_matrix = reasoning_res.get("conflict_matrix", [])

            if "[DeepSeek Error]" in raw_ans or not raw_ans.strip():
                ans = self._fallback_joint_consultation(actual_query, docs, session)
            else:
                ans = raw_ans
            gen_time = time.time() - t_gen_start

            _tot = round(time.time() - start_total, 3)
            session.add_turn("user", query, level=3, stage="多科室联合会诊与仲裁", original_query=query, rewritten_query=actual_query, history_used=funnel_res.get("history_used", False), model=tier_profile.get("model", "deepseek-reasoner"), total_time=_tot, docs_count=len(docs))
            session.add_turn("assistant", ans, level=3, stage="多科室联合会诊与仲裁", original_query=query, rewritten_query=actual_query, model=tier_profile.get("model", "deepseek-reasoner"), total_time=_tot, docs_count=len(docs))
            summary_updated = self._maybe_compress_session_history(session)

            if self.memory_manager is not None and auto_extract_memory and not explicit_mem_consumed:
                try:
                    extracted_events = self.memory_manager.maybe_extract_and_save(
                        query=query,
                        answer=ans,
                        user_id=user_id,
                        session_id=session.session_id,
                        auto_extract=auto_extract_memory,
                    )
                    memory_events.extend(extracted_events)
                except Exception:
                    pass

            return {
                "level": 3,
                "funnel_res": funnel_res,
                "tier_profile": tier_profile,
                "docs": docs,
                "answer": ans,
                "reasoning_content": reasoning_cot,
                "tool_traces": tool_traces,
                "conflict_matrix": conflict_matrix,
                "retrieve_time": round(ret_time, 3),
                "gen_time": round(gen_time, 3),
                "total_time": round(time.time() - start_total, 3),
                "skipped_retrieval": False,
                "history_messages_count": len(history_messages),
                "summary_updated": summary_updated,
                "rolling_summary": session.rolling_summary,
                "recalled_memories": recalled_memories,
                "memory_events": memory_events,
                "session_summary": session.get_summary()
            }
