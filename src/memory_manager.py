# -*- coding: utf-8 -*-
# src/memory_manager.py
import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple

import src.generator as _gen_mod
from src.memory_store import MemoryStore


MEMORY_CATEGORY_LABELS = {
    "user_profile": "个人画像",
    "preference": "回答偏好",
    "project_focus": "关注项目",
    "knowledge_fact": "业务结论",
    "custom": "自定义记忆",
}

# 显式“记住”指令匹配
EXPLICIT_REMEMBER_PATTERNS = [
    re.compile(r"^(?:请你?|帮我)?(?:以后)?(?:务必)?(?:记住|记一下|记下来|牢记)[：:，,\s]*(.+)$", re.IGNORECASE),
    re.compile(r"^(?:请)?把(?:下面|这个|以下)?(?:内容|信息)?记住[：:，,\s]*(.+)$", re.IGNORECASE),
]

# 显式“遗忘/删除记忆”指令匹配
EXPLICIT_FORGET_PATTERNS = [
    re.compile(r"^(?:请你?|帮我)?(?:忘掉|忘记|删除|删掉|移除|清除)(?:掉)?(?:关于)?(.+?)(?:的)?记忆[。！!？?]*$", re.IGNORECASE),
    re.compile(r"^(?:不要|别)再?记(?:住|着)?(.+?)(?:了)?[。！!？?]*$", re.IGNORECASE),
]

# 混合触发自动抽取（选项 A）高信号触发词：低成本过滤，只对有长期记忆价值的句子触发 LLM 抽取
HIGH_SIGNAL_PATTERNS = [
    re.compile(r"(我叫|我的名字是|叫我|称呼我|我是做|我是负责|我是一名|我是一个|我在.+团队|我在.+部门)"),
    re.compile(r"(我负责|我主要负责|我平时用|我主要用|我擅长|我用.+开发|我的技术栈)"),
    re.compile(r"(我喜欢|我偏好|我习惯|以后请|以后都|以后回答|下次回答|回复请|回答请|尽量简短|精简一点|给代码示例|用中文回答|用英文回答|用要点)"),
    re.compile(r"(长期关注|一直跟进|我们团队|我们项目|我们的集群|我们的系统|重点关注|备忘一下)"),
]

# 判断复合句是否包含额外业务问题（例如“记住我是SRE，顺便帮我查一下ENG-39100”）
COMPOUND_QUESTION_CLUES = [
    "？", "?", "帮我查", "查询", "顺便", "请问", "怎么", "如何", "为什么", "是什么", "有哪些", "能不能", "能否", "评估", "分析", "解释"
]


def _tokenize_mixed(text: str) -> List[str]:
    """中英文混合轻量分词：英文单词/编号 + 中文单字与双字词片"""
    if not text:
        return []
    lowered = text.lower()
    en_tokens = re.findall(r"[a-z0-9][a-z0-9_\-]*", lowered)
    zh_chunks = re.findall(r"[\u4e00-\u9fa5]+", lowered)
    zh_tokens: List[str] = []
    for chunk in zh_chunks:
        if len(chunk) == 1:
            zh_tokens.append(chunk)
        else:
            zh_tokens.extend(list(chunk))
            for i in range(len(chunk) - 1):
                zh_tokens.append(chunk[i : i + 2])
    return en_tokens + zh_tokens


def _text_similarity(a: str, b: str) -> float:
    """计算两段中英文文本的 Jaccard + 子串相似度，用于本地去重与相关性打分"""
    sa = (a or "").strip().lower()
    sb = (b or "").strip().lower()
    if not sa or not sb:
        return 0.0
    if sa == sb:
        return 1.0
    if sa in sb or sb in sa:
        return min(0.92, max(len(sa), len(sb)) / (len(sa) + len(sb)) + 0.45)
    ta = set(_tokenize_mixed(sa))
    tb = set(_tokenize_mixed(sb))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0


def _infer_category(content: str) -> str:
    c = (content or "").lower()
    if any(w in c for w in ["以后", "回答", "回复", "偏好", "喜欢", "习惯", "风格", "精简", "简短", "详细", "代码", "列表", "要点", "语言"]):
        return "preference"
    if any(w in c for w in ["我叫", "名字", "称呼", "我是", "职位", "架构师", "开发", "工程师", "经理", "sre", "运维", "团队", "部门"]):
        return "user_profile"
    if any(w in c for w in ["集群", "cluster", "eng-", "jira", "pr-", "项目", "系统", "美西", "us-west", "服务", "模块"]):
        return "project_focus"
    return "custom"


def _extract_keywords(content: str) -> str:
    en = re.findall(r"[A-Za-z0-9_\-]{2,}", content or "")
    zh = re.findall(r"[\u4e00-\u9fa5]{2,6}", content or "")
    seen = []
    for tok in en + zh:
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= 8:
            break
    return " ".join(seen)


class LongTermMemoryManager:
    """
    跨会话长期记忆引擎：
    1. 显式指令处理（记住…… / 忘掉关于……的记忆）
    2. 混合触发自动抽取（高信号对话 → 轻量 deepseek-chat JSON 抽取 → 去重/冲突更新）
    3. 混合召回与注入（核心画像常驻 + 相关性 Top-K 打分 + 命中计数更新）
    """

    def __init__(self, store: MemoryStore, model_name: str = "deepseek-chat"):
        self.store = store
        self.model_name = model_name

    # =================================================================
    # 1. 显式指令识别与执行（记住 / 遗忘）
    # =================================================================

    def handle_explicit_command(
        self,
        query: str,
        user_id: str = "default_user",
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        检查用户输入是否命中显式“记住”或“遗忘”指令：
        - 若命中“遗忘”：删除匹配记忆，返回 handled_directly=True 及友好回复；
        - 若命中“记住”：立即去重落库；若为纯记住指令则返回 handled_directly=True，
          若句子后半段还附带其他提问（复合句）则返回 handled_directly=False 以便继续问答。
        """
        clean_q = (query or "").strip()
        if not clean_q:
            return None

        # A. 显式遗忘/删除
        for pat in EXPLICIT_FORGET_PATTERNS:
            m = pat.match(clean_q)
            if m:
                target = m.group(1).strip(" ：:，,。！？?\"'“”")
                if not target or target in ("所有", "全部", "一切"):
                    deleted_cnt = self.store.clear_memories(user_id=user_id)
                    return {
                        "command": "forget_all",
                        "handled_directly": True,
                        "events": [{"action": "cleared", "count": deleted_cnt}],
                        "reply": f"🧹 已为你清空全部 **{deleted_cnt}** 条跨会话长期记忆。",
                    }
                deleted_items = self._forget_matching_memories(target, user_id=user_id)
                if deleted_items:
                    lines = "\n".join([f"- ~~{it['content']}~~" for it in deleted_items])
                    return {
                        "command": "forget",
                        "handled_directly": True,
                        "events": [
                            {
                                "action": "deleted",
                                "id": it["id"],
                                "category": it["category"],
                                "content": it["content"],
                            }
                            for it in deleted_items
                        ],
                        "reply": f"🧹 已为你遗忘并删除 **{len(deleted_items)}** 条相关长期记忆：\n\n{lines}",
                    }
                return {
                    "command": "forget",
                    "handled_directly": True,
                    "events": [],
                    "reply": f"🔍 在长期记忆库中未找到与「**{target}**」相关的记录，无需删除。",
                }

        # B. 显式记住
        for pat in EXPLICIT_REMEMBER_PATTERNS:
            m = pat.match(clean_q)
            if m:
                raw_payload = m.group(1).strip()
                if not raw_payload:
                    return None

                # 判断是否复合句（前段要求记住，后段另有业务提问）
                is_compound = False
                mem_text = raw_payload
                for sep in ["，顺便", ",顺便", "，另外", "，帮我", "，请帮我", "。顺便", "。帮我", "，然后"]:
                    if sep in raw_payload:
                        left, right = raw_payload.split(sep, 1)
                        if left.strip() and right.strip():
                            mem_text = left.strip(" ，。;；")
                            is_compound = True
                            break
                if not is_compound and any(clue in raw_payload for clue in COMPOUND_QUESTION_CLUES):
                    # 若整句像问句且带逗号，取前半句作为记忆、继续问答
                    if "，" in raw_payload or "," in raw_payload:
                        first_part = re.split(r"[，,]", raw_payload, maxsplit=1)[0].strip()
                        if len(first_part) >= 4:
                            mem_text = first_part
                            is_compound = True

                cat = _infer_category(mem_text)
                event = self._upsert_single_memory(
                    content=mem_text,
                    category=cat,
                    importance=5,
                    user_id=user_id,
                    session_id=session_id,
                )
                events = [event] if event else []
                cat_label = MEMORY_CATEGORY_LABELS.get(cat, "自定义记忆")
                action_verb = "更新" if (event and event["action"] == "updated") else "记住"
                reply = (
                    f"🧠 已为你{action_verb}以下长期记忆（分类：`{cat_label}`，已持久化至 MySQL）：\n\n"
                    f"> {mem_text}\n\n"
                    "后续跨会话对话中我会自动参考这条记忆；如需修改或遗忘，可直接对我说「忘掉关于……的记忆」或在记忆管理中操作。"
                )
                return {
                    "command": "remember",
                    "handled_directly": not is_compound,
                    "events": events,
                    "reply": reply,
                }

        return None

    def _forget_matching_memories(self, target: str, user_id: str) -> List[Dict[str, Any]]:
        all_mems = self.store.list_memories(user_id=user_id, limit=200)
        target_l = target.lower()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for mem in all_mems:
            text_blob = f"{mem.get('content', '')} {mem.get('keywords', '')}".lower()
            if target_l in text_blob:
                scored.append((1.0, mem))
            else:
                sim = _text_similarity(target, text_blob)
                if sim >= 0.25:
                    scored.append((sim, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        deleted = []
        for _, mem in scored[:5]:
            if self.store.delete_memory(mem["id"]):
                deleted.append(mem)
        return deleted

    # =================================================================
    # 2. 混合触发自动抽取（选项 A：高信号正则门控 + 轻量 LLM JSON 抽取）
    # =================================================================

    @staticmethod
    def has_memory_signal(query: str) -> bool:
        q = (query or "").strip()
        if len(q) < 4:
            return False
        return any(p.search(q) for p in HIGH_SIGNAL_PATTERNS)

    def maybe_extract_and_save(
        self,
        query: str,
        answer: str = "",
        user_id: str = "default_user",
        session_id: Optional[str] = None,
        auto_extract: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        混合触发长期记忆抽取：
        1. 先走 0 成本正则门控（has_memory_signal），普通知识查询/闲聊直接跳过；
        2. 命中高信号时，调用轻量 deepseek-chat 抽取结构化 JSON（支持新增 add 与冲突更新 update）；
        3. 若 API 异常，自动回退为本地规则提取，保证零报错。
        """
        if not auto_extract or not self.has_memory_signal(query):
            return []

        existing_mems = self.store.list_memories(user_id=user_id, limit=30)
        extracted_ops = self._llm_extract_memories(query, answer, existing_mems)
        if extracted_ops is None:
            # API 不可用时的本地兜底提取
            extracted_ops = self._rule_fallback_extract(query)

        events: List[Dict[str, Any]] = []
        for op in extracted_ops[:3]:
            action = (op.get("action") or "add").lower()
            content = (op.get("content") or "").strip()
            if not content or len(content) < 4:
                continue
            category = op.get("category") or _infer_category(content)
            if category not in MEMORY_CATEGORY_LABELS:
                category = _infer_category(content)
            importance = int(op.get("importance") or 4)
            target_id = op.get("target_id")

            if action == "update" and target_id:
                ok = self.store.update_memory(
                    memory_id=int(target_id),
                    content=content,
                    category=category,
                    keywords=_extract_keywords(content),
                    importance=importance,
                )
                if ok:
                    events.append(
                        {
                            "action": "updated",
                            "id": int(target_id),
                            "category": category,
                            "content": content,
                        }
                    )
                    continue

            ev = self._upsert_single_memory(
                content=content,
                category=category,
                importance=importance,
                user_id=user_id,
                session_id=session_id,
            )
            if ev:
                events.append(ev)
        return events

    def _llm_extract_memories(
        self,
        query: str,
        answer: str,
        existing_mems: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        existing_block = "（当前无已有记忆）"
        if existing_mems:
            lines = [f"- [id={m['id']}] ({m['category']}) {m['content']}" for m in existing_mems[:20]]
            existing_block = "\n".join(lines)

        prompt = (
            "请判断用户在最新发言中是否透露了值得【跨会话长期记住】的稳定信息"
            "（如：用户姓名/称呼、职业角色、负责的集群/项目/系统、长期回答格式偏好、明确约定）。\n"
            "注意：\n"
            "- 不要保存一次性的临时问题、寒暄或具体工单查出来的临时答案；\n"
            "- 如果新信息更新/修正了【已有记忆】中的某一条（例如职位或偏好变化），请使用 action=\"update\" 并填对应 target_id；\n"
            "- 如果与已有记忆完全重复，或没有值得长期记住的内容，请返回空数组 []；\n"
            "- 严格输出合法 JSON 数组，不要输出任何额外解释。每项格式：\n"
            '  {"action": "add" 或 "update", "target_id": null或整数, "category": "user_profile|preference|project_focus|knowledge_fact|custom", "content": "简洁的一句话记忆", "importance": 1到5}\n\n'
            f"【已有记忆】：\n{existing_block}\n\n"
            f"【用户最新发言】：\n{query.strip()[:500]}"
        )

        try:
            resp = _gen_mod.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是精准的长期记忆抽取器，只输出严格 JSON 数组。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=320,
            )
            raw = (resp.choices[0].message.content or "").strip()
            # 清理可能的 ```json 代码块包裹
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return []
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            return []
        except Exception:
            return None

    def _rule_fallback_extract(self, query: str) -> List[Dict[str, Any]]:
        clean = (query or "").strip()
        if not clean:
            return []
        return [
            {
                "action": "add",
                "target_id": None,
                "category": _infer_category(clean),
                "content": clean[:120],
                "importance": 4,
            }
        ]

    def _upsert_single_memory(
        self,
        content: str,
        category: str,
        importance: int,
        user_id: str,
        session_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        clean = (content or "").strip()
        if not clean:
            return None

        existing = self.store.list_memories(user_id=user_id, limit=100)
        best_sim = 0.0
        best_mem = None
        for mem in existing:
            sim = _text_similarity(clean, mem.get("content", ""))
            if sim > best_sim:
                best_sim = sim
                best_mem = mem

        kw = _extract_keywords(clean)
        # 高度相似则更新已有条目，防止重复堆砌
        if best_mem is not None and best_sim >= 0.62:
            if best_mem["content"].strip() == clean:
                # 完全一致时仅刷新重要度/时间，不重复播报新增
                self.store.update_memory(best_mem["id"], importance=max(best_mem.get("importance", 3), importance))
                return {
                    "action": "updated",
                    "id": best_mem["id"],
                    "category": best_mem["category"],
                    "content": clean,
                }
            self.store.update_memory(
                best_mem["id"],
                content=clean,
                category=category,
                keywords=kw,
                importance=max(best_mem.get("importance", 3), importance),
            )
            return {
                "action": "updated",
                "id": best_mem["id"],
                "category": category,
                "content": clean,
            }

        mid = self.store.add_memory(
            content=clean,
            category=category,
            keywords=kw,
            source_session_id=session_id,
            importance=importance,
            user_id=user_id,
        )
        if mid > 0:
            return {
                "action": "added",
                "id": mid,
                "category": category,
                "content": clean,
            }
        return None

    # =================================================================
    # 3. 混合召回与 Prompt 上下文组装
    # =================================================================

    def recall(
        self,
        query: str,
        user_id: str = "default_user",
        top_k: int = 5,
        include_core_profile: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        从 MySQL 长期记忆中按需召回：
        - 核心画像/高优先级回答偏好（user_profile / preference 且 importance >= 4）优先保底召回；
        - 其余记忆按“词法+双字片相似度 + 重要度 + 时间新鲜度”综合打分召回 Top-K；
        - 命中后自动回写 MySQL 累加 hit_count 与 last_accessed_at。
        """
        all_mems = self.store.list_memories(user_id=user_id, limit=200)
        if not all_mems:
            return []

        now = time.time()
        q_tokens = set(_tokenize_mixed(query))
        scored_items: List[Tuple[float, Dict[str, Any], str]] = []

        for mem in all_mems:
            content = mem.get("content", "")
            keywords = mem.get("keywords", "")
            category = mem.get("category", "custom")
            importance = int(mem.get("importance", 3) or 3)
            updated_at = float(mem.get("updated_at", now) or now)

            blob = f"{content} {keywords}"
            m_tokens = set(_tokenize_mixed(blob))
            overlap = len(q_tokens & m_tokens)
            lexical_score = (overlap / max(1, min(len(q_tokens), 8))) if q_tokens else 0.0
            sim_score = _text_similarity(query, blob)
            rel_score = max(lexical_score, sim_score)

            # 30 天内轻微时间新鲜度加分
            age_days = max(0.0, (now - updated_at) / 86400.0)
            recency_bonus = max(0.0, 0.12 * (1.0 - min(age_days, 30.0) / 30.0))
            imp_bonus = (importance / 5.0) * 0.18

            is_core = include_core_profile and category in ("user_profile", "preference") and importance >= 4
            total_score = rel_score * 0.70 + imp_bonus + recency_bonus + (0.35 if is_core else 0.0)

            # 命中门槛：核心画像/偏好直接入围，其他记忆需具备实际语义/词法相关性
            if is_core or rel_score >= 0.16:
                reason = "核心画像/偏好" if (is_core and rel_score < 0.16) else "语义相关命中"
                enriched = dict(mem)
                enriched["recall_score"] = round(total_score, 3)
                enriched["recall_reason"] = reason
                scored_items.append((total_score, enriched, reason))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        selected = [item[1] for item in scored_items[:top_k]]

        if selected:
            self.store.touch_memories([m["id"] for m in selected])
        return selected

    @staticmethod
    def format_memory_context(
        recalled_memories: Optional[List[Dict[str, Any]]] = None,
        rolling_summary: str = "",
    ) -> str:
        """统一拼装注入 System Prompt 的【长期记忆 + 短期滚动摘要】上下文"""
        blocks: List[str] = []

        if recalled_memories:
            ltm_lines = []
            for m in recalled_memories:
                cat_cn = MEMORY_CATEGORY_LABELS.get(m.get("category", "custom"), "记忆")
                ltm_lines.append(f"- [{cat_cn}] {m.get('content', '')}")
            blocks.append("【跨会话长期记忆（用户画像、偏好与已记住的事实）】：\n" + "\n".join(ltm_lines))

        if rolling_summary and rolling_summary.strip():
            blocks.append(f"【本会话早期对话摘要记忆】：\n{rolling_summary.strip()}")

        return "\n\n".join(blocks)
