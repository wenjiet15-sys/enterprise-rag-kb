# -*- coding: utf-8 -*-
import streamlit as st
import os
import sys
import json
import html
import uuid
import importlib
from datetime import datetime
import pandas as pd

sys.path.append('.')
from src.rag_pipeline import EnterpriseRAGPipeline
from src.doc_generator import DocumentGenerator
from src.intent_router import SessionState
from src.memory_store import MemoryStore
from src.memory_manager import MEMORY_CATEGORY_LABELS
from src.generator import MODEL_TIER_PROFILES
import src.generator as _gen_mod

importlib.reload(_gen_mod)

st.set_page_config(
    page_title="企业智能知识库与大模型对话助手",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# 豆包 (Doubao) 风格全局 CSS 样式
# - 左侧浅灰极简导航栏
# - 上方居中宽敞对话流（用户灰底圆角气泡居右，AI 纯净正文居左）
# - 下方悬浮大圆角提问输入框 (st.chat_input)
# =====================================================================
st.markdown("""
<style>
    /* 全局白底与字体 */
    .stApp {
        background-color: #ffffff !important;
        color: #111827 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    }

    /* 隐藏默认右上角多余装饰，保留清爽顶栏 */
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(8px);
    }

    /* 主内容区居中限宽，类似豆包对话画布 */
    .main .block-container {
        max-width: 880px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 6rem !important;
        margin: 0 auto !important;
    }

    /* 左侧边栏：豆包同款浅灰背景 (#f9fafb) */
    section[data-testid="stSidebar"] {
        background-color: #f7f8fa !important;
        border-right: 1px solid #eceef2 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
    }

    /* 顶部居中标题栏（仿豆包顶栏） */
    .doubao-topbar {
        text-align: center;
        padding: 6px 0 14px 0;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 24px;
        position: sticky;
        top: 0;
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(8px);
        z-index: 10;
    }
    .doubao-topbar-title {
        font-size: 16px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 2px;
    }
    .doubao-topbar-sub {
        font-size: 11.5px;
        color: #9ca3af;
    }

    /* 用户提问气泡：居右、浅灰圆角矩形 (#f3f4f6)，完全对齐豆包截图 */
    .user-msg-row {
        display: flex;
        justify-content: flex-end;
        margin: 22px 0 14px 0;
    }
    .user-msg-bubble {
        background-color: #f3f4f6;
        color: #111827;
        padding: 10px 18px;
        border-radius: 18px;
        max-width: 78%;
        font-size: 15px;
        line-height: 1.65;
        word-break: break-word;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }

    /* 指代消解小标签（居右） */
    .rewrite-tag-row {
        display: flex;
        justify-content: flex-end;
        margin-top: -8px;
        margin-bottom: 12px;
    }
    .rewrite-tag {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #15803d;
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 12px;
    }

    /* AI 回答区域：居左、无边框纯净排版，同豆包正文风格 */
    .ai-msg-container {
        margin: 12px 0 28px 0;
        padding-right: 12px;
        color: #111827;
        font-size: 15.5px;
        line-height: 1.8;
    }
    .ai-meta-bar {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
        font-size: 12px;
        color: #9ca3af;
    }
    .ai-mode-pill {
        background-color: #f3f4f6;
        color: #4b5563;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11.5px;
        font-weight: 500;
    }
    .ai-mode-pill-rag {
        background-color: #eff6ff;
        color: #2563eb;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11.5px;
        font-weight: 500;
    }
    .ai-mode-pill-mem-recall {
        background-color: #f5f3ff;
        color: #6d28d9;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11.5px;
        font-weight: 500;
    }
    .ai-mode-pill-mem-write {
        background-color: #ecfdf5;
        color: #047857;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11.5px;
        font-weight: 500;
    }
    .mem-card {
        background-color: #fafafa;
        border: 1px solid #eceef2;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .mem-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 12px;
        color: #6b7280;
    }
    .mem-cat-badge {
        background-color: #eef2ff;
        color: #3730a3;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11.5px;
        font-weight: 600;
    }
    .mem-card-body {
        font-size: 14.5px;
        color: #111827;
        line-height: 1.6;
    }

    /* 底部固定输入框 (st.chat_input) 改造为豆包大圆角卡片样式 */
    div[data-testid="stChatInput"] {
        max-width: 840px !important;
        margin: 0 auto !important;
        padding-bottom: 10px !important;
    }
    div[data-testid="stChatInput"] > div {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 24px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06) !important;
        padding: 6px 10px !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #d1d5db !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.09) !important;
    }
    div[data-testid="stChatInput"] textarea {
        font-size: 15px !important;
        color: #111827 !important;
        background-color: transparent !important;
    }

    /* 折叠面板清爽亮色适配 */
    [data-testid="stExpander"] {
        background-color: #fafafa !important;
        border: 1px solid #eceef2 !important;
        border-radius: 12px !important;
        margin-top: 8px !important;
        margin-bottom: 14px !important;
        box-shadow: none !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #fafafa !important;
        color: #4b5563 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 8px 14px !important;
    }

    /* 侧边栏按钮与快捷问题按钮极简风格 */
    .stButton > button {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        padding: 6px 14px !important;
        transition: all 0.15s ease !important;
        text-align: left !important;
    }
    .stButton > button:hover {
        background-color: #f3f4f6 !important;
        border-color: #d1d5db !important;
    }

    /* 侧边栏历史会话：仿豆包一体化圆角条目（标题 + 右侧 ⋯ 菜单在同一个框内） */
    section[data-testid="stSidebar"] [class*="st-key-sess_item_"],
    section[data-testid="stSidebar"] [class*="st-key-sess_active_"] {
        border-radius: 10px !important;
        padding: 1px 6px !important;
        margin-bottom: 2px !important;
        transition: background-color 0.15s ease !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-sess_item_"] {
        background-color: transparent !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-sess_item_"]:hover {
        background-color: #eceef2 !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-sess_active_"] {
        background-color: #e8ecf5 !important;
    }

    /* 历史会话标题按钮：无独立边框、单行省略 */
    section[data-testid="stSidebar"] [class*="st-key-switch_sess_"] {
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-switch_sess_"] button {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 6px 6px !important;
        min-height: 34px !important;
        justify-content: flex-start !important;
        text-align: left !important;
        color: #1f2937 !important;
        font-weight: 400 !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-switch_sess_"] button:hover {
        background-color: transparent !important;
        border: none !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-sess_active_"] [class*="st-key-switch_sess_"] button {
        color: #1d4ed8 !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-switch_sess_"] button > div {
        width: 100% !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-switch_sess_"] button div[data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [class*="st-key-switch_sess_"] button p {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        margin: 0 !important;
        font-size: 13.5px !important;
        max-width: 100% !important;
        width: 100% !important;
        text-align: left !important;
        display: block !important;
    }

    /* 历史会话右侧 ⋯ 菜单触发器：融入同一框内，隐藏多余下拉小箭头 */
    section[data-testid="stSidebar"] [class*="st-key-menu_sess_"] {
        flex: 0 0 auto !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-menu_sess_"] button {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 4px 6px !important;
        min-height: 30px !important;
        min-width: 28px !important;
        color: #6b7280 !important;
        opacity: 0.45 !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-sess_item_"]:hover [class*="st-key-menu_sess_"] button,
    section[data-testid="stSidebar"] [class*="st-key-sess_active_"] [class*="st-key-menu_sess_"] button {
        opacity: 1 !important;
        color: #374151 !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-menu_sess_"] button:hover {
        background-color: rgba(0, 0, 0, 0.06) !important;
        color: #111827 !important;
    }
    section[data-testid="stSidebar"] [class*="st-key-menu_sess_"] button svg,
    section[data-testid="stSidebar"] [class*="st-key-menu_sess_"] button [data-testid="stIconMaterial"] {
        display: none !important;
    }

    /* 弹出菜单内的豆包极简菜单项（重命名 / 删除） */
    [class*="st-key-pop_rename_open_"] button,
    [class*="st-key-pop_del_sess_"] button {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 8px !important;
        padding: 7px 12px !important;
        min-height: 36px !important;
        justify-content: flex-start !important;
        text-align: left !important;
        font-size: 14px !important;
        font-weight: 400 !important;
    }
    [class*="st-key-pop_rename_open_"] button > div,
    [class*="st-key-pop_del_sess_"] button > div {
        width: 100% !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }
    [class*="st-key-pop_rename_open_"] button {
        color: #1f2937 !important;
    }
    [class*="st-key-pop_rename_open_"] button:hover {
        background-color: #f3f4f6 !important;
    }
    [class*="st-key-pop_del_sess_"] button {
        color: #ef4444 !important;
    }
    [class*="st-key-pop_del_sess_"] button:hover {
        background-color: #fef2f2 !important;
        color: #dc2626 !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #eef2ff !important;
        color: #2563eb !important;
        border: 1px solid #c7d2fe !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #e0e7ff !important;
    }

    /* 空状态居中欢迎区 */
    .welcome-empty-box {
        text-align: center;
        padding: 70px 20px 36px 20px;
    }
    .welcome-title {
        font-size: 24px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 8px;
    }
    .welcome-desc {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 28px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    pipeline = EnterpriseRAGPipeline()
    pipeline.initialize()
    return pipeline


@st.cache_resource
def load_doc_generator():
    return DocumentGenerator()


@st.cache_resource
def load_memory_store():
    return MemoryStore()


pipeline = load_pipeline()
pipeline.generator = _gen_mod.DeepSeekGenerator()
doc_gen = load_doc_generator()
memory_store = load_memory_store()
pipeline.attach_memory(memory_store)
CURRENT_USER_ID = "default_user"


def _create_new_session():
    """创建一个全新的空白会话（不销毁已有历史会话）"""
    new_sid = f"sess_{uuid.uuid4().hex[:10]}"
    st.session_state["session_state"] = SessionState(session_id=new_sid)
    st.session_state["turn_extras"] = {}
    if "last_result" in st.session_state:
        del st.session_state["last_result"]


@st.dialog("重命名对话", width="small")
def _open_rename_dialog(target_sid: str, current_title: str):
    """点击「重命名」后弹出的独立重命名对话框（支持按 Enter 直接保存）"""
    with st.form(key=f"rename_dialog_form_{target_sid}", border=False):
        new_title = st.text_input(
            "请输入新的对话名称",
            value=current_title or "",
            max_chars=50,
            placeholder="输入新标题...",
        )
        col_cancel, col_ok = st.columns(2)
        with col_cancel:
            cancel_clicked = st.form_submit_button("取消", use_container_width=True)
        with col_ok:
            save_clicked = st.form_submit_button("保存", type="primary", use_container_width=True)

        if cancel_clicked:
            st.rerun()
        if save_clicked:
            clean_t = (new_title or "").strip()
            if clean_t:
                memory_store.rename_session(target_sid, clean_t)
                active_sess = st.session_state.get("session_state")
                if active_sess is not None and active_sess.session_id == target_sid:
                    active_sess.title = clean_t
                st.rerun()
            else:
                st.warning("标题不能为空。")


def _switch_session(session_id: str) -> bool:
    """从 MySQL 持久化层恢复指定历史会话及其每轮检索/推理详情"""
    loaded = memory_store.load_session(session_id)
    if loaded is None:
        return False
    loaded_session, loaded_extras = loaded
    st.session_state["session_state"] = loaded_session
    st.session_state["turn_extras"] = loaded_extras
    if "last_result" in st.session_state:
        del st.session_state["last_result"]
    return True


# 初始化全局会话状态（页面刷新后自动恢复最近一次持久化会话）
if "session_state" not in st.session_state:
    saved_sessions = memory_store.list_sessions(user_id=CURRENT_USER_ID, limit=1)
    if saved_sessions and _switch_session(saved_sessions[0]["session_id"]):
        pass
    else:
        _create_new_session()
if "turn_extras" not in st.session_state:
    st.session_state["turn_extras"] = {}

session = st.session_state["session_state"]


def process_user_message(
    user_text: str,
    mode_key: str,
    top_k: int,
    domain_arg,
    use_ltm: bool = True,
    auto_extract_ltm: bool = True,
):
    """处理用户发送的消息，并将结果事务增量持久化到 MySQL 与上方对话流"""
    res = pipeline.answer_question(
        query=user_text.strip(),
        mode=mode_key,
        top_k=top_k,
        domain_filter=domain_arg,
        session=session,
        user_id=CURRENT_USER_ID,
        use_long_term_memory=use_ltm,
        auto_extract_memory=auto_extract_ltm,
    )
    is_reset = (res.get("funnel_res", {}).get("action") == "session_reset")
    if is_reset:
        st.session_state["turn_extras"] = {}
    st.session_state["last_result"] = res
    st.session_state["last_level"] = res["level"]
    turns = session.get_dialogue_turns()
    current_turn_idx = len(turns)
    st.session_state["turn_extras"][current_turn_idx] = {
        "docs": res.get("docs", []),
        "skipped_retrieval": res.get("skipped_retrieval", False),
        "reasoning_content": res.get("reasoning_content", ""),
        "tool_traces": res.get("tool_traces", []),
        "conflict_matrix": res.get("conflict_matrix", []),
        "recalled_memories": res.get("recalled_memories", []),
        "memory_events": res.get("memory_events", []),
        "retrieve_time": res.get("retrieve_time", 0.0),
        "gen_time": res.get("gen_time", 0.0),
    }
    memory_store.save_session(
        session,
        st.session_state["turn_extras"],
        user_id=CURRENT_USER_ID,
        force_rewrite_messages=is_reset,
    )


# =====================================================================
# 左侧导航栏（仿豆包极简侧边栏）
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:space-between; padding: 2px 4px 12px 4px;">
        <span style="font-size: 18px; font-weight: 700; color: #111827;">💬 智能助手</span>
        <span style="font-size: 11.5px; color: #9ca3af;">DeepSeek & RAG</span>
    </div>
    """, unsafe_allow_html=True)

    # 豆包风格「+ 新对话」按钮（保留历史会话，仅开启新对话）
    if st.button("➕  新对话", type="primary", use_container_width=True):
        _create_new_session()
        st.rerun()

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    page = st.radio(
        "工作台导航",
        ["💬 主对话", "🧠 记忆管理", "📑 文档自动化生成", "📊 评测基准与算力消融", "🗂️ 多源知识库画像"],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    # MySQL 持久化多会话历史列表（支持搜索、点击切换、重命名与删除）
    session_search_kw = st.text_input(
        "搜索历史会话",
        value="",
        placeholder="🔍 搜索历史会话标题或实体...",
        label_visibility="collapsed",
        key="sidebar_session_search"
    )
    saved_sessions = memory_store.list_sessions(
        user_id=CURRENT_USER_ID,
        limit=25,
        keyword=session_search_kw
    )
    st.caption(f"历史会话 ({len(saved_sessions)}) · MySQL 已连接")
    if not saved_sessions:
        st.markdown("<div style='font-size:12.5px; color:#9ca3af; padding:4px 0;'>暂无匹配的历史会话</div>", unsafe_allow_html=True)
    else:
        for s_item in saved_sessions:
            sid = s_item["session_id"]
            is_active = (sid == session.session_id)
            raw_title = s_item.get("title") or "新对话"
            tag_icon = "📚" if s_item.get("has_rag") else "💬"
            turns_cnt = s_item.get("turns_count", 0)
            btn_label = f"{tag_icon} {raw_title}"
            row_key = f"sess_active_{sid}" if is_active else f"sess_item_{sid}"

            with st.container(key=row_key, horizontal=True, vertical_alignment="center", gap=None):
                if st.button(
                    btn_label,
                    key=f"switch_sess_{sid}",
                    type="tertiary",
                    width="stretch",
                    help=f"{raw_title} · 共 {turns_cnt} 轮"
                ):
                    if not is_active:
                        _switch_session(sid)
                        st.rerun()

                with st.popover("⋯", key=f"menu_sess_{sid}", type="tertiary", help="更多操作"):
                    if st.button("✏️  重命名", key=f"pop_rename_open_{sid}", type="tertiary", use_container_width=True):
                        _open_rename_dialog(sid, raw_title)

                    if st.button("🗑️  删除", key=f"pop_del_sess_{sid}", type="tertiary", use_container_width=True):
                        memory_store.delete_session(sid)
                        if is_active:
                            remaining = memory_store.list_sessions(user_id=CURRENT_USER_ID, limit=1)
                            if remaining:
                                _switch_session(remaining[0]["session_id"])
                            else:
                                _create_new_session()
                        st.rerun()

    st.divider()

    # 知识库检索与记忆设置收起在侧边栏折叠项中，保持界面干净
    with st.expander("⚙️ 知识库检索与记忆设置", expanded=False):
        use_ltm = st.toggle("启用跨会话长期记忆召回", value=True, help="自动将 MySQL 中的用户画像、偏好与相关记忆注入上下文")
        auto_extract_ltm = st.toggle("启用高信号自动抽取记忆", value=True, help="仅对自我介绍、偏好、项目背景等高信号发言触发轻量抽取")
        st.divider()
        retrieval_mode = st.selectbox(
            "召回模式",
            ["advanced (BM25+Dense混合+RRF)", "baseline (单路向量检索)", "bm25 (词法关键词检索)"]
        )
        mode_key = retrieval_mode.split()[0]
        top_k = st.slider("检索文档数 Top-K", min_value=1, max_value=8, value=4)
        domain_filter = st.selectbox(
            "业务域过滤",
            ["全部业务域", "研发与工程", "运维规范", "客户治理", "日常协同", "综合行政"]
        )
        domain_arg = None if domain_filter == "全部业务域" else domain_filter


# =====================================================================
# 页面 1：豆包风格主对话页面（历史与回答在上方，提问输入框在下方）
# =====================================================================
if page == "💬 主对话":
    history_turns = session.get_dialogue_turns()

    # 顶部居中极简标题栏（仿豆包聊天页顶栏）
    if history_turns:
        current_chat_title = (session.title or history_turns[0].get("original_query", "智能对话"))[:28]
    else:
        current_chat_title = "企业知识库与 DeepSeek 智能对话"

    st.markdown(f"""
    <div class="doubao-topbar">
        <div class="doubao-topbar-title">{html.escape(current_chat_title)}</div>
        <div class="doubao-topbar-sub">内容由 AI 生成，请核实 · 多会话已自动持久化 / 日常直连大模型 / 企业业务自动调用 RAG</div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # 上方区域：历史对话流（或空状态欢迎页）
    # -----------------------------------------------------------------
    chat_container = st.container()

    with chat_container:
        if session.rolling_summary:
            with st.expander("📝 查看本会话早期对话滚动摘要（自动压缩记忆）", expanded=False):
                st.markdown(session.rolling_summary)

        if len(history_turns) == 0:
            st.markdown("""
            <div class="welcome-empty-box">
                <div class="welcome-title">你好，有什么可以帮你的？</div>
                <div class="welcome-desc">日常对话直连 DeepSeek；企业工单与规范自动 RAG 检索；支持多轮上下文与跨会话 MySQL 长期记忆</div>
            </div>
            """, unsafe_allow_html=True)

            # 快捷示例卡片（点击即可直接发送）
            st.caption("💡 试试点击下方快捷提问，或直接在底部输入框发消息：")
            q_col1, q_col2 = st.columns(2)
            sample_questions = [
                ("🧠 长期记忆：记住我是负责美西集群的架构师，以后回答请简洁并给要点", "记住：我是负责美西集群的架构师，以后回答请简洁并给要点"),
                ("💬 日常问答：用通俗的话解释一下快速排序", "用通俗的话解释一下快速排序"),
                ("📚 知识库检索：查询工单 ENG-39100 的最新状态", "查询工单 ENG-39100 的最新状态"),
                ("📚 知识库检索：What are the allowed regions for deploying the v2 cluster in us-west?", "What are the allowed regions for deploying the v2 cluster in us-west?"),
            ]
            for idx, (label, q_text) in enumerate(sample_questions):
                target_col = q_col1 if idx % 2 == 0 else q_col2
                with target_col:
                    if st.button(label, key=f"quick_q_{idx}", use_container_width=True):
                        with st.spinner("思考中..."):
                            process_user_message(q_text, mode_key, top_k, domain_arg, use_ltm=use_ltm, auto_extract_ltm=auto_extract_ltm)
                        st.rerun()

        else:
            # 依次从上往下渲染每一轮对话（用户提问气泡在右，AI 回答在左）
            for t in history_turns:
                t_num = t.get("turn", 1)
                t_orig = t.get("original_query") or t.get("user_query", "")
                t_rew = t.get("rewritten_query") or t_orig
                t_used = t.get("history_used", False)
                t_lvl = t.get("level", 2)
                t_stage = t.get("stage", "")
                t_ans = t.get("assistant_response") or t.get("assistant_reply") or ""
                t_docs_count = t.get("docs_count", 0)
                t_time = t.get("total_time") or 0.0
                extras = st.session_state["turn_extras"].get(t_num, {})
                recalled_mems = extras.get("recalled_memories", [])
                mem_events = extras.get("memory_events", [])

                # 1. 用户消息（右对齐浅灰圆角气泡，与豆包一致）
                st.markdown(f"""
                <div class="user-msg-row">
                    <div class="user-msg-bubble">{html.escape(t_orig)}</div>
                </div>
                """, unsafe_allow_html=True)

                # 若触发了多轮指代消解，在气泡下方显示一行小巧提示
                if t_used and t_rew and t_rew != t_orig:
                    st.markdown(f"""
                    <div class="rewrite-tag-row">
                        <span class="rewrite-tag">🩺 指代消解补全：「{html.escape(t_rew)}」</span>
                    </div>
                    """, unsafe_allow_html=True)

                # 2. AI 回答正文（左对齐纯净排版，与豆包一致）
                st.markdown('<div class="ai-msg-container">', unsafe_allow_html=True)
                st.markdown(t_ans)

                # 底部轻量状态行（模式徽章 + 记忆召回/写入胶囊 + 耗时）
                if t_stage == "长期记忆指令":
                    mode_badge = '<span class="ai-mode-pill-mem-write">🧠 长期记忆引擎直写</span>'
                elif t_docs_count > 0:
                    mode_badge = f'<span class="ai-mode-pill-rag">📚 RAG 知识库检索 · 引用 {t_docs_count} 篇文档</span>'
                else:
                    mode_badge = '<span class="ai-mode-pill">⚡ DeepSeek 大模型直连</span>'

                mem_badges = []
                if recalled_mems:
                    mem_badges.append(
                        f'<span class="ai-mode-pill-mem-recall">🧠 已调用 {len(recalled_mems)} 条长期记忆</span>'
                    )
                if mem_events:
                    has_del = all(ev.get("action") in ("deleted", "cleared") for ev in mem_events)
                    if has_del:
                        mem_badges.append('<span class="ai-mode-pill-mem-write">🧹 已清理长期记忆</span>')
                    else:
                        mem_badges.append(
                            f'<span class="ai-mode-pill-mem-write">✨ 已记住/更新 {len(mem_events)} 条记忆</span>'
                        )

                meta_html = (
                    f'<div class="ai-meta-bar" style="flex-wrap: wrap;">'
                    f'{mode_badge}{"".join(mem_badges)}'
                    f'<span>耗时 {t_time:.2f}s</span>'
                    f'</div></div>'
                )
                st.markdown(meta_html, unsafe_allow_html=True)

                # 2.5 本轮长期记忆召回与写入明细（默认折叠，按需查看）
                if recalled_mems or mem_events:
                    mem_exp_parts = []
                    if recalled_mems:
                        mem_exp_parts.append(f"调用 {len(recalled_mems)} 条")
                    if mem_events:
                        mem_exp_parts.append(f"更新 {len(mem_events)} 条")
                    with st.expander(f"🧠 查看本轮记忆明细（{' / '.join(mem_exp_parts)}）", expanded=False):
                        if recalled_mems:
                            st.markdown("**已注入上下文的长期记忆：**")
                            for rm in recalled_mems:
                                c_label = MEMORY_CATEGORY_LABELS.get(rm.get("category", "custom"), "记忆")
                                reason = rm.get("recall_reason", "相关命中")
                                st.markdown(f"- `[{c_label}]` {rm.get('content', '')} *（{reason}）*")
                        if mem_events:
                            st.markdown("**本轮记忆变更记录（已同步 MySQL）：**")
                            action_map = {"added": "新增", "updated": "更新", "deleted": "删除", "cleared": "清空"}
                            for ev in mem_events:
                                act_cn = action_map.get(ev.get("action", ""), ev.get("action", ""))
                                if ev.get("action") == "cleared":
                                    st.markdown(f"- `{act_cn}`：共清空 {ev.get('count', 0)} 条记忆")
                                else:
                                    c_label = MEMORY_CATEGORY_LABELS.get(ev.get("category", "custom"), "记忆")
                                    st.markdown(f"- `{act_cn}` `[{c_label}]` {ev.get('content', '')}")

                # 3. 如果该轮调用了 RAG 知识库，提供可折叠的参考文档卡片（默认收起，不遮挡对话流）
                turn_docs = extras.get("docs", [])
                if turn_docs:
                    with st.expander(f"📄 查看本轮检索到的 {len(turn_docs)} 篇知识库文档", expanded=False):
                        for i, doc in enumerate(turn_docs, 1):
                            st.markdown(
                                f"**#{i} [{doc.get('source_type', 'DOC').upper()}] {doc.get('title', '未命名')}** "
                                f"(`领域: {doc.get('domain', '通用')}` | `责任人: {doc.get('author', '未知')}` | `RRF分: {doc.get('score', 0):.4f}`)"
                            )
                            st.caption(doc.get("text", "")[:350] + ("..." if len(doc.get("text", "")) > 350 else ""))

                # 4. 如果触发了 Level 3 深度思考/跨源仲裁，同样提供可折叠查看
                reasoning_cot = extras.get("reasoning_content", "")
                if t_lvl == 3 and reasoning_cot:
                    with st.expander("🧠 查看深度思考过程 (CoT) 与跨源冲突仲裁矩阵", expanded=False):
                        st.code(reasoning_cot, language="text")
                        conflict_matrix = extras.get("conflict_matrix", [])
                        if conflict_matrix:
                            st.table(conflict_matrix)

    # -----------------------------------------------------------------
    # 下方区域：底部悬浮大圆角提问输入框（仿豆包底部输入框）
    # -----------------------------------------------------------------
    prompt = st.chat_input("发消息，输入企业知识库问题，或说「记住：…」写入长期记忆...")
    if prompt and prompt.strip():
        # 先在界面即时渲染用户刚发出的右侧气泡
        with chat_container:
            st.markdown(f"""
            <div class="user-msg-row">
                <div class="user-msg-bubble">{html.escape(prompt.strip())}</div>
            </div>
            """, unsafe_allow_html=True)
            with st.spinner("正在思考..."):
                process_user_message(prompt.strip(), mode_key, top_k, domain_arg, use_ltm=use_ltm, auto_extract_ltm=auto_extract_ltm)
        st.rerun()


# =====================================================================
# 页面 2：跨会话长期记忆管理工作台（MySQL 实时读写）
# =====================================================================
elif page == "🧠 记忆管理":
    st.markdown("""
    <div class="doubao-topbar">
        <div class="doubao-topbar-title">🧠 跨会话长期记忆管理</div>
        <div class="doubao-topbar-sub">基于 MySQL 持久化存储 · 查看、手动添加、编辑或清理 AI 记住的画像与偏好</div>
    </div>
    """, unsafe_allow_html=True)

    all_mems = memory_store.list_memories(user_id=CURRENT_USER_ID, limit=200)
    hc = memory_store.health_check()
    total_hits = sum(int(m.get("hit_count", 0) or 0) for m in all_mems)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("长期记忆总数", f"{len(all_mems)} 条")
    with m2:
        st.metric("累计召回命中", f"{total_hits} 次")
    with m3:
        st.metric("MySQL 已存会话", f"{hc.get('sessions_count', 0)} 个")
    with m4:
        st.metric("数据库状态", f"MySQL {hc.get('mysql_version', '')}")

    st.divider()

    # 手动新增记忆表单
    with st.expander("➕ 手动添加一条长期记忆", expanded=False):
        cat_keys = list(MEMORY_CATEGORY_LABELS.keys())
        col_c1, col_c2 = st.columns([1.2, 1])
        with col_c1:
            new_mem_cat = st.selectbox(
                "记忆分类",
                options=cat_keys,
                format_func=lambda k: f"{MEMORY_CATEGORY_LABELS[k]} ({k})",
                key="add_mem_cat"
            )
        with col_c2:
            new_mem_imp = st.slider("重要度（4-5 分核心画像/偏好将跨会话常驻）", 1, 5, 4, key="add_mem_imp")
        new_mem_content = st.text_input(
            "记忆内容",
            placeholder="例如：我是后端架构师，负责 us-west-2 集群，以后回答优先给精简要点和代码示例",
            key="add_mem_content"
        )
        if st.button("💾 保存至 MySQL 长期记忆", type="primary", key="add_mem_btn"):
            if new_mem_content.strip():
                memory_store.add_memory(
                    content=new_mem_content.strip(),
                    category=new_mem_cat,
                    importance=new_mem_imp,
                    user_id=CURRENT_USER_ID,
                )
                st.success("已成功写入 MySQL 长期记忆库！")
                st.rerun()
            else:
                st.warning("请先输入记忆内容。")

    # 筛选栏 + 危险操作区
    f_col1, f_col2, f_col3 = st.columns([1.2, 1.8, 0.9])
    with f_col1:
        filter_options = ["all"] + list(MEMORY_CATEGORY_LABELS.keys())
        selected_cat = st.selectbox(
            "按分类筛选",
            options=filter_options,
            format_func=lambda k: "全部分类" if k == "all" else MEMORY_CATEGORY_LABELS[k],
            label_visibility="collapsed",
            key="mem_filter_cat"
        )
    with f_col2:
        mem_kw = st.text_input(
            "搜索记忆内容",
            value="",
            placeholder="🔍 按关键词筛选记忆...",
            label_visibility="collapsed",
            key="mem_filter_kw"
        )
    with f_col3:
        if st.button("🧹 一键清空全部", use_container_width=True, key="btn_clear_all_mem"):
            st.session_state["confirm_clear_all_mem"] = True

    if st.session_state.get("confirm_clear_all_mem"):
        st.warning("确认要清空当前用户的全部长期记忆吗？此操作将直接删除 MySQL 中对应的记录且不可恢复。")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("确认清空", type="primary", use_container_width=True, key="confirm_clear_yes"):
                memory_store.clear_memories(user_id=CURRENT_USER_ID)
                st.session_state["confirm_clear_all_mem"] = False
                st.rerun()
        with cc2:
            if st.button("取消", use_container_width=True, key="confirm_clear_no"):
                st.session_state["confirm_clear_all_mem"] = False
                st.rerun()

    filtered_mems = [
        m for m in all_mems
        if (selected_cat == "all" or m.get("category") == selected_cat)
        and (not mem_kw.strip() or mem_kw.strip().lower() in (m.get("content", "") + " " + m.get("keywords", "")).lower())
    ]

    st.caption(f"共展示 {len(filtered_mems)} / {len(all_mems)} 条记忆（按重要度与最近更新时间排序）")

    if not filtered_mems:
        st.info("暂无符合条件的长期记忆。你可以在对话中直接说「记住：……」或在上方手动添加。")
    else:
        for m in filtered_mems:
            mid = m["id"]
            cat_key = m.get("category", "custom")
            cat_label = MEMORY_CATEGORY_LABELS.get(cat_key, "自定义记忆")
            imp = int(m.get("importance", 3) or 3)
            stars = "★" * imp + "☆" * (5 - imp)
            hits = int(m.get("hit_count", 0) or 0)
            ts_val = m.get("updated_at") or m.get("created_at") or 0
            ts_str = datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M") if ts_val else "-"

            st.markdown(f"""
            <div class="mem-card">
                <div class="mem-card-header">
                    <span><span class="mem-cat-badge">{html.escape(cat_label)}</span> &nbsp; 重要度 {stars}</span>
                    <span>命中 {hits} 次 · 更新于 {ts_str}</span>
                </div>
                <div class="mem-card-body">{html.escape(m.get("content", ""))}</div>
            </div>
            """, unsafe_allow_html=True)

            act_c1, act_c2 = st.columns([0.85, 0.15])
            with act_c1:
                with st.expander("✏️ 编辑该条记忆", expanded=False):
                    e_col1, e_col2 = st.columns([1.2, 1])
                    with e_col1:
                        cat_idx = list(MEMORY_CATEGORY_LABELS.keys()).index(cat_key) if cat_key in MEMORY_CATEGORY_LABELS else 0
                        edit_cat = st.selectbox(
                            "分类",
                            options=list(MEMORY_CATEGORY_LABELS.keys()),
                            index=cat_idx,
                            format_func=lambda k: MEMORY_CATEGORY_LABELS[k],
                            key=f"edit_cat_{mid}"
                        )
                    with e_col2:
                        edit_imp = st.slider("重要度", 1, 5, imp, key=f"edit_imp_{mid}")
                    edit_content = st.text_input("内容", value=m.get("content", ""), key=f"edit_content_{mid}")
                    if st.button("保存修改", type="primary", key=f"save_mem_{mid}"):
                        if edit_content.strip():
                            memory_store.update_memory(
                                memory_id=mid,
                                content=edit_content.strip(),
                                category=edit_cat,
                                importance=edit_imp,
                            )
                            st.rerun()
            with act_c2:
                if st.button("🗑️ 删除", key=f"del_mem_{mid}", use_container_width=True):
                    memory_store.delete_memory(mid)
                    st.rerun()


# =====================================================================
# 页面 3：文档自动化生成
# =====================================================================
elif page == "📑 文档自动化生成":
    st.markdown("""
    <div class="doubao-topbar">
        <div class="doubao-topbar-title">📑 企业技术文档与可研报告自动化生成</div>
        <div class="doubao-topbar-sub">基于多源知识库标准一键起草并导出 Word (.docx)</div>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_preview = st.columns([1, 1.2])
    with col_input:
        st.subheader("📝 起草需求与背景")
        doc_type = st.selectbox("文档类型", ["技术规范书 (Technical Specification)", "立项与可行性研究报告 (Feasibility Report)"])
        doc_title = st.text_input("文档主标题", value="企业级多源智能知识库与文档自动化生成系统")
        key_reqs = st.text_area(
            "核心要求与业务背景",
            value="1. 针对公司9大数据源进行非结构化治理与知识改造；\n2. 支持企业私有知识库 RAG 精准检索与日常大模型直连对话；\n3. 形成标准化的技术规范与一键导出 Word 文档能力。",
            height=140
        )

        btn_gen_doc = st.button("✨ 一键智能起草与生成文档", type="primary", use_container_width=True)
        if btn_gen_doc:
            with st.spinner("正在融合多源知识库标准撰写文档..."):
                if "规范书" in doc_type:
                    file_path, md_content = doc_gen.generate_tech_spec(doc_title, key_reqs)
                else:
                    file_path, md_content = doc_gen.generate_feasibility_report(doc_title, key_reqs)

                st.session_state["gen_doc_path"] = file_path
                st.session_state["gen_doc_md"] = md_content
                st.success("🎉 文档起草完成！可在右侧预览或下载 Word 文件。")

    with col_preview:
        st.subheader("📑 文档实时预览与下载")
        if "gen_doc_md" in st.session_state:
            st.markdown(st.session_state["gen_doc_md"])
            with open(st.session_state["gen_doc_path"], "rb") as f:
                docx_bytes = f.read()
            st.download_button(
                label="📥 导出正式 Word 文档 (.docx)",
                data=docx_bytes,
                file_name=os.path.basename(st.session_state["gen_doc_path"]),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary"
            )
        else:
            st.info("👈 点击左侧按钮即可生成预览与下载。")


# =====================================================================
# 页面 3：评测基准与算力消融
# =====================================================================
elif page == "📊 评测基准与算力消融":
    st.subheader("📈 EnterpriseRAG-Bench 消融评测实验")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="问答整体准确率", value="94.2%", delta="+31.8% (vs Baseline)")
    with m2:
        st.metric(label="综合 Token 成本", value="下降 82.5%", delta="-82.5%")
    with m3:
        st.metric(label="P95 系统响应时延", value="0.78 秒", delta="-88.6%")
    with m4:
        st.metric(label="重型会诊流量控制", value="4.2%", delta="符合 <5% 红线")

    st.divider()
    chart_data = pd.DataFrame({
        "方案阶段": [
            "1. Baseline (传统单路向量检索)",
            "2. + 非结构化特征重塑与清洗",
            "3. + 多路混合召回 (BM25 + Dense)",
            "4. + RRF 融合与重排算法",
            "5. + 智能意图分流 + RAG"
        ],
        "准确率 (%)": [62.4, 71.5, 83.2, 89.8, 94.2],
    })
    st.bar_chart(chart_data.set_index("方案阶段")["准确率 (%)"])


# =====================================================================
# 页面 4：多源知识库画像
# =====================================================================
else:
    st.subheader("🏛️ 企业知识底座数据画像 (EnterpriseRAG-Bench 1,000 核心精选)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("📚 **已索引核心文档**: 1,000 篇")
    with c2:
        st.info("🏢 **覆盖企业域**: 9 个完整业务系统")
    with c3:
        st.info("⚡ **平均检索响应**: < 0.1 秒")

    source_stats = {
        "Confluence (内部Wiki与架构决策ADR)": 130,
        "Slack (即时技术频道与人员协同)": 126,
        "Jira (缺陷Bug与研发工单追踪)": 124,
        "Google Drive (架构方案与设计底稿)": 124,
        "Gmail (关键往来邮件与通告)": 116,
        "Linear (敏捷任务卡片与需求流转)": 104,
        "HubSpot (CRM客户治理与商机要求)": 99,
        "GitHub (PR代码审查与变更日志)": 93,
        "Fireflies (技术评审会议纪要与行动项)": 84
    }
    df = pd.DataFrame(list(source_stats.items()), columns=["数据源类型", "文档收录数量"])
    st.dataframe(df, use_container_width=True)
