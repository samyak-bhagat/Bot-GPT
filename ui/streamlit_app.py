import os
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Config & styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BOT GPT",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
# If set, login password must match. If unset, any non-empty password is accepted (dev).
STREAMLIT_UI_PASSWORD = os.getenv("STREAMLIT_UI_PASSWORD", "")

_BASE_CSS = """
<style>
    .login-wrap { max-width: 420px; margin: 3rem auto; padding: 2rem;
        border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(145deg, rgba(30,30,35,0.95), rgba(20,20,28,0.98)); }
    .app-title { font-size: 1.75rem; font-weight: 600; margin-bottom: 0.25rem; }
    .app-sub { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 8px; }
</style>
"""
st.markdown(_BASE_CSS, unsafe_allow_html=True)


def _api_headers(user_id: str) -> dict[str, str]:
    return {"X-User-Id": user_id}


def _humanize_api_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        detail: str | None = None
        try:
            payload = exc.response.json()
            detail = payload.get("detail") if isinstance(payload, dict) else None
        except Exception:
            detail = None
        if detail:
            return f"{exc.response.status_code}: {detail}"
        return f"{exc.response.status_code}: {exc.response.text[:200]}"
    return str(exc)


def _init_session() -> None:
    defaults: dict[str, Any] = {
        "authenticated": False,
        "user_email": "",
        "conversation_id": None,
        "loaded_conversation_id": None,
        "chat_messages": [],
        "last_costs": None,
        "documents_cache": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _list_conversations(user_id: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{API_BASE_URL}/api/v1/conversations",
        headers=_api_headers(user_id),
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("items", [])


def _list_documents(user_id: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{API_BASE_URL}/api/v1/documents/",
        headers=_api_headers(user_id),
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("items", [])


def _ingest_document(
    user_id: str,
    file_bytes: bytes,
    filename: str,
    embedding_provider: str | None,
) -> dict[str, Any]:
    params: dict[str, str] = {}
    if embedding_provider:
        params["embedding_provider"] = embedding_provider
    response = requests.post(
        f"{API_BASE_URL}/api/v1/documents",
        headers=_api_headers(user_id),
        files={"file": (filename, file_bytes)},
        params=params,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def _get_document(user_id: str, document_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}/api/v1/documents/{document_id}",
        headers=_api_headers(user_id),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _create_conversation(
    user_id: str,
    provider: str,
    model: str,
    document_ids: list[str] | None = None,
) -> str:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/conversations",
        json={
            "provider": provider,
            "model": model,
            "document_ids": document_ids or [],
        },
        headers=_api_headers(user_id),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["id"]


def _send_message(user_id: str, conversation_id: str, content: str, provider: str, model: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/conversations/{conversation_id}/messages",
        json={"content": content, "provider": provider, "model": model},
        headers=_api_headers(user_id),
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def _conversation_costs(user_id: str, conversation_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}/api/v1/conversations/{conversation_id}/costs",
        headers=_api_headers(user_id),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _conversation_detail(user_id: str, conversation_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}/api/v1/conversations/{conversation_id}",
        headers=_api_headers(user_id),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _delete_conversation(user_id: str, conversation_id: str) -> None:
    response = requests.delete(
        f"{API_BASE_URL}/api/v1/conversations/{conversation_id}",
        headers=_api_headers(user_id),
        timeout=20,
    )
    response.raise_for_status()


def _render_cost_breakdown(costs: dict[str, Any]) -> None:
    llm = costs.get("llm", {})
    embeddings = costs.get("embeddings", {})
    totals = costs.get("totals", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cost (USD)", f"${float(totals.get('cost_usd', 0.0)):.6f}")
    col2.metric("LLM Cost (USD)", f"${float(llm.get('cost_usd', 0.0)):.6f}")
    col3.metric("Embedding Cost (USD)", f"${float(embeddings.get('cost_usd', 0.0)):.6f}")

    token_col1, token_col2, token_col3 = st.columns(3)
    token_col1.metric("LLM Prompt Tokens", str(int(llm.get("prompt_tokens", 0) or 0)))
    token_col2.metric("LLM Completion Tokens", str(int(llm.get("completion_tokens", 0) or 0)))
    token_col3.metric("Embedding Input Tokens", str(int(embeddings.get("total_input_tokens", 0) or 0)))

    docs = embeddings.get("documents", [])
    if isinstance(docs, list) and docs:
        df = pd.DataFrame(docs)
        df["estimated_cost_usd"] = df["estimated_cost_usd"].astype(float)
        st.subheader("Per-document embedding cost")
        st.dataframe(
            df.rename(
                columns={
                    "filename": "Filename",
                    "approx_input_tokens": "Input Tokens",
                    "estimated_cost_usd": "Cost USD",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        chart_df = df[["filename", "estimated_cost_usd"]].set_index("filename")
        st.bar_chart(chart_df)
    else:
        st.info("No embedding-linked documents in this conversation yet.")


def _render_login() -> None:
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    st.markdown('<p class="app-title">BOT GPT</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-sub">Sign in with your email. Chunks are embedded and stored (Postgres / pgvector).</p>', unsafe_allow_html=True)
    email = st.text_input("Email", key="login_email", placeholder="you@company.com")
    password = st.text_input("Password", type="password", key="login_password")
    col_a, col_b = st.columns(2)
    with col_a:
        sign_in = st.button("Sign in", type="primary", use_container_width=True)
    with col_b:
        dev_hint = "Dev: leave password empty if `STREAMLIT_UI_PASSWORD` is unset." if not STREAMLIT_UI_PASSWORD else ""
        if dev_hint:
            st.caption(dev_hint)

    if sign_in:
        if not email or "@" not in email:
            st.error("Enter a valid email address.")
        elif STREAMLIT_UI_PASSWORD:
            if password != STREAMLIT_UI_PASSWORD:
                st.error("Invalid email or password.")
            else:
                st.session_state.authenticated = True
                st.session_state.user_email = email.strip()
                st.rerun()
        else:
            st.session_state.authenticated = True
            st.session_state.user_email = email.strip()
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _format_ts(iso_val: Any) -> str:
    if iso_val is None:
        return ""
    if isinstance(iso_val, datetime):
        return iso_val.strftime("%Y-%m-%d %H:%M")
    s = str(iso_val)
    if "T" in s:
        return s.replace("T", " ")[:16]
    return s[:16]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
_init_session()

if not st.session_state.authenticated:
    _render_login()
    st.stop()

user_id = st.session_state.user_email

with st.sidebar:
    st.markdown("### Session")
    st.caption(user_id)
    if st.button("Sign out", type="secondary"):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.session_state.conversation_id = None
        st.session_state.loaded_conversation_id = None
        st.session_state.chat_messages = []
        st.session_state.last_costs = None
        st.session_state.documents_cache = None
        st.rerun()

    st.divider()
    st.markdown("### Model")
    provider_options = ["openai", "groq", "ollama"]
    default_provider = os.getenv("DEFAULT_PROVIDER", "openai").lower()
    if default_provider not in provider_options:
        default_provider = "openai"
    provider = st.selectbox(
        "Provider",
        provider_options,
        index=provider_options.index(default_provider),
        help="You can switch provider/model any time; new messages use current selection.",
    )

    model_default = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    model = st.text_input("Model", value=model_default)

    try:
        conversations = _list_conversations(user_id)
    except Exception as exc:
        conversations = []
        st.error(f"API unavailable: {exc}")

    if conversations:
        history_df = pd.DataFrame(conversations)[["title", "mode", "total_tokens", "total_cost_usd", "updated_at"]]
        history_df = history_df.rename(
            columns={
                "title": "Title",
                "mode": "Mode",
                "total_tokens": "LLM Tokens",
                "total_cost_usd": "Total Cost USD",
                "updated_at": "Updated At",
            }
        )
        st.subheader("History")
        st.dataframe(history_df, width="stretch", hide_index=True)

    conversation_options = {f"{item['title']} ({str(item['id'])[:8]})": item["id"] for item in conversations}
    selected_label = st.selectbox(
        "Conversation",
        options=["<new conversation>"] + list(conversation_options.keys()),
    )

    if selected_label == "<new conversation>":
        pass
    else:
        st.session_state.conversation_id = conversation_options[selected_label]

    if st.session_state.conversation_id and st.session_state.loaded_conversation_id != st.session_state.conversation_id:
        try:
            detail = _conversation_detail(user_id, st.session_state.conversation_id)
            st.session_state.chat_messages = [
                {"role": item["role"], "content": item["content"]} for item in detail.get("messages", [])
            ]
            st.session_state.last_costs = _conversation_costs(user_id, st.session_state.conversation_id)
        except Exception as exc:
            st.warning(f"Could not load messages: {exc}")
            st.session_state.chat_messages = []
        st.session_state.loaded_conversation_id = st.session_state.conversation_id

    if st.session_state.conversation_id:
        st.subheader("Manage")
        confirm_delete = st.checkbox("Confirm delete", value=False)
        if st.button("Delete conversation", type="secondary"):
            if confirm_delete:
                try:
                    _delete_conversation(user_id, st.session_state.conversation_id)
                    st.session_state.conversation_id = None
                    st.session_state.loaded_conversation_id = None
                    st.session_state.chat_messages = []
                    st.session_state.last_costs = None
                    st.success("Deleted.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
            else:
                st.warning("Confirm delete first.")

    if st.session_state.last_costs:
        totals = st.session_state.last_costs.get("totals", {})
        llm = st.session_state.last_costs.get("llm", {})
        st.metric("Session cost (USD)", f"${float(totals.get('cost_usd', 0.0)):.6f}")
        st.metric("LLM tokens", str(int(llm.get("total_tokens", 0) or 0)))
    else:
        st.metric("Session cost (USD)", "$0.000000")
        st.metric("LLM tokens", "0")

tab_chat, tab_kb = st.tabs(["Chat", "Knowledge base"])

with tab_kb:
    st.subheader("Upload & ingest")
    st.caption(
        "Files are chunked, embedded, and stored in Postgres (vectors used for RAG retrieval). "
        "Pick an embedding backend below, then attach documents when you start a chat."
    )
    emb_choice = st.radio(
        "Embedding provider",
        ["Default (app settings)", "huggingface", "openai"],
        horizontal=True,
    )
    emb_param: str | None
    if emb_choice == "Default (app settings)":
        emb_param = None
    else:
        emb_param = emb_choice

    uploaded = st.file_uploader(
        "Document",
        type=["pdf", "txt", "md", "docx"],
        help="PDF, plain text, Markdown, or Word (.docx)",
    )
    ingest_clicked = st.button("Ingest document", type="primary", disabled=uploaded is None)
    if ingest_clicked and uploaded is not None:
        data = uploaded.getvalue()
        fname = uploaded.name or "upload.bin"
        with st.spinner("Embedding chunks and writing to the database…"):
            try:
                result = _ingest_document(user_id, data, fname, emb_param)
                st.success(
                    f"Ingested **{fname}** — `{result['document_id']}` — "
                    f"{result.get('chunk_count', 0)} chunks (status: {result.get('status')})."
                )
                usage = result.get("embedding_usage") or {}
                st.json(
                    {
                        "embedding_provider": usage.get("embedding_provider"),
                        "embedding_model": usage.get("embedding_model"),
                        "approx_input_tokens": usage.get("approx_input_tokens"),
                        "estimated_cost_usd": usage.get("estimated_cost_usd"),
                    }
                )
                st.session_state.documents_cache = None
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.subheader("Your documents")
    refresh_docs = st.button("Refresh list")
    if refresh_docs or st.session_state.documents_cache is None:
        try:
            st.session_state.documents_cache = _list_documents(user_id)
        except Exception as exc:
            st.error(f"Could not list documents: {exc}")
            st.session_state.documents_cache = []

    items = st.session_state.documents_cache or []
    if items:
        rows = []
        for item in items:
            rows.append(
                {
                    "Filename": item.get("filename"),
                    "Status": item.get("status"),
                    "Chunks": item.get("chunk_count"),
                    "Created": _format_ts(item.get("created_at")),
                    "ID": item.get("document_id"),
                }
            )
        doc_df = pd.DataFrame(rows)
        st.dataframe(doc_df, width="stretch", hide_index=True)
        inspect_id = st.selectbox("Inspect document", options=[i["document_id"] for i in items])
        if st.button("Load details"):
            try:
                detail = _get_document(user_id, inspect_id)
                st.json(detail)
            except Exception as exc:
                st.warning(str(exc))
    else:
        st.info("No documents yet — upload a file above.")

with tab_chat:
    st.subheader("Chat")
    col_new1, col_new2 = st.columns([3, 1])
    with col_new1:
        if st.session_state.documents_cache is None:
            try:
                st.session_state.documents_cache = _list_documents(user_id)
            except Exception:
                st.session_state.documents_cache = []
        doc_items = st.session_state.documents_cache or []
        id_to_label = {}
        for d in doc_items:
            fn = d.get("filename") or ""
            short = fn if len(fn) <= 40 else fn[:37] + "…"
            did = d.get("document_id", "")
            id_to_label[did] = f"{short} ({str(did)[:8]})"
        selected_docs = st.multiselect(
            "RAG: attach ingested documents (optional)",
            options=list(id_to_label.keys()),
            format_func=lambda x: id_to_label.get(x, x),
            help="Only documents you uploaded for this user appear here. Start a new conversation to bind them.",
        )
    with col_new2:
        st.write("")
        st.write("")
        if st.button("New chat"):
            st.session_state.conversation_id = _create_conversation(user_id, provider, model, selected_docs)
            st.session_state.chat_messages = []
            st.session_state.last_costs = None
            st.session_state.loaded_conversation_id = None
            st.rerun()

    for msg in st.session_state.chat_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input("Ask a question…")
    if prompt:
        if not st.session_state.conversation_id:
            st.session_state.conversation_id = _create_conversation(user_id, provider, model, selected_docs)

        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        try:
            result = _send_message(user_id, st.session_state.conversation_id, prompt, provider, model)
            assistant_text = result.get("assistant_message", "")
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})
            st.session_state.last_costs = _conversation_costs(user_id, st.session_state.conversation_id)
            st.rerun()
        except Exception as exc:
            st.error(f"Message failed: {_humanize_api_error(exc)}")

    if st.session_state.conversation_id:
        with st.expander("Cost breakdown", expanded=False):
            try:
                costs = _conversation_costs(user_id, st.session_state.conversation_id)
                st.session_state.last_costs = costs
                _render_cost_breakdown(costs)
            except Exception as exc:
                st.warning(f"Could not load costs: {exc}")
