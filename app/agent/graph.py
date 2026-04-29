from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.llm.factory import Provider, get_chat_model
from app.rag.retrieval import rank_candidate_contexts


SYSTEM_PROMPT = (
    "You are BOT GPT, a concise and helpful assistant. "
    "Prefer direct answers and call out uncertainty when needed."
)

class AgentState(TypedDict):
    user_message: str
    provider: Provider
    model: str
    mode: Literal["open", "rag"]
    active_document_ids: list[str]
    history: list[dict[str, str]]
    candidate_contexts: list[dict[str, str]]
    selected_contexts: list[dict[str, str]]
    assistant_message: str
    usage: dict[str, int]


def _build_messages(state: AgentState) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    if state.get("selected_contexts"):
        context_block = "\n\n".join(
            f"[{idx + 1}] {item['content']}" for idx, item in enumerate(state["selected_contexts"])
        )
        messages.append(
            SystemMessage(
                content=(
                    "Use the retrieved context below when relevant. "
                    "If context is insufficient, say so explicitly.\n\n"
                    f"{context_block}"
                )
            )
        )
    for item in state.get("history", []):
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
    messages.append(HumanMessage(content=state["user_message"]))
    return messages


async def _generate(state: AgentState) -> AgentState:
    chat_model = get_chat_model(provider=state["provider"], model=state["model"])
    response = await chat_model.ainvoke(_build_messages(state))
    usage: dict[str, int] = {}
    raw_usage: Any = getattr(response, "usage_metadata", None) or {}
    if isinstance(raw_usage, dict):
        usage = {
            "input_tokens": int(raw_usage.get("input_tokens", 0) or 0),
            "output_tokens": int(raw_usage.get("output_tokens", 0) or 0),
            "total_tokens": int(raw_usage.get("total_tokens", 0) or 0),
        }
    state["assistant_message"] = response.content if isinstance(response.content, str) else str(response.content)
    state["usage"] = usage
    return state


def _route(state: AgentState) -> str:
    if state["mode"] == "rag" and state.get("active_document_ids"):
        return "retrieve"
    return "generate"


async def _retrieve(state: AgentState) -> AgentState:
    candidates = state.get("candidate_contexts", [])
    if not candidates:
        state["selected_contexts"] = []
        return state

    if all("score" in item for item in candidates):
        sorted_candidates = sorted(
            candidates,
            key=lambda item: float(item.get("score", 0.0) or 0.0),
            reverse=True,
        )
        state["selected_contexts"] = sorted_candidates[:3]
        return state

    state["selected_contexts"] = rank_candidate_contexts(state["user_message"], candidates, top_k=3)
    return state


_graph = StateGraph(AgentState)
_graph.add_node("retrieve", _retrieve)
_graph.add_node("generate", _generate)
_graph.add_conditional_edges(START, _route, {"retrieve": "retrieve", "generate": "generate"})
_graph.add_edge("retrieve", "generate")
_graph.add_edge("generate", END)
_compiled = _graph.compile()


async def run_graph(state: AgentState) -> AgentState:
    result = await _compiled.ainvoke(state)
    return result
