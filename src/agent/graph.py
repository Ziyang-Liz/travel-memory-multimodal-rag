from typing import List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from src.agent.router import route_query
from src.agent.answer import generate_answer
from src.retrieval.retrievers import TravelRetrievers
from src.retrieval.vector_store import SearchResult

load_dotenv()


class AgentState(TypedDict):
    question: str
    history: str
    route: str
    evidence: List[SearchResult]
    answer: str
    verification: str


retrievers = TravelRetrievers()


def router_node(state: AgentState) -> AgentState:
    route = route_query(state["question"], state.get("history", ""))
    return {**state, "route": route}


def retrieval_node(state: AgentState) -> AgentState:
    question = state["question"]
    route = state["route"]

    if route == "cross_modal":
        image_results = retrievers.image_caption_retriever(question, top_k=3)
        memory_results = retrievers.memory_metadata_retriever(question, top_k=2)
        evidence = image_results + memory_results
    elif route == "multi_hop":
        evidence = retrievers.hybrid_retriever(question, top_k=8)
    elif route == "follow_up":
        combined_query = state.get("history", "") + "\n" + question
        evidence = retrievers.hybrid_retriever(combined_query, top_k=8)
    else:
        text_results = retrievers.text_retriever(question, top_k=4)
        memory_results = retrievers.memory_metadata_retriever(question, top_k=3)
        evidence = text_results + memory_results

    by_id = {}
    for item in evidence:
        if item.doc_id not in by_id or item.score > by_id[item.doc_id].score:
            by_id[item.doc_id] = item
    evidence = sorted(by_id.values(), key=lambda x: x.score, reverse=True)[:8]

    return {**state, "evidence": evidence}


def answer_node(state: AgentState) -> AgentState:
    answer = generate_answer(
        question=state["question"],
        evidence=state.get("evidence", []),
        route=state["route"],
        history=state.get("history", ""),
    )
    return {**state, "answer": answer}


def verification_node(state: AgentState) -> AgentState:
    evidence = state.get("evidence", [])
    if not evidence:
        verification = "No retrieved evidence. Answer may be unsupported."
    else:
        top_ids = ", ".join([e.doc_id for e in evidence[:3]])
        verification = f"Grounding check: answer generated using retrieved evidence. Top evidence: {top_ids}."
    return {**state, "verification": verification}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("answer", answer_node)
    graph.add_node("verify", verification_node)

    graph.add_edge(START, "router")
    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


travel_agent_graph = None


def ask_agent(question: str, history: str = "") -> dict:
    global travel_agent_graph
    if travel_agent_graph is None:
        travel_agent_graph = build_graph()

    result = travel_agent_graph.invoke({
        "question": question,
        "history": history,
        "route": "",
        "evidence": [],
        "answer": "",
        "verification": "",
    })
    return result
