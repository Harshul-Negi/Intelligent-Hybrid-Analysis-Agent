import pandas as pd

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from models.state import AgentState
from agents.router import router_node
from agents.analytics_agent import AnalyticsAgent
from agents.analytics_graph import create_analytics_graph
from services.analytics_service import AnalyticsService
from utils.dataset_pofiler import DatasetProfiler
from agents.rag_agent import RAGAgent
from agents.hybrid_agent import HybridAgent


def create_graph(df: pd.DataFrame):
    """
    Build a LangGraph workflow for the supplied dataset.

    This allows the Streamlit application to rebuild the
    analytics agent whenever the user uploads a new CSV.
    """

    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    # --------------------------------------------------
    # Dataset setup
    # --------------------------------------------------

    analytics_service = AnalyticsService(df)

    profiler = DatasetProfiler()
    schema = profiler.get_schema(df)

    analytics_agent = AnalyticsAgent(
        analytics=analytics_service,
        schema=schema
    )

    analytics_graph = create_analytics_graph(
        analytics_agent
    )

    # --------------------------------------------------
    # Other agents
    # --------------------------------------------------

    rag_agent = RAGAgent()
    hybrid_agent = HybridAgent()

    # --------------------------------------------------
    # Nodes
    # --------------------------------------------------

    def analytics_node(state: AgentState):

        result = analytics_graph.invoke({
            "messages": state["messages"]
        })

        response = result["messages"][-1]

        return {
            "messages": result["messages"],
            "analytics_result": response.content,
            "final_answer": response.content
        }

    def rag_node(state: AgentState):

        question = state["question"]

        result = rag_agent.answer(question)

        return {
            "messages": [
                AIMessage(content=result["answer"])
            ],
            "rag_result": result["answer"],
            "final_answer": result["answer"],
            "sources": result.get("sources", [])
        }

    def hybrid_analytics_node(state: AgentState):

        result = analytics_graph.invoke({
            "messages": state["messages"]
        })

        response = result["messages"][-1]

        return {
            "analytics_result": response.content
        }

    def hybrid_rag_node(state: AgentState):

        question = state["question"]

        result = rag_agent.answer(question)

        return {
            "rag_result": result["answer"],
            "sources": result.get("sources", [])
        }

    def synthesis_node(state: AgentState):

        final_answer = hybrid_agent.answer(
            question=state["question"],
            analytics_result=state["analytics_result"],
            rag_result=state["rag_result"]
        )

        return {
            "messages": [
                AIMessage(content=final_answer)
            ],
            "final_answer": final_answer
        }

    # --------------------------------------------------
    # Routing
    # --------------------------------------------------

    def route_after_router(state: AgentState):

        route = state["route"]

        if route == "ANALYTICS":
            return "analytics"

        elif route == "RAG":
            return "rag"

        elif route == "HYBRID":
            return "hybrid_analytics"

        else:
            raise ValueError(
                f"Unknown route: {route}"
            )

    # --------------------------------------------------
    # Build graph
    # --------------------------------------------------

    builder = StateGraph(AgentState)

    builder.add_node(
        "router",
        router_node
    )

    builder.add_node(
        "analytics",
        analytics_node
    )

    builder.add_node(
        "rag",
        rag_node
    )

    builder.add_node(
        "hybrid_analytics",
        hybrid_analytics_node
    )

    builder.add_node(
        "hybrid_rag",
        hybrid_rag_node
    )

    builder.add_node(
        "synthesis",
        synthesis_node
    )

    # --------------------------------------------------
    # Edges
    # --------------------------------------------------

    builder.add_edge(
        START,
        "router"
    )

    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "analytics": "analytics",
            "rag": "rag",
            "hybrid_analytics": "hybrid_analytics"
        }
    )

    builder.add_edge(
        "analytics",
        END
    )

    builder.add_edge(
        "rag",
        END
    )

    builder.add_edge(
        "hybrid_analytics",
        "hybrid_rag"
    )

    builder.add_edge(
        "hybrid_rag",
        "synthesis"
    )

    builder.add_edge(
        "synthesis",
        END
    )

    return builder.compile()


# --------------------------------------------------
# Default graph
# --------------------------------------------------

df = pd.read_csv("survey.csv")

graph = create_graph(df)