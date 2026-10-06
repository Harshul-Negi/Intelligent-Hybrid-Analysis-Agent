from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from models.state import AgentState
from agents.analytics_agent import AnalyticsAgent


def create_analytics_graph(
    analytics_agent: AnalyticsAgent
):

    tools = analytics_agent.tools

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    def agent_node(state):
        response = analytics_agent.run(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    graph.add_node(
        "analytics_agent",
        agent_node
    )

    graph.add_node(
        "tools",
        tool_node
    )

    graph.add_edge(
        START,
        "analytics_agent"
    )

    graph.add_conditional_edges(
        "analytics_agent",
        lambda state: (
            "tools"
            if state["messages"][-1].tool_calls
            else END
        )
    )

    graph.add_edge(
        "tools",
        "analytics_agent"
    )

    return graph.compile()