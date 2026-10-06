from llm import llm
from models.state import AgentState

from langchain_core.messages import HumanMessage, SystemMessage


def router_node(state: AgentState):

    question = state["question"]

    system_prompt = """
You are a routing agent for a hybrid data analysis and
document question-answering system.

Your job is to classify the user's question into exactly
one of these three categories:

ANALYTICS
Use ANALYTICS when the question requires calculations,
statistics, filtering, sorting, aggregation, comparisons,
or information directly from the CSV dataset.

Examples:
- What is the average National Electoral Integrity?
- Which state has the highest PEI score?
- How many experts were from California?
- What is the average score by state?

RAG
Use RAG when the question asks about concepts, definitions,
methodology, documentation, or background information
contained in the documentation.

Examples:
- What does the PEI Index measure?
- What is the Electoral Integrity Project?
- How is the PEI Index constructed?
- What are the stages of the electoral cycle?

HYBRID
Use HYBRID when the question requires BOTH:
1. numerical/data analysis from the CSV
AND
2. conceptual/contextual information from the documentation.

Examples:
- Which state has the highest PEI score and what does that score mean?
- What is the average PEI score and how should it be interpreted?
- Which states performed best and what does the PEI methodology say about the score?

Return ONLY one word:

ANALYTICS
RAG
HYBRID
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)

    route = response.content.strip().upper()

    if route not in ["ANALYTICS", "RAG", "HYBRID"]:
        route = "RAG"

    return {
        "route": route
    }