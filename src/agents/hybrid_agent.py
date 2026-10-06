from langchain_core.messages import HumanMessage, SystemMessage

from llm import llm


class HybridAgent:
    """Combine analytics and RAG results into one answer."""

    def __init__(self):
        self.llm = llm

    def answer(
        self,
        question: str,
        analytics_result: str,
        rag_result: str
    ):

        system_prompt = """
You are a synthesis agent in a hybrid data analysis
and document question-answering system.

You will receive:

1. The user's question.
2. A result produced by the Analytics Agent.
3. A result produced by the RAG Agent.

Your job is to combine these results into one clear,
accurate answer.

IMPORTANT RULES:

1. Do not invent information.
2. Do not change numerical results produced by the
   Analytics Agent.
3. Use the Analytics result for numerical/data claims.
4. Use the RAG result for definitions, methodology,
   interpretation, and documentation.
5. If either result is missing information, say so.
6. Do not perform new numerical calculations yourself.
7. Clearly connect the analytical result with the
   documentation/context.
"""

        user_prompt = f"""
USER QUESTION:

{question}


ANALYTICS RESULT:

{analytics_result}


RAG RESULT:

{rag_result}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)

        return response.content