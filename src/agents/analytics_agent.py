from langchain_core.messages import SystemMessage

from llm import llm
from services.analytics_service import AnalyticsService
from services.analytics_tools import create_analytics_tools


class AnalyticsAgent:
    """LLM agent capable of answering analytical questions."""

    def __init__(
        self,
        analytics: AnalyticsService,
        schema: list
    ):
        self.analytics = analytics
        self.schema = schema

        self.tools = create_analytics_tools(
            analytics
        )

        self.llm = llm.bind_tools(
            self.tools
        )

    def _create_system_prompt(self) -> str:

        schema_lines = []

        for column in self.schema:
            schema_lines.append(
                f"- {column['name']} | "
                f"type={column['semantic_type']} | "
                f"dtype={column['dtype']} | "
                f"unique={column['unique_values']} | "
                f"samples={column['sample_values']}"
            )

        schema_text = "\n".join(schema_lines)

        return f"""
You are a data analytics assistant.

You answer questions about the provided dataset.

IMPORTANT RULES:

1. Use analytics tools for numerical calculations.
2. Never invent column names.
3. Only use columns that exist in the dataset schema.
4. Choose the column whose meaning best matches the user's question.
5. If the question is ambiguous, ask for clarification.
6. Do not calculate numerical results yourself when a tool can calculate them.
7. After receiving a tool result, explain it clearly to the user.

DATASET SCHEMA:

{schema_text}
"""

    def run(self, messages):

        system_message = SystemMessage(
            content=self._create_system_prompt()
        )

        response = self.llm.invoke(
            [system_message] + messages
        )

        return response