from langchain_core.messages import HumanMessage, SystemMessage

from llm import llm
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore


class RAGAgent:
    """Agent that answers questions using retrieved documents."""

    def __init__(self):

        embedding_service = EmbeddingService()

        vector_store = VectorStore(
            embedding_service.get_embeddings()
        )

        self.retriever = vector_store.get_retriever(
            k=4
        )

        self.llm = llm

    def retrieve(self, question: str):

        return self.retriever.invoke(
            question
        )

    def answer(self, question: str):

        documents = self.retrieve(
            question
        )

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        system_prompt = """
You are a knowledgeable assistant answering
questions about the provided dataset documentation.

Use ONLY the supplied context to answer.

Rules:

1. Do not invent information.
2. If the answer cannot be found in the context,
   say that the available documentation does not
   contain enough information.
3. Give a clear and concise answer.
4. Treat the PEI Index as an expert-perception measure,
   not as a direct measurement of election fraud.
5. Do not perform numerical calculations from the
   documentation. Those should be handled by the
   Analytics Agent.

CONTEXT:

{context}
"""

        messages = [
            SystemMessage(
                content=system_prompt.format(
                    context=context
                )
            ),
            HumanMessage(
                content=question
            )
        ]

        response = self.llm.invoke(
            messages
        )

        sources = []

        for document in documents:

            source = document.metadata.get(
                "source"
            )

            page = document.metadata.get(
                "page"
            )

            source_info = {
                "source": source,
                "page": page
            }

            if source_info not in sources:
                sources.append(
                    source_info
                )

        return {
            "answer": response.content,
            "sources": sources
        }