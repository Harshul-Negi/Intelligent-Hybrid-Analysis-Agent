from pathlib import Path

from langchain_chroma import Chroma


class VectorStore:
    """Manage the Chroma vector database."""

    def __init__(
        self,
        embedding_function,
        persist_directory: str = "vector_store"
    ):

        self.persist_directory = Path(
            persist_directory
        )

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.vectorstore = Chroma(
            collection_name="election_knowledge",
            embedding_function=embedding_function,
            persist_directory=str(
                self.persist_directory
            )
        )

    def add_documents(self, documents):

        self.vectorstore.add_documents(
            documents
        )

    def get_retriever(
        self,
        k: int = 4
    ):

        return self.vectorstore.as_retriever(
            search_kwargs={
                "k": k
            }
        )