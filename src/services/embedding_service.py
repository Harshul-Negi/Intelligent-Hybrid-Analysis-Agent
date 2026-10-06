from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingService:
    """Create embeddings for documents and queries."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def get_embeddings(self):
        return self.embeddings