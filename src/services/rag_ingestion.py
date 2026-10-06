from utils.document_loader import DocumentLoader
from services.text_splitter import TextSplitter
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore


class RAGIngestion:

    def __init__(self):

        self.loader = DocumentLoader()

        self.splitter = TextSplitter()

        self.embedding_service = (
            EmbeddingService()
        )

        self.vector_store = VectorStore(
            self.embedding_service.get_embeddings()
        )

    def ingest(self, file_path: str):

        print("Loading document...")

        documents = self.loader.load(
            file_path
        )

        print(
            f"Loaded {len(documents)} documents."
        )

        print("Splitting document...")

        chunks = self.splitter.split(
            documents
        )

        print(
            f"Created {len(chunks)} chunks."
        )

        print("Creating embeddings and storing...")

        self.vector_store.add_documents(
            chunks
        )

        print("Ingestion complete.")