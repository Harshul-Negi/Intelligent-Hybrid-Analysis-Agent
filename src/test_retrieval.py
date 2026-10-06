from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore


embedding_service = EmbeddingService()

vector_store = VectorStore(
    embedding_service.get_embeddings()
)

retriever = vector_store.get_retriever(
    k=4
)


question = (
    "What does the Electoral Integrity Index measure?"
)

documents = retriever.invoke(
    question
)


print("===== RETRIEVED DOCUMENTS =====")

for i, document in enumerate(
    documents,
    start=1
):

    print(f"\n--- Document {i} ---")

    print(
        document.page_content[:1000]
    )

    print(
        "Metadata:",
        document.metadata
    )