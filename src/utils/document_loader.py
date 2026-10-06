from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)


class DocumentLoader:
    """Load supported documents for the RAG pipeline."""

    def load(self, file_path: str):

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        extension = path.suffix.lower()

        if extension == ".pdf":
            loader = PyPDFLoader(
                str(path)
            )

        elif extension == ".txt":
            loader = TextLoader(
                str(path),
                encoding="utf-8"
            )

        else:
            raise ValueError(
                f"Unsupported document type: {extension}"
            )

        return loader.load()