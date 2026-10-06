from services.rag_ingestion import RAGIngestion


ingestion = RAGIngestion()

ingestion.ingest(
    "documents\PEI_US_2016_Dataset_Detailed_Guide.pdf"
)