from __future__ import annotations
import structlog
import chromadb
from chromadb.utils import embedding_functions
from app.config import get_settings

logger = structlog.get_logger(__name__)

settings = get_settings()

COLLECTION_NAME = "medical_journals"

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_embedding_function() -> embedding_functions.DefaultEmbeddingFunction:
    return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_path)
    return _client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_chroma_client()
        ef = _get_embedding_function()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_documents(
    documents: list[str],
    metadatas: list[dict],
    ids: list[str] | None = None,
) -> None:
    """Add medical journal documents to the vector store."""
    import uuid as _uuid
    collection = get_collection()
    if ids is None:
        ids = [str(_uuid.uuid4()) for _ in documents]
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    logger.info("chroma_documents_added", count=len(documents))


def query_documents(
    query_text: str,
    n_results: int = 5,
) -> list[dict]:
    """
    Query the vector store for relevant medical journal chunks.
    Returns a list of dicts with keys: text, metadata, distance.
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        logger.warning("chroma_empty_collection", message="No documents in ChromaDB; skipping retrieval")
        return []

    n_results = min(n_results, count)
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        output.append(
            {
                "text": doc,
                "metadata": meta or {},
                "distance": dist,
            }
        )

    logger.info("chroma_query", query_length=len(query_text), n_results=len(output))
    return output
