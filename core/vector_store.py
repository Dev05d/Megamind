import uuid
import ollama
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

from core.config import DB_PATH, COLLECTION_NAME, VECTOR_SIZE, EMBED_MODEL
from core.chunker import chunk_text

# Lazy singleton client
_client: QdrantClient | None = None

# Initialize BM25 Sparse Embedder (downloads a tiny ~30MB keyword map on first run)
_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=DB_PATH)
    return _client

def init_db() -> None:
    """Creates the Qdrant Hybrid Collection if it does not exist."""
    client = _get_client()
    
    # If it exists, we are good to go! No complex checking or deleting needed.
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            }
        )
        print(f"[DB] Created fresh Hybrid collection '{COLLECTION_NAME}'.")
    else:
        print(f"[DB] Hybrid collection '{COLLECTION_NAME}' loaded and ready.")

def save_chunks(points: list[dict]) -> None:
    client = _get_client()
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[models.PointStruct(**p) for p in points]
    )

def ingest_text(text: str, source: str, extra_payload: dict | None = None) -> None:
    extra_payload = extra_payload or {}
    chunks = chunk_text(text)

    if not chunks:
        print(f"[Warning] No content extracted from '{source}'. Nothing saved.")
        return

    print(f"[Embedding] Generating Dense & Sparse vectors for {len(chunks)} chunks...")
    points = []

    for i, chunk in enumerate(chunks):
        # 1. Dense Vector (Semantic Meaning via Ollama)
        dense_response = ollama.embed(model=EMBED_MODEL, input=chunk)
        dense_vec = dense_response["embeddings"][0]

        # 2. Sparse Vector (Keyword Indexing via FastEmbed BM25)
        sparse_gen = list(_sparse_model.embed([chunk]))[0]
        sparse_vec = models.SparseVector(
            indices=sparse_gen.indices.tolist(),
            values=sparse_gen.values.tolist()
        )

        unique_string = f"{source}_{i}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

        points.append({
            "id": point_id,
            "vector": {
                "dense": dense_vec,   # Semantic mapping
                "sparse": sparse_vec  # Keyword mapping
            },
            "payload": {
                "text": chunk, 
                "source": source, 
                "chunk_index": i,     # <-- Added so you know the exact order!
                **extra_payload
            },
        })

    save_chunks(points)
    print(f"[DB] Saved {len(points)} hybrid vectors from '{source}'.")

def close_db() -> None:
    if _client is not None:
        _client.close()