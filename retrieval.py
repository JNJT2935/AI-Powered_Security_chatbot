import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Configuration
# ============================================================

DB_PATH = "vector_db"
COLLECTION_NAME = "cybersecurity_knowledge"

# IMPORTANT:
# This MUST be the same embedding model used in ingestion.py.
MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 7


# ============================================================
# 2. Connect to ChromaDB
# ============================================================

def get_collection():
    """Connect to ChromaDB and return the knowledge collection."""

    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not open ChromaDB collection "
            f"'{COLLECTION_NAME}': {e}"
        ) from e

    return collection


# ============================================================
# 3. Load embedding model
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# ============================================================
# 4. Retrieve relevant chunks
# ============================================================

def retrieve(query, top_k=DEFAULT_TOP_K):
    """
    Retrieve the top-k most relevant chunks from ChromaDB.

    Returns a list of dictionaries containing:
        chunk_id
        text
        source
        chunk_index
        distance
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # --------------------------------------------------------
    # Connect to collection
    # --------------------------------------------------------

    collection = get_collection()

    # --------------------------------------------------------
    # Check whether the collection contains data
    # --------------------------------------------------------

    collection_count = collection.count()

    if collection_count == 0:
        return []

    # --------------------------------------------------------
    # Prevent requesting more chunks than exist
    # --------------------------------------------------------

    top_k = min(top_k, collection_count)

    # Convert the question into an embedding.
    query_embedding = model.encode(
        query.strip()
    ).tolist()

    # Retrieve the nearest chunks.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_chunks = []

    for i, chunk_id in enumerate(ids):
        metadata = metadatas[i] or {}

        chunk = {
            "chunk_id": chunk_id,
            "text": documents[i],
            "source": metadata.get("source", "Unknown"),
            "chunk_index": metadata.get("chunk_index", "Unknown"),
            "distance": distances[i]
        }

        retrieved_chunks.append(chunk)

    return retrieved_chunks


# ============================================================
# 5. Display retrieval results
# ============================================================

def display_results(query, results):
    """Display retrieved chunks in a readable format."""

    print("\n" + "=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)

    print(f"\nQuery: {query}")

    if not results:
        print("\nNo chunks were found.")
        return

    print(f"\nRetrieved {len(results)} chunk(s):")

    for rank, result in enumerate(results, start=1):
        print("\n" + "-" * 80)
        print(f"Rank:             {rank}")
        print(f"Chunk ID:         {result['chunk_id']}")
        print(f"Source:           {result['source']}")
        print(f"Chunk Index:      {result['chunk_index']}")
        print(f"Distance:         {result['distance']:.4f}")
        print("\nText:")
        print(result["text"])

    print("\n" + "=" * 80)


# ============================================================
# 6. Main program
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("CYBERSECURITY KNOWLEDGE RETRIEVAL")
    print("=" * 80)

    query = input("\nEnter your question: ").strip()

    if not query:
        print("\nError: Please enter a question.")
        return

    top_k_input = input(
        f"Number of chunks to retrieve "
        f"[default: {DEFAULT_TOP_K}]: "
    ).strip()

    if top_k_input:
        try:
            top_k = int(top_k_input)

            if top_k <= 0:
                print("\nError: Number of chunks must be greater than 0.")
                return

        except ValueError:
            print("\nError: Please enter a valid integer.")
            return
    else:
        top_k = DEFAULT_TOP_K

    try:
        results = retrieve(query, top_k)
        display_results(query, results)

    except Exception as e:
        print("\nERROR:")
        print(e)


if __name__ == "__main__":
    main()