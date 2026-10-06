import os
import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

KNOWLEDGE_DIR = "data/knowledge_base"
VECTOR_DB_DIR = "vector_db"

COLLECTION_NAME = "cybersecurity_knowledge"


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# --------------------------------------------------
# 3. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# --------------------------------------------------
# 4. Read knowledge files
# --------------------------------------------------

documents = []
metadatas = []
ids = []

# Keep track of chunk numbers for each topic
source_counters = {}


for filename in sorted(os.listdir(KNOWLEDGE_DIR)):

    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(KNOWLEDGE_DIR, filename)

    # Use the filename to generate the ID prefix
    source_name = os.path.splitext(filename)[0]
    prefix = source_name[:2].lower()

    # Initialise numbering for this source
    source_counters[filename] = 1

    print(f"Reading: {filename}")
    print(f"ID prefix: {prefix}")

    # Read file
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    # Split into paragraphs
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    # Create chunk records
    for paragraph in paragraphs:

        chunk_index = source_counters[filename]

        # Example: ma001, ma002, ne001
        chunk_id = f"{prefix}{chunk_index:03d}"


        # Store the actual text
        documents.append(paragraph)
        ids.append(chunk_id)

        # Store only source and chunk_index
        metadatas.append({
            "source": filename,
            "chunk_index": chunk_index
        })

        source_counters[filename] += 1


# --------------------------------------------------
# 5. Create embeddings
# --------------------------------------------------

if not documents:
    raise ValueError(
        f"No text chunks found in {KNOWLEDGE_DIR}"
    )

print(f"\nCreating embeddings for {len(documents)} chunks...")

embeddings = model.encode(documents).tolist()


# --------------------------------------------------
# 6. Store chunks in ChromaDB
# --------------------------------------------------

# Prevent old records from remaining in the collection
client.delete_collection(name=COLLECTION_NAME)

collection = client.create_collection(
    name=COLLECTION_NAME
)

# Determine the maximum batch size supported by this ChromaDB/SQLite setup.
# ChromaDB typically caps this around 5461 due to SQLite variable limits.
try:
    MAX_BATCH_SIZE = client.get_max_batch_size()
except AttributeError:
    # Fallback for older ChromaDB versions that lack the method
    MAX_BATCH_SIZE = 5000

# Apply a safety margin to avoid edge-case failures
BATCH_SIZE = min(MAX_BATCH_SIZE, 5000)

print(f"\nChromaDB max batch size: {MAX_BATCH_SIZE}")
print(f"Using batch size: {BATCH_SIZE}")

# Insert records in smaller batches to stay under ChromaDB's limit
total = len(ids)
for start in range(0, total, BATCH_SIZE):
    end = start + BATCH_SIZE

    batch_ids = ids[start:end]
    batch_documents = documents[start:end]
    batch_embeddings = embeddings[start:end]
    batch_metadatas = metadatas[start:end]

    collection.add(
        ids=batch_ids,
        documents=batch_documents,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas
    )

    print(f"  Inserted batch {start}–{end - 1} "
          f"({len(batch_ids)} records)")


# --------------------------------------------------
# 7. Display result
# --------------------------------------------------

print("\nIngestion completed successfully.")

print(f"Chunks stored: {collection.count()}")

print("\nChunks by source:")

for filename, count in source_counters.items():
    print(f"  {filename}: {count - 1}")

print(f"\nVector database location: {VECTOR_DB_DIR}")