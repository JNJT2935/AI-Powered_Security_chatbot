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

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)


# --------------------------------------------------
# 7. Display result
# --------------------------------------------------

print("\nIngestion completed successfully.")

print(f"Chunks stored: {collection.count()}")

print("\nChunks by source:")

for filename, count in source_counters.items():
    print(f"  {filename}: {count - 1}")

print(f"\nVector database location: {VECTOR_DB_DIR}")