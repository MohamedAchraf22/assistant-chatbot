"""
Diagnostic script — run from the project root.
Inspects the Chroma collection for common_obsessions_compulsions.md
and explains why similarity search may not return it.

Usage:
    python diagnose_chroma.py
"""

import sys

# ── 1. Load the collection ───────────────────────────────────────────────────
from vector_store import load_vector_store, _embeddings

db = load_vector_store()
collection = db._collection
total = collection.count()
print(f"\n{'='*60}")
print(f"Total chunks in Chroma collection: {total}")
print(f"{'='*60}\n")

if total == 0:
    print("❌ Collection is empty. Re-run ingestion first.")
    sys.exit(1)

# ── 2. Search for chunks from the target document ────────────────────────────
TARGET = "common_obsessions_compulsions.md"

all_data = collection.get(include=["metadatas", "documents", "embeddings"])

ids        = all_data["ids"]
metadatas  = all_data["metadatas"]
documents  = all_data["documents"]
embeddings = all_data["embeddings"]

# Match on any metadata field that could reference the file
matching_indices = [
    i for i, m in enumerate(metadatas)
    if TARGET in str(m.get("object_name", ""))
    or TARGET in str(m.get("source", ""))
    or TARGET in str(m.get("chunk_id", ""))
]

print(f"Chunks found for '{TARGET}': {len(matching_indices)}")
print()

if not matching_indices:
    print("❌ No chunks found for this document in the collection.")
    print()
    print("── All unique sources/object_names in the collection ──")
    seen = set()
    for m in metadatas:
        key = m.get("object_name") or m.get("source") or m.get("chunk_id", "UNKNOWN")
        seen.add(key)
    for s in sorted(seen):
        print(f"  {s}")
    sys.exit(0)

# ── 3. Print chunk details ────────────────────────────────────────────────────
for rank, i in enumerate(matching_indices):
    print(f"── Chunk {rank} ──────────────────────────────────────────")
    print(f"  ID       : {ids[i]}")
    print(f"  Metadata : {metadatas[i]}")
    content = documents[i] or ""
    print(f"  Content  : {repr(content[:100])}")
    has_embedding = embeddings is not None and embeddings[i] is not None
    print(f"  Embedding: {'✅ present' if has_embedding else '❌ MISSING'}")
    print()

# ── 4. Run similarity search and check if target appears ─────────────────────
QUERY = "what is the Common Types of Compulsions?"
print(f"{'='*60}")
print(f"Similarity search (k=50): '{QUERY}'")
print(f"{'='*60}\n")

results = db.similarity_search_with_score(QUERY, k=50)

target_in_results = False
for rank, (doc, score) in enumerate(results):
    m = doc.metadata
    is_target = (
        TARGET in str(m.get("object_name", ""))
        or TARGET in str(m.get("source", ""))
        or TARGET in str(m.get("chunk_id", ""))
    )
    if is_target:
        target_in_results = True
        print(f"✅ Found at rank {rank+1} — score: {score:.4f}")
        print(f"   Metadata : {m}")
        print(f"   Content  : {repr(doc.page_content[:100])}")

if not target_in_results:
    print(f"❌ None of the top-50 results are from '{TARGET}'.")
    print()
    print("── Top 5 results actually returned ──")
    for rank, (doc, score) in enumerate(results[:5]):
        print(f"  #{rank+1}  score={score:.4f}  meta={doc.metadata}")
        print(f"       {repr(doc.page_content[:80])}")

# ── 5. Direct embedding distance check ───────────────────────────────────────
if matching_indices and embeddings is not None:
    import numpy as np

    print(f"\n{'='*60}")
    print("Direct embedding distance check")
    print(f"{'='*60}\n")

    query_embedding = _embeddings.embed_query(QUERY)
    query_vec = np.array(query_embedding)

    for rank, i in enumerate(matching_indices):
        chunk_vec = np.array(embeddings[i])
        # L2 distance (same metric Chroma uses by default)
        l2 = float(np.linalg.norm(query_vec - chunk_vec))
        cosine = float(
            np.dot(query_vec, chunk_vec)
            / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec))
        )
        print(f"  Chunk {rank} — L2: {l2:.4f}  Cosine similarity: {cosine:.4f}")
        print(f"  Content: {repr((documents[i] or '')[:80])}")
        print()