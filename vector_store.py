# AFTER:
import os
import shutil
import time                                           # ← add this
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import DATA_PATH, CHROMA_PATH, RETRIEVAL_THRESHOLD, EMBEDDING_MODEL, STORAGE_PROVIDER, RETRIEVAL_K
from dotenv import load_dotenv
from minio import Minio
from storage.minio_client import get_minio_client

load_dotenv()

# ---------------------------------------------------------------------------
# Singletons — loaded once at startup, reused on every request
# ---------------------------------------------------------------------------
print("⏳ Loading embedding model (once)...")
_t = time.perf_counter()
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print(f"✅ Embedding model ready — {time.perf_counter() - _t:.2f}s")

_vector_store: Chroma | None = None


def load_vector_store() -> Chroma:
    """Return the shared Chroma instance, opening it once and reusing it."""
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=_embeddings,
        )
    return _vector_store


# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------

def load_documents_from_local():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def load_documents_from_minio() -> list[Document]:
    """
    Downloads all objects from the configured MinIO bucket and returns them
    as a list of LangChain Documents.

    Each Document contains:
        - page_content : UTF-8 decoded file content
        - metadata     : object_name, bucket_name, etag, last_modified
    """
    bucket_name = os.getenv("MINIO_BUCKET")

    client = get_minio_client()

    documents: list[Document] = []
    objects = client.list_objects(bucket_name, recursive=True)

    for obj in objects:
        response = client.get_object(bucket_name, obj.object_name)
        try:
            raw_bytes = response.read()
        finally:
            response.close()
            response.release_conn()

        try:
            content = raw_bytes.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            print(f"  ⚠ Skipping non-UTF-8 object: {obj.object_name}")
            continue

        doc = Document(
            page_content=content,
            metadata={
                "object_name":   obj.object_name,
                "bucket_name":   bucket_name,
                "etag":          obj.etag,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
            },
        )
        documents.append(doc)
        print(f"  ✅ Loaded: {obj.object_name} ({len(content)} chars)")

    print(f"\nLoaded {len(documents)} document(s) from MinIO bucket '{bucket_name}'.")
    return documents


def load_document_from_minio(object_name: str) -> Document:
    """
    Download a single object from MinIO and return it as a LangChain Document.

    Args:
        object_name: The full object key to download from the bucket.

    Returns:
        A Document with UTF-8 decoded content and MinIO metadata.

    Raises:
        RuntimeError: If the object cannot be downloaded or decoded.
    """
    bucket_name = os.getenv("MINIO_BUCKET")
    client = get_minio_client()

    try:
        response = client.get_object(bucket_name, object_name)
        try:
            raw_bytes = response.read()
        finally:
            response.close()
            response.release_conn()
    except Exception as e:
        raise RuntimeError(
            f"Failed to download '{object_name}' from bucket '{bucket_name}': {e}"
        ) from e

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError(f"Failed to decode '{object_name}' as UTF-8: {e}") from e

    stat = client.stat_object(bucket_name, object_name)

    return Document(
        page_content=content,
        metadata={
            "object_name":   object_name,
            "bucket_name":   bucket_name,
            "etag":          stat.etag,
            "last_modified": stat.last_modified.isoformat() if stat.last_modified else None,
        },
    )

def load_documents():
    if STORAGE_PROVIDER == 'local':
        return load_documents_from_local()
    elif STORAGE_PROVIDER == 'minio':
        return load_documents_from_minio()
    else:
        raise ValueError(f"Unknown STORAGE_PROVIDER: '{STORAGE_PROVIDER}'. Expected 'local' or 'minio'.")


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def split_text(documents):
    # Step 1: Split on Markdown headers first so each chunk respects document
    # structure (sections don't bleed across headings).
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "h1"),
            ("##",  "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,  # keep the heading text inside the chunk content
    )
 
    # Step 2: If a section is still too large after header splitting, cut it
    # further with the recursive splitter.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
 
    chunks = []
    for doc in documents:
        # MarkdownHeaderTextSplitter works on raw text, not Documents, so we
        # pass page_content and then restore the original metadata on every
        # resulting section.
        sections = header_splitter.split_text(doc.page_content)
 
        for section in sections:
            # Merge original document metadata with any header metadata added
            # by the splitter (h1, h2, h3 keys), giving priority to the
            # original so object_name / source / etag are never overwritten.
            section.metadata = {**section.metadata, **doc.metadata}
 
        # Apply recursive splitting to each section; metadata is propagated
        # automatically by split_documents.
        chunks.extend(recursive_splitter.split_documents(sections))
    for i, chunk in enumerate(chunks):
        print("=" * 80)
        print(f"Chunk {i}")
        print(chunk.metadata)
        print(chunk.page_content)
 
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks: list[Document]):
    global _vector_store

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Reset singleton so next load_vector_store() picks up the new DB
    _vector_store = None

    db = Chroma.from_documents(
        chunks,
        _embeddings,           # reuse the singleton — no reload
        persist_directory=CHROMA_PATH,
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")




def add_documents(documents: list[Document]) -> None:
    """
    Split the provided documents into chunks and add them to the existing
    Chroma database without deleting any existing data.
    """
    chunks = split_text(documents)

    db = load_vector_store()
    db.add_documents(chunks)
    db.persist()

    print(f"Added {len(chunks)} chunk(s) to the existing vector store.")



def delete_documents_by_object_name(object_name: str) -> None:
    """
    Delete all chunks belonging to a given object from the Chroma vector store.
    Matches on metadata field 'object_name'.
    """
    db = load_vector_store()

    db._collection.delete(where={"object_name": object_name})
    db.persist()

    print(f"Deleted all chunks for '{object_name}' from the vector store.")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def get_docs_by_distance(question: str, k: int = RETRIEVAL_K, threshold: float = RETRIEVAL_THRESHOLD):
    db = load_vector_store()

    # Chroma raises or returns Documents with page_content=None when the
    # collection is empty. Guard against this before querying.
    if db._collection.count() == 0:
        return []

    docs_with_scores = db.similarity_search_with_score(question, k=k)
    return [(doc, distance) for doc, distance in docs_with_scores if distance <= threshold]