from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from config import RERANKER_THRESHOLD

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ---------------------------------------------------------------------------
# Singleton — loaded once at startup, reused on every request
# ---------------------------------------------------------------------------
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Return the shared CrossEncoder instance, loading it once at first call."""
    global _reranker
    if _reranker is None:
        print(f"⏳ Loading reranker model '{RERANKER_MODEL}' (once)...")
        _reranker = CrossEncoder(RERANKER_MODEL)
        print("✅ Reranker model ready.")
    return _reranker


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------

def rerank(query: str, candidates: list[Document], threshold: float = RERANKER_THRESHOLD) -> list[tuple]:
    """
    Score each candidate against the query using the cross-encoder and return
    only those that meet the relevance threshold, sorted best-first.

    Args:
        query:      The user's question.
        candidates: Documents returned by the vector search.
        threshold:  Minimum reranker score to pass (higher = stricter).

    Returns:
        Filtered and reordered list of Documents.
    """
    if not candidates:
        return []

    reranker = get_reranker()

    # Build (query, passage) pairs — the input format CrossEncoder expects
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)  # returns a numpy array of floats

    # Sort best-first; return (score, doc) so the caller can log and filter
    return sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)