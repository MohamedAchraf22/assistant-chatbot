import time
from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm
from vector_store import get_docs_by_distance
from reranker import rerank
from config import RERANKER_THRESHOLD, RERANKER_TOP_N

# ---------------------------------------------------------------------------
# Prompt: query contextualization
# ---------------------------------------------------------------------------
CONTEXTUALIZE_PROMPT = """You are a query rewriter for a retrieval system.

Given the recent conversation history and the user's current question, rewrite
the question into a single self-contained search query that includes all
necessary context from the conversation.

Rules:
- Output ONLY the rewritten query. No explanation, no preamble, no quotes.
- Do NOT answer the question.
- If the question is already fully self-contained, return it unchanged.
- If the question references something from history (e.g. "those teams", "he",
  "that tournament"), resolve the reference and make it explicit.

Conversation history:
{history}

Current question: {question}

Standalone retrieval query:"""

CONTEXTUALIZE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("human", CONTEXTUALIZE_PROMPT),
])

# ---------------------------------------------------------------------------
# Prompt: answer generation
# ---------------------------------------------------------------------------
ANSWER_PROMPT = """You are a specialized assistant.

Rules:
- Answer ONLY using the Context from documents provided below.
- If the context does not contain enough information to answer the resolved question,
  reply EXACTLY with: "That's not my specialization."
- Do not use general knowledge.
- Use the resolved question to understand exactly what the user is asking.
- Use the original question and conversation history only to preserve conversational context.
- Answer naturally and directly.

Recent conversation (for reference only):
{history}

Original user question:
{original_question}

Resolved question:
{resolved_question}

Context from documents:
{context}
"""

ANSWER_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", ANSWER_PROMPT),
    ("human", "Answer the user's question."),
])


# ---------------------------------------------------------------------------
# LLM singleton
# ---------------------------------------------------------------------------
_llm = None


def get_cached_llm():
    global _llm
    if _llm is None:
        print("⏳ Loading LLM (once)...")
        _t = time.perf_counter()
        _llm = get_llm()
        print(f"✅ LLM ready — {time.perf_counter() - _t:.2f}s")
    return _llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def _preview(text: str, limit: int = 100) -> str:
    return (text.replace("\n", " ").strip()[:limit] + "..."
            if len(text) > limit else text.replace("\n", " ").strip())


# ---------------------------------------------------------------------------
# Query contextualization
# ---------------------------------------------------------------------------

def contextualize_query(question: str, history: list[dict], llm) -> str:
    """
    If there is no history, return the question unchanged immediately —
    avoids an unnecessary LLM call for the common case.
    """
    if not history:
        return question

    prompt = CONTEXTUALIZE_TEMPLATE.invoke({
        "history": _format_history(history),
        "question": question,
    })
    raw = llm.invoke(prompt).content
    rewritten = "".join(
        item.get("text", str(item)) if isinstance(item, dict) else str(item)
        for item in raw
    ).strip() if isinstance(raw, list) else (raw or "").strip()

    return rewritten if rewritten else question


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_rag_chain():
    _t = time.perf_counter()
    llm = get_cached_llm()
    print(f"✅ RAG pipeline initialised — {time.perf_counter() - _t:.2f}s")

    def full_pipeline(question: str, history: list[dict]):
        t_total = time.perf_counter()
        history_str = _format_history(history)

        # ------------------------------------------------------------------
        # 1. Contextualize
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        standalone_query = contextualize_query(question, history, llm)
        t_contextualize = time.perf_counter() - t0

        print(f"\n[DEBUG] Question  : {question}")
        if standalone_query != question:
            print(f"[DEBUG] Rewritten : {standalone_query}")

        # ------------------------------------------------------------------
        # 2. Vector retrieval
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        vector_results = get_docs_by_distance(standalone_query)
        candidates = [doc for doc, _ in vector_results]
        t_vector = time.perf_counter() - t0

        print(f"\nVECTOR SEARCH  ({len(candidates)} candidates, threshold applied)")

        # ------------------------------------------------------------------
        # 3. Rerank
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        scored = rerank(standalone_query, candidates)
        above_threshold = [(score, doc) for score, doc in scored
                           if score >= RERANKER_THRESHOLD]
        final_docs = [doc for _, doc in above_threshold[:RERANKER_TOP_N]]
        t_rerank = time.perf_counter() - t0

        print(f"\nRERANKER  (threshold={RERANKER_THRESHOLD}, top_n={RERANKER_TOP_N})")
        for i, (score, doc) in enumerate(scored, 1):
            if i > max(len(above_threshold), RERANKER_TOP_N) + 2:
                break
            selected = (score, doc) in above_threshold[:RERANKER_TOP_N]
            label = "✅" if selected else "❌"
            print(f"  {i}. score={score:.4f} {label} | {_preview(doc.page_content)}")

        print(f"\nSUMMARY")
        print(f"  Vector candidates : {len(candidates)}")
        print(f"  Above threshold   : {len(above_threshold)}")
        print(f"  Sent to LLM       : {len(final_docs)}")

        # ------------------------------------------------------------------
        # 4. Prompt building
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        context = "\n\n---\n\n".join(doc.page_content for doc in final_docs) if final_docs else ""
        prompt = ANSWER_TEMPLATE.invoke({
        "original_question": question,
        "resolved_question": standalone_query,
        "context": context,
        "history": history_str,
        })
        t_prompt = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # 5. Final LLM generation
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        answer = llm.invoke(prompt).content
        t_llm = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total

        # ------------------------------------------------------------------
        # Performance summary
        # ------------------------------------------------------------------
        print("\n" + "─" * 40)
        print("  PERFORMANCE SUMMARY")
        print("─" * 40)
        print(f"  Contextualization : {t_contextualize:.2f}s")
        print(f"  Vector Search     : {t_vector:.2f}s")
        print(f"  Reranking         : {t_rerank:.2f}s")
        print(f"  Prompt Building   : {t_prompt:.2f}s")
        print(f"  Final LLM         : {t_llm:.2f}s")
        print("─" * 40)
        print(f"  Total Pipeline    : {t_total:.2f}s")
        print("─" * 40 + "\n")

        print(f"\nANSWER\n  {answer}\n")
        return answer

    return full_pipeline


def ask_question(question: str, history: list[dict] = None) -> str:
    rag_chain = build_rag_chain()
    return rag_chain(question, history or [])