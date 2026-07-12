from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm
from vector_store import get_docs_by_distance
from reranker import rerank
from config import RERANKER_THRESHOLD, RERANKER_TOP_N

# ---------------------------------------------------------------------------
# Prompt: query contextualization
# Rewrites the user question into a standalone retrieval query using recent
# session history. Strictly rewrites — never answers.
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
# Uses the original user question and the retrieved context chunks.
# History is included only as readable background — not as few-shot examples.
# ---------------------------------------------------------------------------
ANSWER_PROMPT = """You are a specialized assistant.

Rules:
- Answer ONLY using the Context from documents provided below.
- If the context does not contain enough information to answer, reply EXACTLY
  with: "That's not my specialization."
- Do not use general knowledge.
- Do not be influenced by the conversation history when deciding whether to
  answer — use it only to understand what the user means.

Recent conversation (for reference only):
{history}

Context from documents:
{context}

Now answer the following question:"""

ANSWER_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", ANSWER_PROMPT),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_history(history: list[dict]) -> str:
    """Convert [{role, content}, ...] list to a readable string for prompts."""
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
    if not history:
        return question

    prompt = CONTEXTUALIZE_TEMPLATE.invoke({
        "history": _format_history(history),
        "question": question,
    })
    
    raw = llm.invoke(prompt).content
    if isinstance(raw, list):
        rewritten = "".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in raw
        ).strip()
    else:
        rewritten = (raw or "").strip()

    return rewritten if rewritten else question


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_rag_chain():
    llm = get_llm()

    def full_pipeline(question: str, history: list[dict]):
        history_str = _format_history(history)

        # ------------------------------------------------------------------
        # 1. Contextualize the query using recent session history
        # ------------------------------------------------------------------
        standalone_query = contextualize_query(question, history, llm)

        print(f"\n[DEBUG] Question  : {question}")
        if standalone_query != question:
            print(f"[DEBUG] Rewritten : {standalone_query}")

        # ------------------------------------------------------------------
        # 2. Vector retrieval (k=20) using the standalone query
        # ------------------------------------------------------------------
        vector_results = get_docs_by_distance(standalone_query)
        candidates = [doc for doc, _ in vector_results]

        print(f"\nVECTOR SEARCH  ({len(candidates)} candidates, threshold applied)")

        # ------------------------------------------------------------------
        # 3. Rerank candidates using the standalone query; apply threshold
        #    then keep only top RERANKER_TOP_N
        # ------------------------------------------------------------------
        scored = rerank(standalone_query, candidates)

        # Apply threshold first, then cap at top-N
        above_threshold = [(score, doc) for score, doc in scored
                           if score >= RERANKER_THRESHOLD]
        final_docs = [doc for _, doc in above_threshold[:RERANKER_TOP_N]]

        print(f"\nRERANKER  (threshold={RERANKER_THRESHOLD}, top_n={RERANKER_TOP_N})")
        for i, (score, doc) in enumerate(scored, 1):
            if i > max(len(above_threshold), RERANKER_TOP_N) + 2:
                break  # stop printing once clearly below relevance
            selected = (score, doc) in above_threshold[:RERANKER_TOP_N]
            label = "✅" if selected else "❌"
            print(f"  {i}. score={score:.4f} {label} | {_preview(doc.page_content)}")

        print(f"\nSUMMARY")
        print(f"  Vector candidates : {len(candidates)}")
        print(f"  Above threshold   : {len(above_threshold)}")
        print(f"  Sent to LLM       : {len(final_docs)}")

        # ------------------------------------------------------------------
        # 4. Answer generation using the ORIGINAL question + final context
        # ------------------------------------------------------------------
        context = "\n\n---\n\n".join(doc.page_content for doc in final_docs) if final_docs else ""

        prompt = ANSWER_TEMPLATE.invoke({
            "question": question,        # original — not the rewritten query
            "context": context,
            "history": history_str,      # labelled as reference only
        })

        raw = llm.invoke(prompt).content
        if isinstance(raw, list):
            answer = "".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in raw
            )
        else:
            answer = raw or ""
        print(f"\nANSWER\n  {answer}\n")
        return answer

    return full_pipeline


def ask_question(question: str, history: list[dict] = None) -> str:
    rag_chain = build_rag_chain()
    return rag_chain(question, history or [])