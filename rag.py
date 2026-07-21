import json
import time
from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm
from vector_store import get_docs_by_distance
from reranker import rerank
from config import RERANKER_THRESHOLD, RERANKER_TOP_N, RERANKER_ENABLED

# ---------------------------------------------------------------------------
# Prompt: combined router + query contextualization
# ---------------------------------------------------------------------------
CONTEXTUALIZE_PROMPT = """You are a routing and query-rewriting assistant for a retrieval system.

Given the conversation history and the user's current question, you must:
1. Decide the route.
2. Rewrite the question into a self-contained standalone query.

ROUTING RULES — choose exactly one:
- "social"  : ONLY for clearly lightweight conversational messages with zero
               information need — greetings, thanks, farewells, simple filler
               (e.g. "hi", "thanks!", "bye", "you're welcome").
- "rag"     : Everything else — any question, any request for information,
               anything that might need a knowledge-base lookup, or any mixed
               message that combines chat with a question
               (e.g. "hello, what is the vacation policy?" → rag).
               When in doubt, use "rag".

REWRITING RULES:
- Produce a single self-contained search query using context from history.
- If the question is already self-contained, return it unchanged.
- Resolve pronouns or references (e.g. "those teams", "he", "that policy").
- Preserve the user's original intent exactly.
- Do NOT introduce new entities, competitions, organizations, products, people, dates, or assumptions that are not explicitly mentioned in the user's question or conversation history.
- Do NOT expand abbreviations or infer missing context unless it is explicitly available in the conversation history.
- If multiple interpretations are possible, prefer the most literal interpretation instead of guessing.
- For "social" messages, set standalone_query to the original message as-is.
- Do NOT answer the question.

OUTPUT FORMAT — respond with valid JSON only, no markdown, no extra text:
{{"route": "rag" | "social", "standalone_query": "<rewritten query>"}}

Conversation history:
{history}

Current question: {question}"""

CONTEXTUALIZE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("human", CONTEXTUALIZE_PROMPT),
])

# ---------------------------------------------------------------------------
# Prompt: answer generation
# ---------------------------------------------------------------------------
ANSWER_PROMPT = """You are a specialized assistant.

Grounding rules:
- Answer ONLY using the Context from documents provided below.
- Never use external knowledge, and never answer from memory — if a fact the question asks about is not explicitly present in the Context, say plainly that it isn't present rather than guessing.
- Do not refuse if the required information is explicitly present in the context, even if the answer is partial.
- Reply with "That's not my specialization." ONLY when the context contains no relevant information for answering the user's question.

Completeness rules:
- Extract every relevant fact from the Context — do not compress a multi-part fact down to just its headline clause.
- When the Context lists several fields, properties, options, or steps, include ALL of them, not only the most prominent ones.
- Preserve validation rules and constraints in full: nullability, required-vs-optional, ranges/minimums/maximums, and defaults are each a separate requirement — keep every one, not just the main clause.
- Preserve lists completely rather than summarizing them down to a few examples.

Style:
- Use the resolved question to understand exactly what the user is asking; use the original question and conversation history only to preserve conversational context.
- Ignore context that is unrelated to the question, but never trim relevant details out of the parts that ARE related.
- Write natural, direct English. Use bullets/lists only when the question itself asks for a list or the facts are inherently list-shaped (e.g. several named fields); otherwise answer in plain sentences that still state every fact.

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
# Prompt: social / small-talk response
# ---------------------------------------------------------------------------
SOCIAL_PROMPT = """You are a friendly assistant. The user sent a casual conversational
message — a greeting, farewell, thanks, or simple filler. Reply naturally and briefly.
Do not mention documents, knowledge bases, or your specialization unless asked.

Conversation history (for reference):
{history}

User message: {question}"""

SOCIAL_TEMPLATE = ChatPromptTemplate.from_messages([
    ("human", SOCIAL_PROMPT),
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
# Query contextualization + routing (single LLM call)
# ---------------------------------------------------------------------------

def _extract_text(raw) -> str:
    """Safely extract a plain string from an LLM response .content value."""
    if isinstance(raw, list):
        return "".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in raw
        ).strip()
    return (raw or "").strip()


def contextualize_query(question: str, history: list[dict], llm) -> tuple[str, str]:
    """
    Always calls the LLM to both route the message and rewrite the query.

    Returns:
        (route, standalone_query) where route is "rag" or "social".
        Falls back to ("rag", question) on any parse error.
    """
    prompt = CONTEXTUALIZE_TEMPLATE.invoke({
        "history": _format_history(history),
        "question": question,
    })
    raw = _extract_text(llm.invoke(prompt).content)

    try:
        # Strip accidental markdown fences before parsing
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        route = parsed.get("route", "rag").lower()
        standalone_query = parsed.get("standalone_query", "").strip() or question
        if route not in ("rag", "social"):
            route = "rag"
    except (json.JSONDecodeError, AttributeError):
        print(f"[WARN] Router could not parse LLM output — defaulting to rag. Raw: {raw!r}")
        route, standalone_query = "rag", question

    return route, standalone_query


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
        # 1. Contextualize + Route (single LLM call)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        route, standalone_query = contextualize_query(question, history, llm)
        t_contextualize = time.perf_counter() - t0

        print(f"\n[DEBUG] Question       : {question}")
        print(f"[DEBUG] Route          : {route.upper()}")
        print(f"[DEBUG] Standalone Q   : {standalone_query}")

        # ------------------------------------------------------------------
        # 2a. Social branch — skip retrieval entirely
        # ------------------------------------------------------------------
        if route == "social":
            print("[DEBUG] RAG pipeline   : SKIPPED (social route)")
            t0 = time.perf_counter()
            social_prompt = SOCIAL_TEMPLATE.invoke({
                "history": history_str,
                "question": question,
            })
            raw_answer = llm.invoke(social_prompt).content
            answer = _extract_text(raw_answer) if not isinstance(raw_answer, str) else raw_answer
            t_social = time.perf_counter() - t0
            t_total = time.perf_counter() - t_total

            print("\n" + "─" * 40)
            print("  PERFORMANCE SUMMARY  (social)")
            print("─" * 40)
            print(f"  Contextualization : {t_contextualize:.2f}s")
            print(f"  Social reply      : {t_social:.2f}s")
            print("─" * 40)
            print(f"  Total Pipeline    : {t_total:.2f}s")
            print("─" * 40 + "\n")
            print(f"\nANSWER\n  {answer}\n")
            return answer

        # ------------------------------------------------------------------
        # 2b. RAG branch
        # ------------------------------------------------------------------
        print("[DEBUG] RAG pipeline   : RUNNING")

        # Vector retrieval
        t0 = time.perf_counter()
        vector_results = get_docs_by_distance(standalone_query)
        candidates = [doc for doc, _ in vector_results]
        t_vector = time.perf_counter() - t0

        print(f"\nVECTOR SEARCH  ({len(candidates)} candidates, threshold applied)")

        # Rerank
        t0 = time.perf_counter()
        if RERANKER_ENABLED:
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
        else:
            # Reranker disabled — send the top RERANKER_TOP_N vector-retrieved
            # candidates straight to the LLM (rerank() above is left untouched
            # so this can be flipped back on via RERANKER_ENABLED in config.py).
            final_docs = candidates[:RERANKER_TOP_N]
            t_rerank = time.perf_counter() - t0

            print(f"\nRERANKER : DISABLED")
            for i, doc in enumerate(final_docs, 1):
                print(f"  {i}. | {_preview(doc.page_content)}")

            print(f"\nSUMMARY")
            print(f"  Vector candidates : {len(candidates)}")
            print(f"  Documents sent to LLM: {len(final_docs)}")

        # Prompt building
        t0 = time.perf_counter()
        context = "\n\n---\n\n".join(doc.page_content for doc in final_docs) if final_docs else ""
        # --- TEMP DEBUG: "CONTEXT SENT TO LLM" full dump suppressed while
        # debugging retrieval quality via the compact report in
        # vector_store.get_docs_by_distance(). Re-enable by uncommenting below.
        #
        # print("\n" + "=" * 100)
        # print("CONTEXT SENT TO LLM")
        # print("=" * 100)
        #
        # for i, doc in enumerate(final_docs, 1):
        #     print(f"\n---------- CHUNK {i} ----------")
        #     print(doc.page_content)
        #     print("\nMetadata:")
        #     print(doc.metadata)
        #
        # print("=" * 100 + "\n")

        prompt = ANSWER_TEMPLATE.invoke({
            "original_question": question,
            "resolved_question": standalone_query,
            "context": context,
            "history": history_str,
        })
        t_prompt = time.perf_counter() - t0

        # Final LLM generation
        t0 = time.perf_counter()
        answer = llm.invoke(prompt).content
        t_llm = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total

        print("\n" + "─" * 40)
        print("  PERFORMANCE SUMMARY  (rag)")
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