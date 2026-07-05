from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from llm import get_llm
from conversation_memory import get_relevant_history  
from vector_store import get_docs_by_distance, debug_rag_retrieval

SYSTEM_PROMPT = """
You are a specialized assistant.

Rules:
- If the Context from documents contains relevant information, answer using it.
- If the context is irrelevant or empty, reply EXACTLY with: "That's not my specialization."
- Do not answer using general knowledge.

Previous conversations:
{history}

Context from documents:
{context}

Now answer the user question following the rules above.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    llm = get_llm()

    def full_pipeline(question):
        docs = get_docs_by_distance(question)  # ← direct distance filter

        context = "\n\n---\n\n".join(
            doc.page_content for doc in docs
        ) if docs else ""

        history = get_relevant_history(question)

        prompt = prompt_template.invoke({
            "question": question,
            "context": context,
            "history": history
        })

        return llm.invoke(prompt).content

    return full_pipeline


def ask_question(question: str):
    print(f"\n[DEBUG] Question: {question}")
    debug_rag_retrieval(question)

    rag_chain = build_rag_chain()

    docs = get_docs_by_distance(question, k=4, threshold=0.65)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs) if docs else ""

    print("\n" + "="*80)
    print("FINAL PROMPT WITH CONTEXT:")
    print("="*80)
    print(prompt_template.invoke({
        "question": question,
        "context": context,
        "history": get_relevant_history(question)
    }))
    print("="*80 + "\n")

    return rag_chain(question)