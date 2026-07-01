from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from llm import get_llm
from vector_store import get_retriever
from conversation_memory import get_relevant_history  
from vector_store import debug_rag_retrieval

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
    retriever = get_retriever()

    def full_pipeline(question):
        docs = retriever.invoke(question)
        
        context = ""
        if isinstance(docs, list):
            texts = []
            for item in docs:
                if hasattr(item, 'page_content'):
                    texts.append(item.page_content)
                elif isinstance(item, dict):
                    texts.append(str(item.get('page_content') or item))
                else:
                    texts.append(str(item))
            context = "\n\n---\n\n".join(texts)
        
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
    debug_rag_retrieval(question, threshold=0.65)
    
    rag_chain = build_rag_chain()
    
    # Print prompt
    docs = get_retriever().invoke(question)
    context = "\n\n---\n\n".join(
        d.page_content if hasattr(d, 'page_content') else str(d) 
        for d in docs
    ) if isinstance(docs, list) else str(docs)
    
    print("\n" + "="*80)
    print("FINAL PROMPT WITH CONTEXT:")
    print("="*80)
    print(prompt_template.invoke({
        "question": question,
        "context": context,
        "history": get_relevant_history(question)
    }))
    print("="*80 + "\n")
    
    answer = rag_chain(question)
    print(f"[DEBUG] Final Answer Length: {len(answer)}")
    
    return answer