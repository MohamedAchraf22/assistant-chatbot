import os
import uuid
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import CHROMA_PATH, EMBEDDING_MODEL
import shutil

# Path for conversation memory
CONVERSATION_PATH = os.path.join(CHROMA_PATH, "conversations")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

_conversation_store: Chroma | None = None


def get_conversation_vectorstore() -> Chroma:
    """Return a shared Chroma instance for conversation memory, opened once."""
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = Chroma(
            persist_directory=CONVERSATION_PATH,
            embedding_function=embeddings,
        )
    return _conversation_store


def save_conversation(user_message: str, bot_reply: str, session_id: str = None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    os.makedirs(CONVERSATION_PATH, exist_ok=True)

    timestamp = datetime.now().isoformat()

    content = f"User: {user_message}\nAssistant: {bot_reply}"

    metadata = {
        "session_id": session_id,
        "timestamp": timestamp,
        "type": "conversation"
    }

    doc = Document(page_content=content, metadata=metadata)

    vectorstore = get_conversation_vectorstore()
    vectorstore.add_documents([doc])
    vectorstore.persist()  # save changes to disk

    print(f"Conversation saved | Session: {session_id[:8]}...")
    return session_id


def get_relevant_history(question: str, k: int = 6, threshold: float = 0.65):
    try:
        vectorstore = get_conversation_vectorstore()
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=k)
    except Exception as e:
        print(f"  → Conversation store not ready yet: {e}")
        return "No previous conversations."

    relevant_docs = []
    for doc, distance in docs_with_scores:
        print(f"  → Distance: {distance:.4f} {'✅' if distance <= threshold else '❌'}")
        if distance <= threshold:
            relevant_docs.append(doc)

    if not relevant_docs:
        return "No previous conversations."

    return "\n\n".join(doc.page_content for doc in relevant_docs)


def clear_conversation_history():
    global _conversation_store
    try:
        if os.path.exists(CONVERSATION_PATH):
            _conversation_store = None  # drop the cached instance before deleting the dir
            shutil.rmtree(CONVERSATION_PATH)
            print("Conversation history has been completely cleared.")
            return True
        else:
            print("No conversation history found to clear.")
            return False
    except Exception as e:
        print(f"Error while clearing history: {e}")
        return False


def clear_conversation_by_session(session_id: str):
    pass