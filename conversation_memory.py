import os
import uuid
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import CHROMA_PATH
import shutil

# Path for conversation memory
CONVERSATION_PATH = os.path.join(CHROMA_PATH, "conversations")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


def get_conversation_vectorstore():
    # Don't create the directory here; let Chroma handle it on first save
    return Chroma(
        persist_directory=CONVERSATION_PATH,
        embedding_function=embeddings
    )


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
    vectorstore.persist() # save changes in the disk

    print(f"Conversation saved | Session: {session_id[:8]}...")
    return session_id


def get_relevant_history(question: str, k: int = 6, threshold: float = 0.65):
    try:
        vectorstore = get_conversation_vectorstore()
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=k)
    except Exception as e:
        # Happens when the conversation store exists on disk but has no index yet
        # (i.e. no conversations have been saved yet)
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
    try:
        if os.path.exists(CONVERSATION_PATH):
            shutil.rmtree(CONVERSATION_PATH)
            print(" Conversation history has been completely cleared.")
            return True
        else:
            print("No conversation history found to clear.")
            return False
    except Exception as e:
        print(f" Error while clearing history: {e}")
        return False


def clear_conversation_by_session(session_id: str):
   
    pass