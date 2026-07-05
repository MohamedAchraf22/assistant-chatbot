import os
import shutil

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import DATA_PATH, CHROMA_PATH , RETRIEVAL_THRESHOLD


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    print(
        f"Split {len(documents)} documents into {len(chunks)} chunks."
    )

    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )

    db.persist()

    print(
        f"Saved {len(chunks)} chunks to {CHROMA_PATH}"
    )


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return db



def get_docs_by_distance(question: str, k: int = 4, threshold: float = RETRIEVAL_THRESHOLD):
    """
    Returns documents whose L2 distance to the query is <= threshold.
    Lower distance = more similar. Typical good range: 0.0 – 0.8.
    """
    db = load_vector_store()
    docs_with_scores = db.similarity_search_with_score(question, k=k)
    return [doc for doc, distance in docs_with_scores if distance <= threshold]


def debug_rag_retrieval(question: str, threshold: float = RETRIEVAL_THRESHOLD):
    db = load_vector_store()
    docs_with_scores = db.similarity_search_with_score(question, k=6)

    print(f"\n{'='*70}")
    print(f"🔍 RAG DEBUG - Query: {question}")
    print(f"Distance threshold (lower = better): <= {threshold}")
    print(f"{'='*70}\n")

    relevant = 0
    for i, (doc, distance) in enumerate(docs_with_scores, 1):
        is_relevant = distance <= threshold
        print(f"{i}. Distance: {distance:.4f} → {'✅' if is_relevant else '❌'}")
        print(f"   Content: {doc.page_content[:180]}...\n")
        if is_relevant:
            relevant += 1

    print(f"✅ Total relevant documents returned: {relevant}")
    print(f"{'='*70}\n")
    return relevant