from vector_store import (
    load_documents,
    split_text,
    save_to_chroma
)


def generate_data_store():
    documents = load_documents()

    chunks = split_text(documents)

    save_to_chroma(chunks)


if __name__ == "__main__":
    generate_data_store()