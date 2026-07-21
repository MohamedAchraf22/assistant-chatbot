import os

from vector_store import (
    load_documents,
    split_text,
    save_to_chroma
)
from state.snapshot import generate_snapshot
from state.state_manager import save_state, LOCK_FILE


def generate_data_store():
    documents = load_documents()

    chunks = split_text(documents)

    save_to_chroma(chunks)

    # Chroma was just rebuilt from the current MinIO contents — record that
    # same snapshot as the ingestion state, so the next sync() run compares
    # against what's actually in Chroma instead of reclassifying everything
    # as new.
    current_state = generate_snapshot()
    save_state(current_state)


if __name__ == "__main__":
    # Filesystem lock: blocks the concurrently-running scheduler's sync()
    # (running in another process) from calling add_documents() on top of
    # an in-progress rebuild. Removed in `finally` so a crash never leaves
    # sync() permanently locked out.
    open(LOCK_FILE, "w").close()
    try:
        generate_data_store()
    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)