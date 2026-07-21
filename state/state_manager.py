import json
import os

STATE_FILE = "ingestion_state.json"
LOCK_FILE = ".ingest.lock"


def load_state() -> dict:
    """Load the ingestion state from disk. Returns an empty dict if the file does not exist."""
    if not os.path.exists(STATE_FILE):
        return {}

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f) #convert json to dict


def save_state(state: dict) -> None:
    """Persist the ingestion state to disk as indented JSON."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False) #convert dict to json