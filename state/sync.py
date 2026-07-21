import os

from .state_manager import load_state, save_state, LOCK_FILE
from .snapshot import generate_snapshot
from .comparator import compare_states
from vector_store import load_document_from_minio , add_documents,delete_documents_by_object_name


def sync():
    if os.path.exists(LOCK_FILE):
        print("⏸️  Ingest lock present — a full rebuild is in progress, skipping this sync cycle.")
        return

    print("=" * 50)
    print("  Ingestion Sync")
    print("=" * 50)

    # 1. Load previous state
    old_state = load_state()
    print(f"\n📂 Loaded previous state ({len(old_state)} object(s)).")
 
    # 2. Generate current snapshot from MinIO
    current_state = generate_snapshot()
    print(f"📡 Current snapshot fetched ({len(current_state)} object(s)).")
 
    # 3. Compare both states
    changes = compare_states(old_state, current_state)
 
    new_files     = changes["new_files"]
    updated_files = changes["updated_files"]
    deleted_files = changes["deleted_files"]
 
    print(f"\n--- Detected Changes ---")
    print(f"  🆕 New: {len(new_files)}  ✏️  Updated: {len(updated_files)}  🗑️  Deleted: {len(deleted_files)}")
 
    # 4. Process new files
    if new_files:
        print(f"\n🆕 Adding new files...")
        for name in new_files:
            print(f"  + {name}")
            document = load_document_from_minio(name)
            add_documents([document])
 
    # 5. Process updated files
    if updated_files:
        print(f"\n✏️  Updating changed files...")
        for name in updated_files:
            print(f"  ~ {name}")
            delete_documents_by_object_name(name)
            document = load_document_from_minio(name)
            add_documents([document])
 
    # 6. Process deleted files
    if deleted_files:
        print(f"\n🗑️  Removing deleted files...")
        for name in deleted_files:
            print(f"  - {name}")
            delete_documents_by_object_name(name)
 
    # 7. Save the new snapshot only after all operations succeed
    save_state(current_state)
    print("\n✅ Sync complete. State saved.")
    print("=" * 50)
 
 
if __name__ == "__main__":
    sync()