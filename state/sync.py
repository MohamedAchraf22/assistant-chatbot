from .state_manager import load_state, save_state
from .snapshot import generate_snapshot
from .comparator import compare_states


def sync():
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

    # 4. Print detected changes
    print("\n--- Detected Changes ---")

    new_files = changes["new_files"]
    updated_files = changes["updated_files"]
    deleted_files = changes["deleted_files"]

    if new_files:
        print(f"\n🆕 New files ({len(new_files)}):")
        for name in new_files:
            print(f"   + {name}")
    else:
        print("\n🆕 New files: none")

    if updated_files:
        print(f"\n✏️  Updated files ({len(updated_files)}):")
        for name in updated_files:
            print(f"   ~ {name}")
    else:
        print("\n✏️  Updated files: none")

    if deleted_files:
        print(f"\n🗑️  Deleted files ({len(deleted_files)}):")
        for name in deleted_files:
            print(f"   - {name}")
    else:
        print("\n🗑️  Deleted files: none")

    # 5. Save current snapshot as the new state
    save_state(current_state)
    print("\n✅ State saved successfully.")
    print("=" * 50)


if __name__ == "__main__":
    sync()