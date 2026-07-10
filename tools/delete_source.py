from core.vector_store import _get_client
from core.config import COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue


def delete_source():
    """Allows the user to completely remove a source and all its chunks from the database."""
    client = _get_client()
    try:
        if not client.collection_exists(COLLECTION_NAME):
            print("\n[Megamind] Database is currently empty.\n")
            return

        print("\n[Megamind] Gathering list of sources...")
        offset = None
        unique_sources = set()
        
        while True:
            records, next_offset = client.scroll(
                collection_name=COLLECTION_NAME, offset=offset, limit=1000, 
                with_payload=True, with_vectors=False
            )
            for r in records:
                source = r.payload.get("source") if r.payload else None
                if source: unique_sources.add(source)
            if next_offset is None: break
            offset = next_offset

        if not unique_sources:
            print("\n[Megamind] No sources found in the database.\n")
            return

        sources_list = sorted(list(unique_sources))
        print("\n=== 🗑️ Select a Source to DELETE ===")
        for idx, src in enumerate(sources_list, start=1):
            print(f"[{idx}] {src}")
        print("====================================")
        
        choice = input(f"Enter a number (1-{len(sources_list)}) or 'q' to cancel: ").strip()
        if choice.lower() == 'q' or not choice: return
            
        try:
            selected_idx = int(choice) - 1
            if not (0 <= selected_idx < len(sources_list)):
                print("[Error] Invalid selection.")
                return
            selected_source = sources_list[selected_idx]
        except ValueError:
            print("[Error] Please enter a valid number.")
            return

        # Double check before deleting!
        confirm = input(f"\n⚠️ Are you sure you want to delete ALL data for:\n{selected_source}\n(y/N): ")
        if confirm.lower() != 'y':
            print("Deletion cancelled.")
            return

        # Execute the deletion filter in Qdrant
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=selected_source))]
            )
        )
        print(f"\n[Megamind] 💥 Successfully purged '{selected_source}' from the knowledge base.\n")

    except Exception as e:
        print(f"\n[Error] Could not delete source: {e}\n")