from core.vector_store import _get_client
from core.config import COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue



def show_raw_data():
    """Allows interactive inspection of raw text chunks and their vector embeddings."""
    client = _get_client()
    try:
        if not client.collection_exists(COLLECTION_NAME):
            print("\n[Megamind] Database is currently empty.\n")
            return

        # Step 1: Scan for all unique sources
        print("\n[Megamind] Gathering list of sources...")
        offset = None
        unique_sources = set()
        
        while True:
            records, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                offset=offset,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            for r in records:
                source = r.payload.get("source") if r.payload else None
                if source:
                    unique_sources.add(source)
            if next_offset is None:
                break
            offset = next_offset

        if not unique_sources:
            print("\n[Megamind] No sources found in the database.\n")
            return

        # Step 2: Display sources as a numbered menu
        sources_list = sorted(list(unique_sources))
        print("\n=== 📂 Select a Source to Inspect ===")
        for idx, src in enumerate(sources_list, start=1):
            print(f"[{idx}] {src}")
        print("=====================================")
        
        choice = input(f"Enter a number (1-{len(sources_list)}) or 'q' to go back: ").strip()
        if choice.lower() == 'q' or not choice:
            return
            
        try:
            selected_idx = int(choice) - 1
            if not (0 <= selected_idx < len(sources_list)):
                print("[Error] Invalid selection.")
                return
            selected_source = sources_list[selected_idx]
        except ValueError:
            print("[Error] Please enter a valid number.")
            return

        # Step 3: Fetch chunks and VECTORS for ONLY the selected source
        print(f"\n[Megamind] Loading chunks for: {selected_source}...\n")
        
        # We use a Qdrant Filter to match the specific source string
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=selected_source))
                ]
            ),
            limit=100,         # Adjust if you have documents with >100 chunks
            with_payload=True,
            with_vectors=True  # CRITICAL: Tell Qdrant to load the vector arrays into RAM
        )

        # Step 4: Print the chunks and embeddings beautifully
        print(f"=== 🔍 Raw Chunks for: {selected_source} ===")
        print(f"Found {len(records)} total vector chunks.\n")
        
        for i, r in enumerate(records, start=1):
            payload = r.payload or {}
            text = payload.get("text", "[No text payload found]")
            # ... (previous text printing code) ...
            
            vector = r.vector  # This is the raw dict containing 'dense' and 'sparse'
            
            print(f"--- 🧩 CHUNK #{i} (Qdrant ID: {r.id}) ---")
            print("[📄 Raw Text Data]:")
            print(text)
            print()
            
            # 1. Safely extract vectors from the Qdrant hybrid dict
            dense_vec = None
            sparse_vec = None
            
            if isinstance(vector, dict):
                dense_vec = vector.get("dense")
                sparse_vec = vector.get("sparse")
            elif hasattr(vector, "dense"): # Handles NamedVector object formats if applicable
                dense_vec = getattr(vector, "dense", None)
                sparse_vec = getattr(vector, "sparse", None)

            # 2. Print Dense Vector if it exists
            if dense_vec is not None:
                vector_preview = ", ".join(f"{val:.4f}" for val in dense_vec[:5])
                print(f"[🧠 Dense Vector (Ollama)] ({len(dense_vec)} dimensions):")
                print(f"  [{vector_preview}, ...]")
            else:
                print("[🧠 Dense Vector]: None found")

            # 3. Print Sparse Vector if it exists
            if sparse_vec is not None:
                # Qdrant sparse vectors can be a dict or an object with an 'indices' attribute
                if isinstance(sparse_vec, dict) and "indices" in sparse_vec:
                    num_keywords = len(sparse_vec["indices"])
                elif hasattr(sparse_vec, "indices"):
                    num_keywords = len(sparse_vec.indices)
                else:
                    num_keywords = "Unknown"
                
                print(f"[🔠 Sparse Vector (FastEmbed)] ({num_keywords} exact keywords mapped)")
            else:
                print("[🔠 Sparse Vector]: None found")
                
            if not dense_vec and not sparse_vec:
                print("[🔢 Vector Embedding]: None/Error loading vector")
                
            print("-" * 50 + "\n")
            
        print("==================================================\n")

    except Exception as e:
        print(f"\n[Error] Could not retrieve raw data: {e}\n")