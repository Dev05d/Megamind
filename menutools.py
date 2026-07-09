import questionary

from core.vector_store import init_db, close_db, _get_client
from core.config import COLLECTION_NAME
from core.retriever import ask_megamind
from scrapers.scrape_pdf     import ingest_pdf
from scrapers.scrape_web     import ingest_web
from scrapers.scrape_youtube import ingest_youtube
from scrapers.scrape_reddit  import ingest_reddit
from qdrant_client.models import Filter, FieldCondition, MatchValue
from core.tracker import log_ingestion, backup_pdf

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
            vector = r.vector  # This is the raw list of floats
            
            print(f"--- 🧩 CHUNK #{i} (Qdrant ID: {r.id}) ---")
            print("[📄 Raw Text Data]:")
            print(text)
            print()
            
            # Vector lists are massive (768 or 1024 dimensions), 
            # so we truncate them cleanly so they don't flood your terminal screen.
            # Vector lists are massive, so we truncate them cleanly.
            # Since the Hybrid Search upgrade, r.vector is now a dictionary!
            if sparse_vec and isinstance(sparse_vec, dict) and "indices" in sparse_vec:
                print(f"[🔠 Sparse Vector (FastEmbed)] ({len(sparse_vec['indices'])} exact keywords mapped)")
                vector_preview = ", ".join(f"{val:.4f}" for val in dense_vec[:5])
                print(f"[🧠 Dense Vector (Ollama)] ({len(dense_vec)} dimensions):")
                print(f"  [{vector_preview}, ...]")
                
                sparse_vec = vector.get("sparse")
                if sparse_vec and hasattr(sparse_vec, "indices"):
                    print(f"[🔠 Sparse Vector (FastEmbed)] ({len(sparse_vec.indices)} exact keywords mapped)")
            else:
                print("[🔢 Vector Embedding]: None/Error loading vector")
                
            print("-" * 50 + "\n")
            
        print("==================================================\n")

    except Exception as e:
        print(f"\n[Error] Could not retrieve raw data: {e}\n")

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

def show_db_info():
    """Fetches and displays detailed analytics about the Qdrant collection."""
    client = _get_client()
    try:
        if not client.collection_exists(COLLECTION_NAME):
            print("\n[Megamind] Database is currently empty (collection does not exist).\n")
            return

        # Dictionary to map unique sources to their document type
        source_types = {}
        total_chunks = 0
        
        offset = None
        print("\n[Megamind] Scanning database... (this might take a second)")
        
        while True:
            records, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                offset=offset,
                limit=1000, 
                with_payload=True,
                with_vectors=False
            )
            
            for r in records:
                total_chunks += 1
                payload = r.payload or {}
                
                source = payload.get("source")
                if not source:
                    continue
                
                # Only process the type if we haven't seen this source before
                if source not in source_types:
                    doc_type = payload.get("type")
                    if not doc_type:
                        src = str(source).lower()
                        if "youtube.com" in src or "youtu.be" in src: doc_type = "youtube"
                        elif "reddit.com" in src: doc_type = "reddit"
                        elif src.endswith(".pdf"): doc_type = "pdf"
                        elif src.startswith("http"): doc_type = "web"
                        else: doc_type = "unknown"
                    
                    source_types[source] = doc_type
                    
            if next_offset is None:
                break
            offset = next_offset
            
        # Tally up the types based on unique sources, not chunks
        types_count = {"web": 0, "pdf": 0, "youtube": 0, "reddit": 0, "unknown": 0}
        for doc_type in source_types.values():
            types_count[doc_type] = types_count.get(doc_type, 0) + 1
            
        print("\n=== 🧠 Megamind Database Stats ===")
        print(f"Total Unique Sources: {len(source_types)}")
        print(f"Total Vector Chunks:  {total_chunks}")
        print("\n--- Breakdown by Content Type (Sources) ---")
        print(f"🌐 Websites:      {types_count.get('web', 0)}")
        print(f"📄 PDFs:          {types_count.get('pdf', 0)}")
        print(f"📺 YouTube:       {types_count.get('youtube', 0)}")
        print(f"👾 Reddit:        {types_count.get('reddit', 0)}")
        print(f"❓ Unknown:       {types_count.get('unknown', 0)}")
        print("===========================================\n")
        
    except Exception as e:
        print(f"\n[Error] Could not fetch DB info: {e}\n")

def interactive_menu():
    """Displays the interactive arrow-key menu."""
    while True:
        action = questionary.select(
            "🧠 What would you like Megamind to do?",
            choices=[
                "Ask a Question",
                "Ingest Web Article",
                "Ingest PDF",
                "Ingest YouTube Video",
                "Ingest Reddit Post",
                "Database Info",
                "Show Raw Chunk Data",
                "Delete a Source",
                "Exit"
            ]
        ).ask()

        if action == "Ask a Question":
            query = questionary.text("Enter your question:").ask()
            if query: ask_megamind(query)

        elif action == "Ingest Web Article":
            url = questionary.text("Enter URL:").ask()
            if url: 
                ingest_web(url)
                log_ingestion("web", url)

        elif action == "Ingest PDF":
            path = questionary.path("Enter PDF file path:").ask()
            if path:
                clean_path = path.strip().replace("\\ ", " ").strip("'\"")
                ingest_pdf(clean_path)
                backup_pdf(clean_path)  # <--- physically saves the PDF to the folder!

        elif action == "Ingest YouTube Video":
            url = questionary.text("Enter YouTube URL:").ask()
            if url: 
                ingest_youtube(url)
                log_ingestion("youtube", url)

        elif action == "Ingest Reddit Post":
            url = questionary.text("Enter Reddit Post URL:").ask()
            if url: 
                ingest_reddit(url)
                log_ingestion("reddit", url)

        elif action == "Database Info":
            show_db_info()

        elif action == "Show Raw Chunk Data":  # <-- Connected choice to function
            show_raw_data()

        elif action == "Delete a Source":       # <-- TRIGGER FUNCTION
            delete_source()

        elif action == "Exit" or action is None:
            print("Shutting down Megamind. Goodbye!")
            break