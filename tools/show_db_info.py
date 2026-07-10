from core.vector_store import _get_client
from core.config import COLLECTION_NAME


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