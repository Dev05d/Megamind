import os
import re
import json
import uuid
import questionary
import ollama
from qdrant_client import models
from qdrant_client.models import PointVectors
from fastembed import SparseTextEmbedding

from core.vector_store import _get_client
from core.config import COLLECTION_NAME, EMBED_MODEL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _update_config_files(new_model: str, new_dim: int):
    """Safely rewrites EMBED_MODEL and VECTOR_SIZE using absolute paths and robust parsing."""
    
    # 1. Update .env file
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
            
        embed_found, vec_found = False, False
        for i, line in enumerate(lines):
            if line.strip().startswith("EMBED_MODEL="):
                lines[i] = f"EMBED_MODEL={new_model}\n"
                embed_found = True
            elif line.strip().startswith("VECTOR_SIZE="):
                lines[i] = f"VECTOR_SIZE={new_dim}\n"
                vec_found = True
                
        if not embed_found:
            lines.append(f"\nEMBED_MODEL={new_model}\n")
        if not vec_found:
            lines.append(f"VECTOR_SIZE={new_dim}\n")
            
        with open(env_path, "w") as f:
            f.writelines(lines)
            
    # 2. Update core/config.py
    config_path = os.path.join(BASE_DIR, "core", "config.py")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            content = f.read()
            
        # Safely target the specific os.getenv strings without complex regex groups
        content = re.sub(
            r'os\.getenv\(\s*["\']EMBED_MODEL["\']\s*,\s*["\'][^"\']*["\']\s*\)',
            lambda m: f'os.getenv("EMBED_MODEL", "{new_model}")',
            content
        )
        content = re.sub(
            r'os\.getenv\(\s*["\']VECTOR_SIZE["\']\s*,\s*\d+\s*\)',
            f'os.getenv("VECTOR_SIZE", {new_dim})', 
            content
        )
        
        with open(config_path, "w") as f:
            f.write(content)

    print(f"\n[Megamind] ⚙️  Updated configs: EMBED_MODEL='{new_model}' | VECTOR_SIZE={new_dim}")


def _get_native_dim(new_model: str):
    """Best-effort lookup of the model's native/max embedding size via ollama.show()."""
    try:
        info = ollama.show(new_model)
        model_info = getattr(info, "modelinfo", None)
        if model_info is None and hasattr(info, "get"):
            model_info = info.get("model_info", {})
        if not model_info:
            return None

        for key, val in model_info.items():
            if key.endswith(".embedding_length"):
                return int(val)
    except Exception:
        pass
    return None


def _select_dimension(new_model: str, default_dim: int):
    """Validates custom dimensional truncation if the user requests it."""
    native_dim = _get_native_dim(new_model)
    if native_dim:
        print(f"[Megamind] '{new_model}' native dimension: {native_dim}")

    while True:
        prompt = f"Dimension for '{new_model}' [Enter = default {default_dim}]:"
        raw = questionary.text(prompt, default="").ask()

        if raw is None:
            return None, None 

        raw = raw.strip()
        if raw == "":
            return default_dim, {}

        if not raw.isdigit() or int(raw) <= 0:
            print("[Error] Please enter a positive integer, or press Enter for the default.")
            continue

        custom_dim = int(raw)

        if native_dim and custom_dim > native_dim:
            print(f"[Error] {custom_dim} exceeds '{new_model}'s native dimension of {native_dim}. Try again.")
            continue

        options = {"dimensions": custom_dim}
        print(f"[Megamind] Testing if '{new_model}' supports {custom_dim} dimensions...")
        try:
            test_resp = ollama.embed(model=new_model, input="test", options=options)
            actual_dim = len(test_resp["embeddings"][0])
        except Exception as e:
            print(f"[Error] Model rejected that dimension: {e}")
            continue

        if actual_dim != custom_dim:
            print(f"[Error] '{new_model}' does not support truncation to {custom_dim} (returned {actual_dim}).")
            continue

        return custom_dim, options


def change_embedd_model():
    """Master function to swap embedding models and handle Qdrant updates/migrations."""
    client = _get_client()
    print("\n=== 🧠 Change Embedding Model ===")
    
    try:
        ollama_info = ollama.list()
        if hasattr(ollama_info, 'models'):
            installed_models = [m.model for m in ollama_info.models]
        else:
            installed_models = [m.get("name", m.get("model")) for m in ollama_info.get("models", [])]
        installed_models = [m.replace(":latest", "") for m in installed_models if m]
    except Exception as e:
        print(f"[Error] Could not communicate with Ollama. Is it running? ({e})")
        return

    if not installed_models:
        print("[Error] No models found in Ollama.")
        return

    new_model = questionary.select(
        f"Current Model: {EMBED_MODEL}\nSelect a new model to use for Dense Embeddings:",
        choices=installed_models + ["Cancel"]
    ).ask()
    
    if new_model == "Cancel" or not new_model:
        return

    if new_model == EMBED_MODEL:
        print("[Megamind] That is already the active model!")
        return

    print(f"\n[Megamind] Waking up '{new_model}' and calculating default dimensions...")
    try:
        test_resp = ollama.embed(model=new_model, input="test")
        default_dim = len(test_resp["embeddings"][0])
    except Exception as e:
        print(f"[Error] Failed to run '{new_model}'.\nDetails: {e}")
        return

    new_dim, embed_options = _select_dimension(new_model, default_dim)
    if new_dim is None:
        print("Operation cancelled.")
        return

    if not client.collection_exists(COLLECTION_NAME):
        print(f"[Megamind] Database is empty. Skipping chunk migration.")
        _update_config_files(new_model, new_dim)
        return

    col_info = client.get_collection(COLLECTION_NAME)
    current_dim = col_info.config.params.vectors["dense"].size
    
    print(f" -> Current Database Dimensions: {current_dim}")
    print(f" -> '{new_model}' Selected Dimensions: {new_dim}")

    if new_dim == current_dim:
        print("\n[Megamind] 🟢 Dimensions Match! Using fast in-place vector replacement.")
        _run_inplace_update(client, new_model, embed_options)
    else:
        print("\n[Megamind] 🔴 Dimension Mismatch! A full database rebuild is required.")
        _run_full_migration(client, new_model, new_dim, embed_options)
        
    _update_config_files(new_model, new_dim)
    print("\n[Megamind] 🎉 Embedding model switch complete!")



# Helper Functions (Optimized for Scale)

def _run_inplace_update(client, new_model, embed_options):
    """Updates only the dense vectors without deleting the collection."""
    confirm = questionary.confirm(f"Update ALL chunks to use '{new_model}'?").ask()
    if not confirm:
        print("Operation cancelled.")
        return
 
    # Get an upfront total so we can show real progress instead of just a running count.
    try:
        total_points = client.count(collection_name=COLLECTION_NAME, exact=True).count
    except Exception:
        total_points = None
 
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
 
    print(f"\n[Megamind] Starting surgical vector replacement...")
 
    offset = None
    total_updated = 0
    total_skipped = 0
    batch_size = 100
 
    pbar = tqdm(total=total_points, unit="chunk", desc="Re-embedding") if has_tqdm and total_points else None
 
    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME, offset=offset, limit=batch_size,
            with_payload=True, with_vectors=False
        )
 
        if not records:
            break
 
        valid_records = [r for r in records if r.payload and r.payload.get("text")]
        skipped = len(records) - len(valid_records)
        total_skipped += skipped
 
        texts = [r.payload.get("text") for r in valid_records]
 
        if not texts:
            if pbar:
                pbar.update(len(records))
            elif skipped:
                print(f"  ⚠️  Skipped {skipped} record(s) with no 'text' payload (kept old vectors).")
            offset = next_offset
            if next_offset is None:
                break
            continue
 
        embed_response = ollama.embed(model=new_model, input=texts, options=embed_options)
        new_vectors = embed_response["embeddings"]
 
        points_to_update = [
            PointVectors(id=record.id, vector={"dense": new_vectors[i]})
            for i, record in enumerate(valid_records)
        ]
 
        client.update_vectors(collection_name=COLLECTION_NAME, points=points_to_update)
        total_updated += len(points_to_update)
 
        if pbar:
            pbar.update(len(records))  # advance by full batch (valid + skipped) to match total_points
        else:
            pct = f" ({total_updated / total_points:.0%})" if total_points else ""
            print(f"  -> Re-embedded and updated {total_updated}"
                  f"{f'/{total_points}' if total_points else ''} chunks...{pct}")
            if skipped:
                print(f"  ⚠️  Skipped {skipped} record(s) with no 'text' payload (kept old vectors).")
 
        if next_offset is None:
            break
        offset = next_offset
 
    if pbar:
        pbar.close()
 
    if total_skipped:
        print(f"[Megamind] ⚠️  Skipped {total_skipped} record(s) total with no 'text' payload (kept old vectors).")
    print(f"[Megamind] ✅ Done — {total_updated} chunks re-embedded.")

def _run_full_migration(client, new_model, new_dim, embed_options):
    """Extracts text, destroys the collection, rebuilds with new dimensions, and repopulates."""
    confirm = questionary.confirm(
        f"WARNING: This will temporarily wipe your database and re-embed EVERYTHING into a {new_dim}-dimension format. Proceed?"
    ).ask()
    if not confirm:
        print("Operation cancelled.")
        return
 
    print("\n[Megamind] 📦 Extracting all existing data into memory...")
    all_payloads = []
    offset = None
 
    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME, offset=offset, limit=500,
            with_payload=True, with_vectors=False
        )
        for r in records:
            if r.payload and "text" in r.payload:
                all_payloads.append(r.payload)
 
        if next_offset is None:
            break
        offset = next_offset
 
    print(f" -> Safely buffered {len(all_payloads)} chunks.")
 
    # Backup to disk before destroying anything, in case the rebuild fails partway through.
    backup_path = "megamind_migration_backup.json"
    try:
        import json
        with open(backup_path, "w") as f:
            json.dump(all_payloads, f)
        print(f" -> Backup written to '{backup_path}' (safe to delete after migration succeeds).")
    except Exception as e:
        print(f"[Warning] Could not write backup file: {e}")
 
    print(f"\n[Megamind] 💥 Destroying old collection and rebuilding...")
    client.delete_collection(COLLECTION_NAME)
 
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": models.VectorParams(size=new_dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()}
    )
 
    print(f"\n[Megamind] 🧠 Re-embedding {len(all_payloads)} chunks...")
    _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
 
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
 
    total_chunks = len(all_payloads)
    batch_size = 50
    pbar = tqdm(total=total_chunks, unit="chunk", desc="Migrating") if has_tqdm and total_chunks else None
    migrated = 0
 
    for i in range(0, total_chunks, batch_size):
        batch = all_payloads[i:i + batch_size]
        texts = [p["text"] for p in batch]
 
        dense_response = ollama.embed(model=new_model, input=texts, options=embed_options)
        dense_vectors = dense_response["embeddings"]
        sparse_generators = list(_sparse_model.embed(texts))
 
        points = []
        for j, payload in enumerate(batch):
            source = payload.get("source", "unknown")
            chunk_index = payload.get("chunk_index", j)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}_{chunk_index}"))
 
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vectors[j],
                        "sparse": models.SparseVector(
                            indices=sparse_generators[j].indices.tolist(),
                            values=sparse_generators[j].values.tolist()
                        )
                    },
                    payload=payload
                )
            )
 
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        migrated += len(points)
 
        if pbar:
            pbar.update(len(points))
        else:
            pct = f" ({migrated / total_chunks:.0%})" if total_chunks else ""
            print(f" -> Migrated {migrated}/{total_chunks} chunks...{pct}")
 
    if pbar:
        pbar.close()
 
    print(f"[Megamind] ✅ Done — {migrated} chunks migrated.")
 
    # Migration succeeded — clean up the backup.
    if os.path.exists(backup_path):
        try:
            os.remove(backup_path)
        except Exception:
            pass


def _process_migration_batch(client, new_model, batch, sparse_model, embed_options):
    """Helper to process and upsert a single batch during streaming migration."""
    texts = [p["text"] for p in batch]
    
    dense_response = ollama.embed(model=new_model, input=texts, options=embed_options)
    dense_vectors = dense_response["embeddings"]
    sparse_generators = list(sparse_model.embed(texts))
    
    points = []
    for j, payload in enumerate(batch):
        source = payload.get("source", "unknown")
        chunk_index = payload.get("chunk_index", j)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}_{chunk_index}"))
        
        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vectors[j],
                    "sparse": models.SparseVector(
                        indices=sparse_generators[j].indices.tolist(),
                        values=sparse_generators[j].values.tolist()
                    )
                },
                payload=payload
            )
        )
        
    client.upsert(collection_name=COLLECTION_NAME, points=points)