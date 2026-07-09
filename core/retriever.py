import requests
import ollama
from qdrant_client import models
from fastembed import SparseTextEmbedding
from core.router import route_query

# Retaining your original configuration architecture
from core.config import DB_PATH, COLLECTION_NAME, EMBED_MODEL, CHAT_MODEL, OLLAMA_HOST, CONTEXT_LIMIT
from core.vector_store import _get_client

_SYSTEM_PROMPT = """You are Megamind, a precise personal knowledge assistant.
Answer ONLY from the provided context. If the context doesn't contain enough
information to answer confidently, say so clearly rather than guessing. 
At the end of your response, provide a brief bulleted list of the exact Sources you referenced."""

_CONTEXT_LIMIT = int(CONTEXT_LIMIT)
_SAFETY_BUFFER = 200
_TARGET_LIMIT = _CONTEXT_LIMIT - _SAFETY_BUFFER
URL = f"{OLLAMA_HOST}/api/tokenize"

# Instantiate the local BM25 keyword mapper globally
_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def get_exact_tokens(text: str, model_name: str) -> int:
    """Bypasses the Python SDK to hit the local Ollama tokenization API."""
    try:
        response = requests.post(
            URL,
            json={"model": model_name, "prompt": text},
            timeout=3
        )
        if response.status_code == 200:
            return len(response.json().get("tokens", []))
    except Exception:
        pass
    return len(text) // 4


def enforce_token_budget(system_base_tokens: int, query_tokens: int, active_db_chunks: list, chat_history: list) -> int:
    """Cascading memory manager."""
    def get_current_total():
        chunk_tokens = sum(c["tokens"] for c in active_db_chunks)
        history_tokens = sum(h["tokens"] for h in chat_history)
        return system_base_tokens + history_tokens + chunk_tokens + query_tokens + 1024

    current_total = get_current_total()

    while current_total > _TARGET_LIMIT and len(active_db_chunks) > 2:
        evicted_chunk = active_db_chunks.pop(0)
        current_total -= evicted_chunk["tokens"]
        print(f"[Memory Manager] ⚠️ Context tight. Evicted old DB Chunk ID: {evicted_chunk.get('id')} (-{evicted_chunk['tokens']} tokens).")

    while current_total > _TARGET_LIMIT and len(chat_history) >= 2:
        evicted_user = chat_history.pop(0)
        evicted_ai = chat_history.pop(0)
        freed = evicted_user["tokens"] + evicted_ai["tokens"]
        current_total -= freed
        print(f"[Memory Manager] ⚠️ Context tight. Evicted old Chat Turn (-{freed} tokens).")
        
    return current_total


def _draw_token_bar(used: int, total: int = _CONTEXT_LIMIT):
    """Draws a color-coded CLI progress bar for exact token usage."""
    if used == 0: 
        return
        
    percentage = min(used / total, 1.0)
    bar_length = 30
    filled = int(bar_length * percentage)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    color = "\033[92m" # Green
    if percentage > 0.60: color = "\033[93m" # Yellow
    if percentage > 0.85: color = "\033[91m" # Red
    reset = "\033[0m"
    
    print(f"\n{color}[Context Memory: {bar} {int(percentage*100)}% ({used}/{total} Tokens)]{reset}\n")


def ask_megamind(initial_question: str, top_k: int = 5) -> None:
    chat_history = []
    active_db_chunks = []
    system_base_tokens = get_exact_tokens(_SYSTEM_PROMPT, CHAT_MODEL)
    current_question = initial_question
    
    print("\n[Megamind] 🟢 Entering continuous chat. Type 'exit' or 'q' to return to menu.")
    
    while True:
        if current_question.lower() in ['exit', 'quit', 'q']:
            print("\n[Megamind] Ending chat session...")
            break
            
        query_tokens = get_exact_tokens(current_question, CHAT_MODEL)
        
        # 🔗 PASSED HERE: Now feeding both chat_history and active_db_chunks into the router
        router_decision = route_query(current_question, chat_history, active_db_chunks)
        intent = router_decision.route
        search_query = router_decision.rewritten_query or current_question
        
        if intent in ["needs_retrieval", "needs_novel_retrieval"]:
            print(f"\n[Megamind] 🔍 Routing: [{intent.upper()}]")
            print(f"           -> Search Target: '{search_query}'")
            
            try:
                query_dense = ollama.embed(model=EMBED_MODEL, input=search_query)["embeddings"][0]
                sparse_gen = next(_sparse_model.embed([search_query]))
                query_sparse = models.SparseVector(
                    indices=sparse_gen.indices.tolist(),
                    values=sparse_gen.values.tolist()
                )

                search_prefetches = [models.Prefetch(query=query_dense, using="dense", limit=top_k)]
                if len(sparse_gen.indices) > 0:
                    search_prefetches.append(models.Prefetch(query=query_sparse, using="sparse", limit=top_k))

                query_filter = None
                if intent == "needs_novel_retrieval" and active_db_chunks:
                    existing_ids = [c["id"] for c in active_db_chunks]
                    query_filter = models.Filter(
                        must_not=[models.HasIdCondition(has_id=existing_ids)]
                    )
                    print(f"           -> 🚫 Excluding {len(existing_ids)} active chunk IDs from Qdrant search.")

                response = _get_client().query_points(
                    collection_name=COLLECTION_NAME,
                    prefetch=search_prefetches,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    query_filter=query_filter, 
                    limit=top_k,
                )

                if not response.points:
                    print("\n[Megamind] I don't have any relevant information on that topic yet.\n")
                else:
                    existing_ids = {c["id"] for c in active_db_chunks}
                    for r in response.points:
                        chunk_id = r.id
                        if chunk_id in existing_ids:
                            continue
                            
                        text = r.payload.get('text', '')
                        source = r.payload.get('source', 'unknown')
                        chunk_text = f"[Source: {source}]\n{text}"
                        
                        c_tokens = get_exact_tokens(chunk_text, CHAT_MODEL)
                        active_db_chunks.append({
                            "id": chunk_id,
                            "text": chunk_text, 
                            "source": source, 
                            "tokens": c_tokens
                        })

            except Exception as e:
                print(f"\n[Megamind] ❌ Database Error: {e}")
                current_question = input("\nYou: ").strip()
                continue
        else:
            print(f"\n[Megamind] 🧠 Routing: [{intent.upper()}] -> Bypassing Database.")

        enforce_token_budget(system_base_tokens, query_tokens, active_db_chunks, chat_history)

        active_chunk_ids = [c["id"] for c in active_db_chunks]
        if active_chunk_ids:
            print(f"[Megamind] 🗃️ Active Chunk IDs in Memory: {active_chunk_ids}")

        compiled_context = "\n\n---\n\n".join(c["text"] for c in active_db_chunks)
        sys_content = _SYSTEM_PROMPT
        if compiled_context:
            sys_content += f"\n\nContext Database:\n{compiled_context}"
            
        llm_payload = [{"role": "system", "content": sys_content}]
        
        for msg in chat_history:
            llm_payload.append({"role": msg["role"], "content": msg["content"]})
            
        llm_payload.append({"role": "user", "content": current_question})

        print("\n[Megamind] ", end="", flush=True)
        
        full_response = ""
        ai_eval_count = 0
        
        try:
            stream = ollama.chat(
                model=CHAT_MODEL,
                messages=llm_payload,
                stream=True,
                options={"num_ctx": _CONTEXT_LIMIT}
            )
            
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_response += content
                
                if chunk.get("done") is True:
                    ai_eval_count = chunk.get("eval_count", 0)

        except Exception as e:
            print(f"\n[Megamind] ❌ LLM Generation Error: {e}")
            current_question = input("\nYou: ").strip()
            continue

        print()
        
        chat_history.append({"role": "user", "content": current_question, "tokens": query_tokens})
        
        if ai_eval_count == 0:
            ai_eval_count = get_exact_tokens(full_response, CHAT_MODEL)
            
        chat_history.append({"role": "assistant", "content": full_response, "tokens": ai_eval_count})
        
        total_active_now = (
            system_base_tokens + 
            sum(c["tokens"] for c in active_db_chunks) + 
            sum(h["tokens"] for h in chat_history)
        )
        _draw_token_bar(total_active_now)
        
        try:
            current_question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Megamind] Ending chat session...")
            break
            
        if not current_question:
            print("\n[Megamind] Ending chat session...")
            break