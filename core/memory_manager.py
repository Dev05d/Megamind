import requests
from core.config import OLLAMA_HOST

_SAFETY_BUFFER = 200
URL = f"{OLLAMA_HOST}/api/tokenize"

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

def enforce_token_budget(
    system_base_tokens: int, 
    query_tokens: int, 
    active_db_chunks: list, 
    chat_history: list,
    context_limit: int,
    label: str = "Memory Manager",
    persistent: bool = True,
    generation_buffer: int = 1024
) -> int:
    """Cascading memory manager governed strictly by the passed context_limit.

    label: identifies the caller in log output (e.g. "Router", "Retriever").
    persistent: whether the lists passed in are the real session state
        (evictions permanently shrink chat memory) or local scratch copies
        (evictions only affect this one call, e.g. the router's own prompt).
        This is purely a logging distinction -- callers are responsible for
        actually passing copies vs. the live lists.
    generation_buffer: headroom reserved for the model's own response.
        Defaults to 1024 for chat-generation callers; pass 0 (or a smaller
        value) for classification-only callers like the router that don't
        need room for a long generated reply.
    """
    target_limit = context_limit - _SAFETY_BUFFER
    scope_tag = "session" if persistent else "local-only, not saved"

    def get_current_total():
        chunk_tokens = sum(c.get("tokens", 0) for c in active_db_chunks)
        history_tokens = sum(h.get("tokens", 0) for h in chat_history)
        return system_base_tokens + history_tokens + chunk_tokens + query_tokens + generation_buffer

    current_total = get_current_total()

    while current_total > target_limit and len(active_db_chunks) > 2:
        evicted_chunk = active_db_chunks.pop(0)
        current_total -= evicted_chunk.get("tokens", 0)
        print(f"[{label}] ⚠️ Context tight ({scope_tag}). Evicted old DB Chunk ID: {evicted_chunk.get('id')} (-{evicted_chunk.get('tokens', 0)} tokens).")

    while current_total > target_limit and len(chat_history) >= 2:
        evicted_user = chat_history.pop(0)
        evicted_ai = chat_history.pop(0)
        freed = evicted_user.get("tokens", 0) + evicted_ai.get("tokens", 0)
        current_total -= freed
        print(f"[{label}] ⚠️ Context tight ({scope_tag}). Evicted old Chat Turn (-{freed} tokens).")
        
    return current_total

def draw_token_bar(used: int, context_limit: int):
    """Draws a color-coded CLI progress bar scaled to the provided context_limit."""
    if used == 0 or context_limit <= 0: 
        return
        
    percentage = min(used / context_limit, 1.0)
    bar_length = 30
    filled = int(bar_length * percentage)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    color = "\033[92m" # Green
    if percentage > 0.60: color = "\033[93m" # Yellow
    if percentage > 0.85: color = "\033[91m" # Red
    reset = "\033[0m"
    
    print(f"\n{color}[Context Memory: {bar} {int(percentage*100)}% ({used}/{context_limit} Tokens)]{reset}\n")