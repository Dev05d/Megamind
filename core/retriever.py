import sys
import ollama
from qdrant_client import models
from fastembed import SparseTextEmbedding

# Retaining your original configuration architecture
from core.config import DB_PATH, COLLECTION_NAME, EMBED_MODEL, CHAT_MODEL
from core.vector_store import _get_client

_SYSTEM_PROMPT = """You are Megamind, a precise personal knowledge assistant.
Answer ONLY from the provided context. If the context doesn't contain enough
information to answer confidently, say so clearly rather than guessing. 
At the end of your response, provide a brief bulleted list of the exact Sources you referenced."""

_CONTEXT_LIMIT = 8192

# Instantiate the local BM25 keyword mapper globally
_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def _estimate_tokens(text: str) -> int:
    """Standard production heuristic for English text: ~4 characters per token."""
    return len(text) // 4


def _prune_history_to_fit(chat_history: list[dict], incoming_context_tokens: int, reserve_tokens: int = 1024) -> None:
    """Audits chat history and evicts old turns to ensure the payload fits the context window."""
    system_tokens = _estimate_tokens(chat_history[0]["content"])
    
    while len(chat_history) > 1:
        history_tokens = sum(_estimate_tokens(msg["content"]) for msg in chat_history[1:])
        total_estimated_tokens = system_tokens + history_tokens + incoming_context_tokens + reserve_tokens
        
        if total_estimated_tokens <= _CONTEXT_LIMIT:
            break
            
        if len(chat_history) > 2:
            print(f"\n[Memory Manager] ⚠️ Context tight ({total_estimated_tokens}/{_CONTEXT_LIMIT} estimated tokens). Evicting oldest Q&A turn.")
            del chat_history[1:3]  # Evicts the oldest user/assistant pair
        else:
            break


def _draw_token_bar(used: int, total: int = _CONTEXT_LIMIT):
    """Draws a color-coded CLI progress bar for token usage."""
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


def _classify_query(query: str, chat_history: list[dict]) -> str:
    """
    Evaluates whether the incoming query requires a fresh database lookup (SEARCH)
    or is a follow-up conversation relying on previous assistant replies (CONVERSATIONAL).
    """
    # First turn always defaults to a database search
    if len(chat_history) <= 1:
        return "SEARCH"

    # Grab the last assistant output for immediate evaluation context
    last_assistant_turn = chat_history[-1]["content"] if chat_history[-1]["role"] == "assistant" else ""
    
    classifier_prompt = f"""Analyze the User Query and determine if it requires searching an external database for new facts, or if it is a direct conversational follow-up/instruction regarding the previous discussion.

Previous Assistant Reply: "{last_assistant_turn}"
User Query: "{query}"

Respond with EXACTLY one word from these options:
- SEARCH : If the user is asking a new question that requires looking up facts, documents, or external data.
- CONVERSATIONAL : If the user is asking to summarize, clarify, format, translate, or expand on the previous message, or if it's general greeting/chit-chat.

Output only the uppercase word:"""

    try:
        response = ollama.generate(
            model=CHAT_MODEL, 
            prompt=classifier_prompt,
            options={"temperature": 0.0}  # Force deterministic categorization
        )
        decision = response.get("response", "").strip().upper()
        
        if "SEARCH" in decision:
            return "SEARCH"
        return "CONVERSATIONAL"
    except Exception:
        # Fail-safe path: if the classifier hits an error, default to a full database search
        return "SEARCH"


def ask_megamind(initial_question: str, top_k: int = 5) -> None:
    chat_history = [{"role": "system", "content": _SYSTEM_PROMPT}]
    current_question = initial_question
    
    print("\n[Megamind] 🟢 Entering continuous chat. Type 'exit' or 'q' to return to menu.")
    
    while True:
        if current_question.lower() in ['exit', 'quit', 'q']:
            print("\n[Megamind] Ending chat session...")
            break
            
        # 1. Run the Query Router Classifier
        intent = _classify_query(current_question, chat_history)
        
        if intent == "SEARCH":
            print(f"\n[Megamind] 🔍 Routing: [SEARCH] -> Fetching new context for: '{current_question}'")
            try:
                # Generate Dense Vector (Ollama)
                query_dense = ollama.embed(model=EMBED_MODEL, input=current_question)["embeddings"][0]

                # Generate Sparse Vector (FastEmbed BM25)
                sparse_gen = next(_sparse_model.embed([current_question]))
                query_sparse = models.SparseVector(
                    indices=sparse_gen.indices.tolist(),
                    values=sparse_gen.values.tolist()
                )

                # Construct hybrid prefetch queries
                search_prefetches = [
                    models.Prefetch(query=query_dense, using="dense", limit=top_k)
                ]
                
                if len(sparse_gen.indices) > 0:
                    search_prefetches.append(
                        models.Prefetch(query=query_sparse, using="sparse", limit=top_k)
                    )

                # Execute combined Reciprocal Rank Fusion query against Qdrant
                response = _get_client().query_points(
                    collection_name=COLLECTION_NAME,
                    prefetch=search_prefetches,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                )
                results = response.points

            except Exception as e:
                print(f"\n[Megamind] ❌ Database or Embedding Error: {e}")
                current_question = input("\nYou: ").strip()
                continue

            if not results:
                print("\n[Megamind] I don't have any relevant information on that topic yet.\n")
                context_text = "No relevant documents found."
            else:
                context_blocks = [
                    f"[Source: {r.payload.get('source', 'unknown')}]\n{r.payload.get('text', '')}" 
                    for r in results
                ]
                context_text = "\n\n---\n\n".join(context_blocks)
        else:
            # CONVERSATIONAL Route: Entirely skip dense/sparse model embeddings and DB lookups
            print(f"\n[Megamind] 🧠 Routing: [CONVERSATIONAL] -> Processing via history context.")
            context_text = "User is referencing conversation history or asking a contextual follow-up. Rely strictly on existing chat history."

        # 2. Manage memory allocations before compiling the LLM payload
        incoming_context_tokens = _estimate_tokens(context_text)
        _prune_history_to_fit(chat_history, incoming_context_tokens, reserve_tokens=1024)

        # 3. Assemble full prompt payload
        user_message_content = f"Context:\n{context_text}\n\nQuestion: {current_question}"
        chat_history.append({"role": "user", "content": user_message_content})

        print("\n[Megamind] ", end="", flush=True)
        
        full_response = ""
        total_tokens = 0
        
        try:
            stream = ollama.chat(
                model=CHAT_MODEL,
                messages=chat_history,
                stream=True
            )
            
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_response += content
                
                if chunk.get("done"):
                    total_tokens = chunk.get("prompt_eval_count", 0) + chunk.get("eval_count", 0)

        except Exception as e:
            print(f"\n[Megamind] ❌ LLM Generation Error: {e}")
            # Rollback the history stack to prevent corrupted states on network dropped calls
            chat_history.pop() 
            current_question = input("\nYou: ").strip()
            continue

        print()
        
        # Save Assistant reply
        chat_history.append({"role": "assistant", "content": full_response})
        
        # Context Stripping Hack: Revert prompt back to clean question to save memory allocation
        chat_history[-2]["content"] = current_question 
        
        # Fallback evaluation tracking if the metadata chunk fails or drops mid-stream
        if total_tokens == 0:
            history_str = "".join([msg["content"] for msg in chat_history])
            total_tokens = _estimate_tokens(history_str)
            
        _draw_token_bar(total_tokens)
        
        try:
            current_question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Megamind] Ending chat session...")
            break
            
        if not current_question:
            print("\n[Megamind] Ending chat session...")
            break