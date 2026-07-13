import ollama
from qdrant_client import models
from fastembed import SparseTextEmbedding
from core.router import route_query
from core import config
from core.vector_store import _get_client
from core.memory_manager import get_exact_tokens, enforce_token_budget, draw_token_bar

_SYSTEM_PROMPT = """You are Megamind, a precise personal knowledge assistant.
Answer ONLY from the provided context. If the context doesn't contain enough
information to answer confidently, say so clearly rather than guessing. 
At the end of your response, provide a brief bulleted list of the exact Sources you referenced."""

# Instantiate the local BM25 keyword mapper globally
_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def ask_megamind(initial_question: str, top_k: int = None) -> None:
    if top_k is None:
        top_k = int(config.TOP_K_CHUNKS)
    chat_history = []
    active_db_chunks = []
    system_base_tokens = get_exact_tokens(_SYSTEM_PROMPT, config.CHAT_MODEL)
    current_question = initial_question
    
    # Establish the explicit token limit integer for this pipeline
    main_limit = int(config.CONTEXT_LIMIT)

    print("\n[Megamind] 🟢 Entering continuous chat. Type 'exit' or 'q' to return to menu.")
    
    while True:
        if current_question.lower() in ['exit', 'quit', 'q']:
            print("\n[Megamind] Ending chat session...")
            break
            
        query_tokens = get_exact_tokens(current_question, config.CHAT_MODEL)
        
        # Router checks bounds against its own config limits internally
        router_decision = route_query(current_question, chat_history, active_db_chunks)
        intent = router_decision.route
        search_query = router_decision.rewritten_query or current_question
        
        if intent in ["needs_retrieval", "needs_novel_retrieval"]:
            print(f"\n[Megamind] 🔍 Routing: [{intent.upper()}]")
            print(f"           -> Search Target: '{search_query}'")
            
            try:
                query_dense = ollama.embed(model=config.EMBED_MODEL, input=search_query)["embeddings"][0]
                sparse_gen = next(_sparse_model.embed([search_query]))
                query_sparse = models.SparseVector(
                    indices=sparse_gen.indices.tolist(),
                    values=sparse_gen.values.tolist()
                )

                search_prefetches = [
                    models.Prefetch(
                        query=query_dense, 
                        using="dense", 
                        limit=top_k,
                        score_threshold=config.DENSE_THRESHOLD  
                    )
                ]
                if len(sparse_gen.indices) > 0:
                    search_prefetches.append(
                        models.Prefetch(
                            query=query_sparse, 
                            using="sparse", 
                            limit=top_k
                        )
                    )
                query_filter = None
                if intent == "needs_novel_retrieval" and active_db_chunks:
                    existing_ids = [c["id"] for c in active_db_chunks]
                    query_filter = models.Filter(
                        must_not=[models.HasIdCondition(has_id=existing_ids)]
                    )
                    print(f"           -> 🚫 Excluding {len(existing_ids)} active chunk IDs from Qdrant search.")

                response = _get_client().query_points(
                    collection_name=config.COLLECTION_NAME,
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
                        
                        c_tokens = get_exact_tokens(chunk_text, config.CHAT_MODEL)
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

        # Enforce token budget with explicit context limit parameter
        enforce_token_budget(
            system_base_tokens=system_base_tokens, 
            query_tokens=query_tokens, 
            active_db_chunks=active_db_chunks, 
            chat_history=chat_history,
            context_limit=main_limit,
            label="Retriever",
            persistent=True
        )

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
                model=config.CHAT_MODEL,
                messages=llm_payload,
                stream=True,
                options={"num_ctx": main_limit}
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
            ai_eval_count = get_exact_tokens(full_response, config.CHAT_MODEL)
            
        chat_history.append({"role": "assistant", "content": full_response, "tokens": ai_eval_count})
        
        total_active_now = (
            system_base_tokens + 
            sum(c["tokens"] for c in active_db_chunks) + 
            sum(h["tokens"] for h in chat_history)
        )
        
        # Render the token status bar relative to the main loop context limits
        draw_token_bar(used=total_active_now, context_limit=main_limit)
        
        try:
            current_question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Megamind] Ending chat session...")
            break
            
        if not current_question:
            print("\n[Megamind] Ending chat session...")
            break