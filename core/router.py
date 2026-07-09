import ollama
from pydantic import BaseModel, Field
from typing import Optional
from core.config import ROUTER_MODEL

class RouterResponse(BaseModel):
    route: str = Field(
        description="Must be one of: 'chitchat', 'context_sufficient', 'needs_retrieval', 'needs_novel_retrieval'"
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description="If route is 'needs_novel_retrieval' or 'needs_retrieval', rewrite the query to be standalone. Otherwise, leave null."
    )

def route_query(user_query: str, chat_history: list[dict], active_db_chunks: list[dict]) -> RouterResponse:
    # 1. Dynamically build context blocks to minimize token overhead
    context_blocks = []

    if active_db_chunks:
        chunks_subset = active_db_chunks[-4:]
        chunks_str = "\n\n".join([c["text"] for c in chunks_subset])
        context_blocks.append(f"=== ACTIVE DATABASE CHUNKS IN MEMORY ===\n{chunks_str}")

    if chat_history:
        history_subset = chat_history[-4:] 
        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_subset])
        context_blocks.append(f"=== RECENT CHAT HISTORY ===\n{history_str}")

    context_blocks.append(f"=== USER LATEST QUERY ===\n{user_query}")
    user_message = "\n\n".join(context_blocks)

    # 2. Rich system prompt combining the best classification examples and JSON directives
    system_prompt = """
    You are the central query router for an advanced RAG system. 
    Analyze the user's latest query, context, and classify it into exactly one category.

    CATEGORIES:
    - "chitchat": Greetings, thanks, or small talk. (e.g., "hi", "hello", "thanks!", "how are you?")
    - "context_sufficient": Summarizing, reformatting, or manipulating information ALREADY present in the active database chunks or history. (e.g., "summarize that text", "translate your last answer to Spanish")
    - "needs_retrieval": Factual questions introducing a completely new topic or external knowledge. (e.g., "What is quantum computing?", "Tell me about the Roman Empire.")
    - "needs_novel_retrieval": Asking to go deeper or find NEW facts about the current topic that aren't answered by the active chunks. (e.g., "tell me more about this", "go deeper on that last point")

    REWRITING RULES:
    If category is "needs_retrieval" or "needs_novel_retrieval", provide a standalone `rewritten_query` replacing pronouns (he, she, it, this) with real subjects from history. Otherwise, set `rewritten_query` to null.

    RESPONSE FORMAT:
    You must output a single valid JSON object matching this schema layout:
    {"route": "chitchat", "rewritten_query": null}
    """

    try:
        response = ollama.chat(
            model=ROUTER_MODEL, 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            format=RouterResponse.model_json_schema(),
            options={
                'temperature': 0.1,  
                'seed': 42
            } 
        )
        
        return RouterResponse.model_validate_json(response.message.content)
    except Exception as e:
        print(f"[Router] ⚠️ Parsing or Ollama failure: {e}")
        
        # Smart deterministic fallback logic
        cleaned_query = user_query.lower().strip()
        if cleaned_query in ["hi", "hello", "hey", "thanks", "thank you"]:
            return RouterResponse(route="chitchat", rewritten_query=None)
            
        return RouterResponse(route="needs_retrieval", rewritten_query=user_query)