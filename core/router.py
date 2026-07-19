import ollama
from pydantic import BaseModel, Field
from typing import Optional
from core import config
from core.memory_manager import get_exact_tokens, enforce_token_budget

class RouterResponse(BaseModel):
    route: str = Field(
        description="Must be one of: 'chitchat', 'context_sufficient', 'needs_retrieval', 'needs_novel_retrieval'"
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description="If route is 'needs_novel_retrieval' or 'needs_retrieval', rewrite the query to be standalone. Otherwise, leave null."
    )

def route_query(user_query: str, chat_history: list[dict], active_db_chunks: list[dict]) -> RouterResponse:
    system_prompt = """
    You are the central query router for an advanced RAG system. 
    Analyze the user's latest query, context, and classify it into exactly one category.

    CATEGORIES:
    - "chitchat": Greetings, thanks, or small talk. (e.g., "hi", "hello", "thanks!", "how are you?")
    - "context_sufficient": Summarizing, reformatting, or manipulating information ALREADY present in the active database chunks or history. (e.g., "summarize that text", "translate your last answer to Spanish")
    - "needs_retrieval": Factual questions introducing a completely new topic or external knowledge. (e.g., "What is quantum computing?", "Tell me about the Roman Empire.")
    - "needs_novel_retrieval": Asking to go deeper or find NEW facts about the current topic that aren't answered by the active chunks. (e.g., "tell me more about this", "go deeper on that last point")

    REWRITING RULES:
    If category is "needs_retrieval" or "needs_novel_retrieval", the rewritten_query MUST be a
    standalone, topically-specific query -- never a bare continuation phrase.
    - If the query contains pronouns (he, she, it, this), replace them with the real subject from history.
    - If the query is a vague continuation with no explicit topic ("more info", "tell me more",
    "go deeper", "what else"), synthesize a specific standalone query using the actual subject
    of the current conversation topic from history (e.g. "more info" about a 3D coordinate
    system discussion -> "additional details on the 3D coordinate system, axes, and origin").
    rewritten_query must never be null or empty when the route is needs_retrieval or needs_novel_retrieval.

    RESPONSE FORMAT:
    You must output a single valid JSON object matching this schema layout:
    {"route": "chitchat", "rewritten_query": null}
    """

    # Parse and establish context limits locally
    router_limit = int(config.ROUTER_CONTEXT)
    system_base_tokens = get_exact_tokens(system_prompt, config.ROUTER_MODEL)
    query_tokens = get_exact_tokens(user_query, config.ROUTER_MODEL)
    router_token_key = f"tokens__{config.ROUTER_MODEL}"

    chunks_working_set = []
    for chunk in active_db_chunks:
        chunk_copy = chunk.copy()
        if router_token_key not in chunk_copy:
            chunk_copy[router_token_key] = get_exact_tokens(chunk_copy["text"], config.ROUTER_MODEL)
        # model-specific count without disturbing the original cached value.
        chunk_copy["tokens"] = chunk_copy[router_token_key]
        chunks_working_set.append(chunk_copy)

    history_working_set = []
    for msg in chat_history:
        msg_copy = msg.copy()
        if router_token_key not in msg_copy:
            formatted_msg = f"{msg_copy['role'].capitalize()}: {msg_copy['content']}"
            msg_copy[router_token_key] = get_exact_tokens(formatted_msg, config.ROUTER_MODEL)
        msg_copy["tokens"] = msg_copy[router_token_key]
        history_working_set.append(msg_copy)


    total_estimated_tokens = enforce_token_budget(
        system_base_tokens=system_base_tokens,
        query_tokens=query_tokens,
        active_db_chunks=chunks_working_set,
        chat_history=history_working_set,
        context_limit=router_limit,
        label="Router",
        persistent=False,
        generation_buffer=0
    )



    context_blocks = []

    if chunks_working_set:
        chunks_str = "\n\n".join([c["text"] for c in chunks_working_set])
        context_blocks.append(f"=== ACTIVE DATABASE CHUNKS IN MEMORY ===\n{chunks_str}")

    if history_working_set:
        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_working_set])
        context_blocks.append(f"=== RECENT CHAT HISTORY ===\n{history_str}")

    context_blocks.append(f"=== USER LATEST QUERY ===\n{user_query}")
    user_message = "\n\n".join(context_blocks)

    try:
        response = ollama.chat(
            model=config.ROUTER_MODEL, 
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
        cleaned_query = user_query.lower().strip()
        if cleaned_query in ["hi", "hello", "hey", "thanks", "thank you"]:
            return RouterResponse(route="chitchat", rewritten_query=None)
        return RouterResponse(route="needs_retrieval", rewritten_query=user_query)