from core.browser import fetch_and_clean
from core.vector_store import ingest_text

def scrape_web(url: str) -> str:
    print(f"[Web] Fetching via Browser Engine: {url}")
    return fetch_and_clean(url)

def ingest_web(url: str) -> None:
    text = scrape_web(url)

    # --- DEBUG LINES START ---
    print(f"[Debug] Total characters fetched: {len(text)}")
    print(f"[Debug] Preview of text:\n{text[:500]}\n--- End Preview ---")
    # --- DEBUG LINES END ---

    ingest_text(
        text,
        source=url,
        extra_payload={"type": "web", "url": url},
    )