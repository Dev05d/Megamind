import urllib.request
import json

from core.vector_store import ingest_text

import json
from playwright.sync_api import sync_playwright
from core.vector_store import ingest_text

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
_MAX_COMMENT_DEPTH = 3


def _fetch_json(url: str) -> list:
    """Uses a stealth headless browser to bypass Reddit's TLS fingerprinting and fetch the .json data."""
    api_url = url.split("?")[0].rstrip("/") + ".json"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"}
        )
        page = context.new_page()
        
        print("  -> Loading normal page to pass security checks...")
        
        # 1. Navigate to the NORMAL reddit page first (not the .json)
        # This solves the web firewall challenges and gets valid session cookies.
        page.goto(url, wait_until="domcontentloaded")
        
        # Wait a brief moment (2 seconds) for background security scripts to clear
        page.wait_for_timeout(2000) 
        
        print("  -> Fetching background JSON with trusted session...")
        
        # 2. Run a background fetch() from inside the trusted page context!
        # (Reddit's firewall sees this as a legitimate background data load)
        raw_text = page.evaluate(f'''async () => {{
            const response = await fetch("{api_url}");
            return await response.text();
        }}''')
        
        browser.close()
        
        # --- VERBOSE DEBUG BLOCK ---
        if not raw_text.strip().startswith("[") and not raw_text.strip().startswith("{"):
            print("\n[Verbose Debug] Reddit did NOT return JSON. Here is what they sent instead:")
            print("========================================")
            print(raw_text[:1000]) 
            print("========================================\n")
            raise ValueError("Failed to parse Reddit JSON. See debug output above.")
        # ---------------------------
        
        return json.loads(raw_text)

def _extract_comments(children: list, depth: int = 0) -> str:
    """Recursively flattens a comment tree into readable plain text."""
    if depth > _MAX_COMMENT_DEPTH:
        return ""

    lines = []
    indent = "  " * depth

    for child in children:
        if child.get("kind") != "t1":
            continue

        data   = child["data"]
        author = data.get("author", "[deleted]")
        body   = data.get("body",   "").strip()
        score  = data.get("score",  0)

        if not body or body in ("[deleted]", "[removed]"):
            continue

        lines.append(f"{indent}[{author} | ↑{score}]")
        lines.append(f"{indent}{body}\n")

        replies = data.get("replies")
        if isinstance(replies, dict):
            nested = replies["data"]["children"]
            lines.append(_extract_comments(nested, depth + 1))

    return "\n".join(lines)


def scrape_reddit(url: str) -> str:
    """
    Scrapes a single Reddit post and its comments using the public JSON API.
    No API key, PRAW, or credentials required.

    Args:
        url: Full URL to a Reddit post.
             e.g. https://www.reddit.com/r/python/comments/xyz/title/

    Returns:
        Formatted document string ready for chunking and ingestion.
    """
    print(f"[Reddit] Fetching: {url}")
    data = _fetch_json(url)

    post_data = data[0]["data"]["children"][0]["data"]
    comments  = data[1]["data"]["children"]

    subreddit = post_data.get("subreddit",  "unknown")
    title     = post_data.get("title",      "Untitled")
    score     = post_data.get("score",      0)
    author    = post_data.get("author",     "[deleted]")
    selftext  = post_data.get("selftext",   "").strip()
    post_url  = "https://www.reddit.com" + post_data.get("permalink", "")

    body_section = selftext if selftext else "[Link post — no body text]"
    comment_text = _extract_comments(comments)

    return (
        f"[Source: Reddit]\n"
        f"Subreddit: r/{subreddit}\n"
        f"Title:     {title}\n"
        f"Author:    u/{author}\n"
        f"Score:     {score}\n"
        f"URL:       {post_url}\n\n"
        f"[Post Body]\n{body_section}\n\n"
        f"[Comments]\n{comment_text}"
    ).strip()


def ingest_reddit(url: str) -> None:
    """Scrape a Reddit post and save it to the vector database."""
    text = scrape_reddit(url)
    ingest_text(
        text,
        source=url,
        extra_payload={"type": "reddit", "url": url},
    )