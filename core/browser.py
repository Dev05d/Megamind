# core/browser.py
from playwright.sync_api import sync_playwright
from readability import Document
from markdownify import markdownify as md

def fetch_and_clean(url: str) -> str:
    """Uses a disguised Playwright instance to bypass anti-bot screens and fetch clean text."""
    with sync_playwright() as p:
        # 1. Use a standard user-agent string
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        
        # 2. Launch browser with standard arguments
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"] # Hides automated flag
        )
        
        # 3. Use a persistent or customized context to handle cookies properly
        context = browser.new_context(
            user_agent=user_agent,
            viewport={"width": 1280, "height": 720},
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"}
        )
        
        page = context.new_page()
        
        # 4. Fallback back to "networkidle" to let Medium's scripts process
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        
        html = page.content()
        browser.close()
        
        # Clean HTML and convert to Markdown
        doc = Document(html)
        return md(doc.summary())
