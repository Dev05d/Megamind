import os
import io

import fitz         
import ollama
from PIL import Image

from core.config import OCR_MODEL
from core.vector_store import ingest_text


def _page_to_jpeg_bytes(page: fitz.Page, dpi: int = 150) -> bytes:
    """Renders a PDF page to a high-quality JPEG byte string for OCR."""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _ocr_page(page: fitz.Page) -> str:
    """Sends a rendered page image directly to the local OCR model with anti-looping safeguards."""
    image_bytes = _page_to_jpeg_bytes(page)
    
    response = ollama.chat(
        model=OCR_MODEL,
        messages=[{
            "role":    "user",
            "content": "Extract all text, math equations, and formatting from this image as clean Markdown. Do NOT describe the image and do NOT repeat text.",
            "images":  [image_bytes],
        }],
        options={
            "temperature": 0.0,       
            "repeat_penalty": 1.2,    
            "num_predict": 2048      
        }
    )
    
    raw_text = response["message"]["content"]
    
    lines = raw_text.splitlines()
    clean_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            clean_lines.append(line)
    
        elif not clean_lines or line_stripped != clean_lines[-1].strip():
            clean_lines.append(line)
            
    return "\n".join(clean_lines)


def scrape_pdf(file_path: str) -> str:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc  = fitz.open(file_path)
    name = os.path.basename(file_path)
    print(f"\n[PDF] Opening '{name}' — {len(doc)} pages (OCR).")

    pages_text: list[str] = []

    for i, page in enumerate(doc):
        print(f"  -> Page {i + 1}/{len(doc)}: Running visual OCR model...")
        try:
            extracted_text = _ocr_page(page)
            pages_text.append(extracted_text)
        except Exception as e:
            print(f"  -> [OCR Error on page {i + 1}]: {e}")

    return "\n\n".join(pages_text)


def ingest_pdf(file_path: str) -> None:
    """Main entry point to scrape and save a PDF."""
    text = scrape_pdf(file_path)
    
    if not text.strip():
        print(f"\n[Warning] The OCR model failed to extract any text from '{file_path}'.")
        return
        
    source_name = f"PDF: {os.path.basename(file_path)}"
    ingest_text(
        text,
        source=source_name,
        extra_payload={"type": "pdf", "file_path": file_path},
    )