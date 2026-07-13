import os
import shutil
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SOURCES_DIR = os.path.join(PROJECT_ROOT, "sources")
PDF_DIR = os.path.join(SOURCES_DIR, "raw_pdfs")

# Ensure the audit folders exist
os.makedirs(SOURCES_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
def log_ingestion(source_type: str, data: str):
    """Logs the URL or filename to the respective text file with a timestamp."""
    log_file = os.path.join(SOURCES_DIR, f"{source_type}.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {data}\n")

def backup_pdf(pdf_path: str):
    """Copies the raw PDF into the sources/raw_pdfs folder."""
    if not os.path.exists(pdf_path):
        return
        
    filename = os.path.basename(pdf_path)
    dest_path = os.path.join(PDF_DIR, filename)
    
    shutil.copy2(pdf_path, dest_path)
    log_ingestion("pdf", filename)