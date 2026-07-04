import os
from dotenv import load_dotenv

# 1. Find the absolute path to the project root (one folder up from core/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Force load_dotenv to read the .env file from the project root
load_dotenv(os.path.join(BASE_DIR, ".env"))

# 3. Handle the DB_PATH from .env safely
env_db_path = os.getenv("DB_PATH", "megamind_db")

# If the path from .env is absolute (e.g., C:/...) keep it. 
# Otherwise, force it to anchor to the project's root folder!
if os.path.isabs(env_db_path):
    DB_PATH = env_db_path
else:
    DB_PATH = os.path.normpath(os.path.join(BASE_DIR, env_db_path))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "megamind")
VECTOR_SIZE     = int(os.getenv("VECTOR_SIZE", 768))

# AI Models (all served locally via Ollama)
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL      = os.getenv("CHAT_MODEL",  "llama3")
OCR_MODEL       = os.getenv("OCR_MODEL",   "glm-ocr")

OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_HOST