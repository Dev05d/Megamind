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
VECTOR_SIZE     = int(os.getenv("VECTOR_SIZE", 2560))

# AI Models (all served locally via Ollama)
EMBED_MODEL     = os.getenv("EMBED_MODEL", "qwen3-embedding:4b")
CHAT_MODEL      = os.getenv("CHAT_MODEL",  "llama3.2:3b")
CONTEXT_LIMIT   = os.getenv("CONTEXT_LIMIT", 8192)
ROUTER_MODEL    = os.getenv("ROUTER_MODEL", "llama3.2:3b" )
ROUTER_CONTEXT  = os.getenv("ROUTER_CONTEXT", 4096)
OCR_MODEL       = os.getenv("OCR_MODEL",   "glm-ocr")


## ADVANCED SETTINGS ##
TOP_K_CHUNKS    = os.getenv("TOP_K_CHUNKS", 5)
DENSE_THRESHOLD = os.getenv("DENSE_THRESHOLD", 0.5)


OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_HOST