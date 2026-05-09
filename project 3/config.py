# Configuration for E-Commerce Chatbot
import os
from dotenv import load_dotenv

load_dotenv()

# ============ OpenAI Configuration ============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4-turbo"
OPENAI_TEMPERATURE = 0.7

# ============ MySQL Configuration ============
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "ecommerce_chatbot")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))

# ============ ChromaDB Configuration ============
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = "products"

# ============ Chatbot Configuration ============
MAX_CHAT_HISTORY = 10
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ============ Database Schema Paths ============
SCHEMA_PATH = "./init_db.sql"
SAMPLE_DATA_PATH = "./sample_products.csv"

# ============ Logging Configuration ============
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
