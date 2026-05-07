"""
Configuration file for Real Estate RAG Assistant
Customize these settings for your specific needs
"""

# ============================================
# OLLAMA CONFIGURATION
# ============================================

# Ollama server address
OLLAMA_BASE_URL = "http://localhost:11434"

# LLM Model to use
# Options: mistral, neural-chat, llama2, orca-mini, etc.
LLM_MODEL = "mistral"

# Embedding model to use
# Options: nomic-embed-text (recommended), all-minilm, etc.
EMBEDDING_MODEL = "nomic-embed-text"

# LLM parameters
LLM_TEMPERATURE = 0.3  # Lower = more consistent, Higher = more creative
LLM_TOP_P = 0.9
LLM_TOP_K = 40

# ============================================
# RETRIEVAL CONFIGURATION
# ============================================

# Vector database settings
VECTOR_DB_PATH = "./chroma_db"
VECTOR_DB_COLLECTION = "real_estate"

# Retrieval settings
RETRIEVAL_SEARCH_TYPE = "mmr"  # Options: similarity, mmr (Maximum Marginal Relevance)
RETRIEVAL_K = 5  # Number of top results to retrieve
RETRIEVAL_FETCH_K = 20  # Number to fetch before ranking

# Use compression retriever
USE_COMPRESSION = True

# ============================================
# DOCUMENT PROCESSING
# ============================================

# Text chunking parameters
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
SEPARATORS = ["\n\n", "\n", " ", ""]  # Preferred splitting points

# PDF extraction parameters
PDF_EXTRACT_TEXT = True
PDF_PAGE_LIMIT = None  # None = all pages

# Web scraping parameters
WEB_SCRAPE_TIMEOUT = 10  # Seconds
WEB_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# ============================================
# STREAMLIT CONFIGURATION
# ============================================

# Page settings
PAGE_TITLE = "🏠 Real Estate RAG Assistant"
PAGE_ICON = "🏠"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Max chat history to keep
MAX_CHAT_HISTORY = 50

# Upload file size limit (in MB)
MAX_UPLOAD_SIZE = 200

# ============================================
# REAL ESTATE SPECIFIC
# ============================================

# Property type options
PROPERTY_TYPES = [
    "Any",
    "House",
    "Apartment",
    "Condo",
    "Townhouse",
    "Commercial",
    "Land",
    "Multi-family"
]

# Budget range for search (in dollars)
MIN_BUDGET = 50000
MAX_BUDGET = 5000000
DEFAULT_MIN_BUDGET = 200000
DEFAULT_MAX_BUDGET = 500000

# Features to highlight in search
COMMON_FEATURES = [
    "Pool",
    "Garage",
    "Garden",
    "Patio",
    "Balcony",
    "Waterfront",
    "Views",
    "Renovated",
    "Modern Kitchen",
    "Basement"
]

# ============================================
# PROMPT TEMPLATES
# ============================================

# System prompt for real estate assistant
SYSTEM_PROMPT = """You are a helpful real estate assistant with expertise in property listings, 
market analysis, and investment opportunities. Provide accurate, helpful information about properties, 
neighborhoods, and real estate market trends. Always cite your sources when providing information."""

# Query prompt for property recommendations
PROPERTY_RECOMMENDATION_PROMPT = """Based on the provided property information and the user's criteria, 
recommend the most suitable properties. Explain your recommendations clearly."""

# ============================================
# ADVANCED SETTINGS
# ============================================

# Enable chat memory
ENABLE_CHAT_MEMORY = True

# Enable debug mode (verbose output)
DEBUG_MODE = False

# Log queries for analytics
LOG_QUERIES = True
LOG_FILE = "./query_logs.txt"

# ============================================
# MODEL PERFORMANCE PRESETS
# ============================================

PRESETS = {
    "fast": {
        "model": "orca-mini",
        "chunk_size": 800,
        "retrieval_k": 3,
        "temperature": 0.3
    },
    "balanced": {
        "model": "mistral",
        "chunk_size": 1000,
        "retrieval_k": 5,
        "temperature": 0.3
    },
    "quality": {
        "model": "llama2",
        "chunk_size": 1200,
        "retrieval_k": 7,
        "temperature": 0.2
    }
}

# Default preset
DEFAULT_PRESET = "balanced"

# ============================================
# CUSTOM SETTINGS
# ============================================

# Add your custom settings below
CUSTOM_INSTRUCTION = "Focus on providing detailed property information with market insights."
ENABLE_PROPERTY_COMPARISON = True
ENABLE_INVESTMENT_ANALYSIS = True
