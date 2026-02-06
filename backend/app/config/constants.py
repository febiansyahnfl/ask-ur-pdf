"""Application Constants"""

# File size limits
MAX_FILE_SIZE = 10 * 1024 * 1024 # 10MB in bytes

# Supported file types
ALLOWED_EXTENSIONS = ["pdf"]
ALLOWED_MIME_TYPES = ["application/pdf"]

# Chunking defaults
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# Session defaults 
DEFAULT_SESSION_LIFETIME_HOURS = 2

# API defaults 
DEFAULT_TIMEOUT = 30 #seconds
MAX_RETRIES = 3

# Paths
DATA_DIR = "data"
SESSIONS_DIR = f"{DATA_DIR}/sessions"
CACHE_DIR = f"{DATA_DIR}/cache"
EMBEDDINGS_DIR = f"{DATA_DIR}/embeddings"
VECTOR_DB_DIR = f"{DATA_DIR}/vector_db"
LOGS_DIR = "logs"