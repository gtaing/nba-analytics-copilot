DB_PATH = "db/nba.duckdb"
RAW_DATA_PATH = "data/player_stats_2016.csv"

MIN_GAMES = 50
HIGH_USAGE_THRESHOLD = 25.0

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM Configuration
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Graph Configuration
MAX_ITERATIONS = 5

# RAG Configuration
SIMILARITY_THRESHOLD = 0.15