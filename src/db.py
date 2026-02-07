import duckdb
from src.config import DB_PATH

def get_connection():
    return duckdb.connect(DB_PATH)