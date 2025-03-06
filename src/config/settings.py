# filepath: /research-paper-retrieval-system/research-paper-retrieval-system/src/config/settings.py

DATABASE_URI = "sqlite:///data/paper_index.db"
API_KEY = "your_api_key_here"
SEARCH_ENGINE = "BM25"  # Options: BM25, FAISS
EMBEDDING_MODEL = "bert-base-uncased"  # Options: bert-base-uncased, sentence-transformers/all-MiniLM-L6-v2
MAX_RESULTS = 10
TIMEOUT = 30  # seconds for API requests