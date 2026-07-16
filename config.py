import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"
EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
CHROMA_PATH = "chroma"
DATA_PATH = "data"
RETRIEVAL_THRESHOLD = 1.8      # loose vector-search pre-filter (wide net for recall)
RERANKER_THRESHOLD = 0.1       # cross-encoder minimum score to keep a chunk
RETRIEVAL_K = 8            # candidate pool size fed to the reranker
RERANKER_TOP_N = 5             # final number of chunks sent to the answer LLM
NUM_OF_RETRIEVED_CHUNKS = 6    
STORAGE_PROVIDER='minio'