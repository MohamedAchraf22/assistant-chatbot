import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"
EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
CHROMA_PATH = "chroma"
DATA_PATH = "data"
RETRIEVAL_THRESHOLD = 0.85     # loose vector-search pre-filter (wide net for recall)
RERANKER_THRESHOLD = 0.1       # cross-encoder minimum score to keep a chunk
RETRIEVAL_K = 25           # TEMP DEBUG: widened to 100 for retrieval-report debugging (was 25)
RERANKER_TOP_N = 25           # final number of chunks sent to the answer LLM
RERANKER_ENABLED = False       # set True to re-enable the reranking stage
NUM_OF_RETRIEVED_CHUNKS = 6
STORAGE_PROVIDER='minio'

# File types the ingestion pipeline is allowed to load. Anything else found
# in the MinIO bucket (.DS_Store, images, archives, etc.) is skipped before
# it is ever downloaded or decoded. Add more extensions here as support for
# them is implemented (e.g. ".txt", ".pdf").
SUPPORTED_EXTENSIONS = {".md"}