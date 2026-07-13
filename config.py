import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"
EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
CHROMA_PATH = "chroma"
DATA_PATH = "data"
RETRIEVAL_THRESHOLD = 1.8     
RERANKER_THRESHOLD = 0.3       
RETRIEVAL_K = 10             
RERANKER_TOP_N = 5            
NUM_OF_RETRIEVED_CHUNKS = 6    
STORAGE_PROVIDER='minio'