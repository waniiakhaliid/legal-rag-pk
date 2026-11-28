import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Pinecone
PINECONE_INDEX_NAME = "legal-pk"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
EMBEDDING_DIMENSION = 3072

# Gemini
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
GEMINI_GENERATION_MODEL = "gemini-3-flash-preview"   # final answer
GEMINI_PREPARE_MODEL    = "gemini-flash-lite-latest" # query rewrite (small JSON task)
GEMINI_FALLBACK_MODEL   = "gemini-3.6-flash"
# chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Retrieval
TOP_K_RETRIEVAL = 10      # how many chunks to retrieve before reranking
TOP_K_RERANK = 3          # how many chunks to keep after reranking

# Paths
DATA_DIR = "data/raw"