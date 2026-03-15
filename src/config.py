from dotenv import load_dotenv
import os

load_dotenv()

# ── Gemini (embeddings only) ───────────────────────────────────
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
GEMINI_EMBED_MODEL: str = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-2-preview")

# ── Groq (generation) ──────────────────────────────────────────
GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
GROQ_CHAT_MODEL: str = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

# ── Pinecone ───────────────────────────────────────────────────
PINECONE_API_KEY: str = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "code-explainer")
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_DIMENSION: int = 3072  # gemini-embedding-2-preview output dim

# ── Retrieval ──────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.60"))

# ── Chunking ───────────────────────────────────────────────────
CHUNK_TOKEN_LIMIT: int = int(os.getenv("CHUNK_TOKEN_LIMIT", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# ── Generation ─────────────────────────────────────────────────
MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))

# ── Jina Reranker ──────────────────────────────────────────────
JINA_URL: str = os.getenv("JINA_URL", "https://api.jina.ai/v1/rerank")
JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")
JINA_RERANKER_MODEL: str = os.getenv("JINA_RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
JINA_RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "5"))
