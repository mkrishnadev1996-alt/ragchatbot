import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration settings for the RAG system"""

    # LLM settings (any OpenAI-compatible endpoint: Groq, Google Gemini, etc.)
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    # Input guard (decision model on OpenRouter). Disabled when GUARD_API_KEY is empty.
    GUARD_API_KEY: str = os.getenv("GUARD_API_KEY", "")
    GUARD_URL: str = os.getenv("GUARD_URL", "https://openrouter.ai/api/alpha/decisions")
    GUARD_MODEL: str = os.getenv("GUARD_MODEL", "respan/span-01-lite:free")
    GUARD_THRESHOLD: float = float(os.getenv("GUARD_THRESHOLD", "0.5"))  # block when P >= this

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800  # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100  # Characters to overlap between chunks
    MAX_RESULTS: int = 5  # Maximum search results to return
    MAX_HISTORY: int = 2  # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR
    LOG_FILE: str | None = os.getenv("LOG_FILE") or None  # Optional rotating log file path


config = Config()
