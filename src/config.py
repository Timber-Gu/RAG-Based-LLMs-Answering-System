"""
Configuration settings for the ML Q&A Assistant
"""
import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    """Application settings"""
    
    # API Keys (from environment variables only)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
    
    # Vector Database Settings
    PINECONE_INDEX_NAME: str = "ml-qa-papers"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Data Settings
    MAX_PAPERS: int = 100
    DATA_DIR: str = "data"
    PAPERS_DIR: str = "data/papers"
    KNOWLEDGE_BASE_FILE: str = "data/knowledge_base.json"
    
    # Agent Settings
    MAX_AGENTS: int = 3  # Start simple in Phase 1
    AGENT_TIMEOUT: int = 30
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings() 