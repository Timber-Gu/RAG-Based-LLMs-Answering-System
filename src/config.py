"""
Configuration settings for LangChain Multi-Agent ML Q&A Assistant
"""
import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    """Application settings for LangChain multi-agent system"""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # LangChain Agent Settings
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    AGENT_TEMPERATURE: float = 0.7
    AGENT_MAX_TOKENS: int = 1000
    
    # Vector Store Settings (ChromaDB - simpler than Pinecone)
    VECTOR_STORE_PATH: str = "data/vector_store"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Data Settings
    MAX_PAPERS: int = 20  # Start small for testing
    DATA_DIR: str = "data"
    PAPERS_DIR: str = "data/papers"
    KNOWLEDGE_BASE_FILE: str = "data/knowledge_base.json"
    
    # API Settings
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings() 