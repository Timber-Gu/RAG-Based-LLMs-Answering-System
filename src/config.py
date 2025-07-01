"""
Configuration settings for LangChain Multi-Agent ML Q&A Assistant
"""
import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings(BaseModel):
    """Application settings for LangChain multi-agent system"""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    
    # Multi-Model Agent Settings
    RESEARCH_MODEL: str = os.getenv("RESEARCH_MODEL", "llama3.1") # Ollama model for research agent
    THEORY_MODEL: str = "gpt-4"                       # GPT-4 for theory agent  
    IMPLEMENTATION_MODEL: str = "claude-3-5-sonnet-20241022"  # Claude for implementation agent
    AGENT_TEMPERATURE: float = 0.7
    AGENT_MAX_TOKENS: int = 1000
    
    # Ollama API Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Ollama API endpoint
    OLLAMA_API_KEY: Optional[str] = os.getenv("OLLAMA_API_KEY")  # Optional for hosted Ollama services
    
    # Vector Store Settings
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "pinecone") 
    
    # Pinecone Settings (Cloud)
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT")
    
    # Common Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # LLM-based Chunking Settings
    USE_LLM_CHUNKING: bool = os.getenv("USE_LLM_CHUNKING", "true").lower() == "true"
    LLM_CHUNKING_MODEL: str = os.getenv("LLM_CHUNKING_MODEL", "gpt-3.5-turbo")
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "1500"))
    
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