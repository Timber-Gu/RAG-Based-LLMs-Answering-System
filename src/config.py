"""
Configuration settings for LangChain Multi-Agent ML Q&A Assistant
"""
import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

# Debug: Check if environment variables are loaded correctly
import os
theory_model = os.getenv('THEORY_MODEL', 'NOT_SET')
if theory_model != 'NOT_SET':
    print(f"🔧 Environment loaded: THEORY_MODEL = {theory_model}")
else:
    print(f"🔧 Warning: THEORY_MODEL not found in environment")

class Settings(BaseModel):
    """Application settings for LangChain multi-agent system"""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    
    # Multi-Model Agent Settings
    RESEARCH_MODEL: str = os.getenv("RESEARCH_MODEL", "llama3.1") # Ollama model for research agent
    THEORY_MODEL: str = os.getenv("THEORY_MODEL", "gpt-4")  # Theory agent model (gpt-4, gpt-4o, gpt-4-turbo)
    IMPLEMENTATION_MODEL: str = "claude-3-5-sonnet-20241022"  # Claude for implementation agent
    AGENT_TEMPERATURE: float = 0.7
    
    # Token limits (configured per agent in langchain_agents.py):
    # - Routing LLM (GPT-4o-mini): 512 tokens (classification only)
    # - Research Agent (Claude): 4096 tokens (research responses)
    # - Theory Agent (GPT-4): 6144 tokens (mathematical explanations + CoT)
    # - Implementation Agent (Claude): 8192 tokens (complete code implementations)
    AGENT_MAX_TOKENS: int = 8192  # Default/maximum for implementation tasks
    
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
    
    # Ensemble Retrieval Settings
    USE_ENSEMBLE_RETRIEVAL: bool = os.getenv("USE_ENSEMBLE_RETRIEVAL", "true").lower() == "true"
    ENSEMBLE_USE_COMPRESSION: bool = os.getenv("ENSEMBLE_USE_COMPRESSION", "true").lower() == "true"  # Re-enabled with fixes
    ENSEMBLE_MAX_DOCS: int = int(os.getenv("ENSEMBLE_MAX_DOCS", "4"))  # Increased back to 4 for better coverage
    ENSEMBLE_SIMILARITY_WEIGHT: float = float(os.getenv("ENSEMBLE_SIMILARITY_WEIGHT", "0.4"))
    ENSEMBLE_SELF_QUERY_WEIGHT: float = float(os.getenv("ENSEMBLE_SELF_QUERY_WEIGHT", "0.3"))
    ENSEMBLE_PARENT_DOC_WEIGHT: float = float(os.getenv("ENSEMBLE_PARENT_DOC_WEIGHT", "0.3"))
    ENSEMBLE_COMPRESSION_MAX_CHARS: int = int(os.getenv("ENSEMBLE_COMPRESSION_MAX_CHARS", "1500"))  # Increased for better content preservation
    
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