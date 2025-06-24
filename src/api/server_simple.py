"""
Simplified FastAPI server for ML Q&A Assistant (without RAG)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

from ..agents.orchestrator_simple import SimpleAgentOrchestrator
from ..config import settings

app = FastAPI(
    title="ML Q&A Assistant (Simplified)",
    description="Advanced ML Q&A Assistant with Multi-Agent System (RAG disabled)",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize simple orchestrator
orchestrator = SimpleAgentOrchestrator()

class QueryRequest(BaseModel):
    query: str
    use_rag: bool = False  # RAG disabled in simplified mode

class QueryResponse(BaseModel):
    query: str
    agents_used: list
    final_response: str
    context: str = ""

@app.get("/")
async def root():
    return {
        "message": "ML Q&A Assistant API (Simplified Mode)", 
        "version": "0.1.0",
        "note": "RAG pipeline disabled due to ML dependency issues"
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using the multi-agent system (without RAG)"""
    try:
        result = orchestrator.process_query(request.query)
        return QueryResponse(
            query=result['query'],
            agents_used=result['agents_used'],
            final_response=result['final_response'],
            context=result['context']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def get_agents():
    """Get information about available agents"""
    try:
        return orchestrator.get_agent_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check system health"""
    try:
        status = orchestrator.health_check()
        return {
            "status": "partial" if any(status.values()) else "degraded", 
            "components": status,
            "note": "RAG pipeline disabled in simplified mode"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "src.api.server_simple:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    ) 