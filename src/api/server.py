"""
FastAPI server for ML Q&A Assistant
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import sys
import os

# Add project root to Python path for direct execution
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

try:
    from ..agents.orchestrator import AgentOrchestrator
    from ..config import settings
except ImportError:
    # Fallback for direct execution
    from src.agents.orchestrator import AgentOrchestrator
    from src.config import settings

app = FastAPI(
    title="ML Q&A Assistant",
    description="Advanced ML Q&A Assistant with Multi-Agent System and RAG",
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

# Initialize orchestrator
orchestrator = AgentOrchestrator()

class QueryRequest(BaseModel):
    query: str
    use_rag: bool = True

class QueryResponse(BaseModel):
    query: str
    agents_used: list
    final_response: str
    context: str = ""

@app.get("/")
async def root():
    return {"message": "ML Q&A Assistant API", "version": "0.1.0"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using the multi-agent system"""
    try:
        result = orchestrator.process_query(request.query, request.use_rag)
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
        return {"status": "healthy" if all(status.values()) else "degraded", "components": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    ) 