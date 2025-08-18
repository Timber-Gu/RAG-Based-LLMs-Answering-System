"""
Clean FastAPI server using LangChain multi-agent system
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.langchain_agents import create_langchain_ml_agents
from config import settings

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = None

class QueryResponse(BaseModel):
    query: str
    agent_used: str
    response: str
    success: bool
    error: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(
    title="LangChain ML Q&A Assistant",
    description="Multi-agent system for Machine Learning Q&A using LangChain",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agents instance
agents = None

@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    global agents
    try:
        print("🚀 Initializing LangChain ML Agents...")
        agents = create_langchain_ml_agents()
        print("✅ LangChain ML Agents ready!")
    except Exception as e:
        print(f"❌ Failed to initialize agents: {e}")
        agents = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LangChain ML Q&A Assistant API", 
        "version": "1.0.0",
        "endpoints": {
            "query": "/query",
            "health": "/health",
            "agents": "/agents",
            "docs": "/docs"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using appropriate LangChain agent"""
    if not agents:
        raise HTTPException(status_code=503, detail="Agents not initialized")
    
    try:
        result = agents.process_query(
            query=request.query,
            chat_history=request.chat_history or []
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/agents")
async def get_agents():
    """Get available agents"""
    if not agents:
        return {"agents": [], "error": "Agents not initialized"}
    
    available_agents = agents.get_available_agents()
    return {
        "agents": available_agents,
        "count": len(available_agents),
        "descriptions": {
            "research": "Finds and synthesizes information from ML/DL literature",
            "theory": "Explains mathematical concepts and theoretical foundations",
            "implementation": "Provides code examples and practical guidance"
        }
    }

@app.get("/health")
async def health_check():
    """System health check"""
    if not agents:
        return {"status": "unhealthy", "error": "Agents not initialized"}
    
    try:
        health_status = agents.health_check()
        overall_healthy = all(health_status.values())
        
        return {
            "status": "healthy" if overall_healthy else "partial",
            "components": health_status,
            "openai_configured": bool(settings.OPENAI_API_KEY)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/test")
async def test_endpoint():
    """Test endpoint with sample queries"""
    if not agents:
        return {"error": "Agents not initialized"}
    
    test_queries = [
        "What are convolutional neural networks?",
        "How to implement a neural network in PyTorch?",
        "Recent research on transformer architectures"
    ]
    
    results = []
    for query in test_queries:
        try:
            result = agents.process_query(query)
            results.append({
                "query": query,
                "agent_used": result.get('agent_used', 'unknown'),
                "success": result.get('success', False),
                "response_length": len(result.get('response', ''))
            })
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })
    
    return {"test_results": results}

if __name__ == "__main__":
    print("🌟 Starting LangChain ML Q&A Assistant API")
    print(f"📝 API will be available at: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"📚 Interactive docs at: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    
    uvicorn.run(
        "langchain_server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    ) 