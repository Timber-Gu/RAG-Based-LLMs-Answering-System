"""
Base agent class for the multi-agent system
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import openai
from ..config import settings

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, role: str, model: str = "gpt-3.5-turbo"):
        self.name = name
        self.role = role
        self.model = model
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass
    
    @abstractmethod
    def process_query(self, query: str, context: str = "") -> str:
        """Process a query and return response"""
        pass
    
    def _call_llm(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        """Call OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM for {self.name}: {e}")
            return f"Error: Unable to process query - {str(e)}"
    
    def create_prompt(self, query: str, context: str = "") -> List[Dict[str, str]]:
        """Create prompt messages for the agent"""
        messages = [
            {"role": "system", "content": self.get_system_prompt()}
        ]
        
        if context:
            user_content = f"Context:\n{context}\n\nQuery: {query}"
        else:
            user_content = query
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def should_handle_query(self, query: str) -> bool:
        """Determine if this agent should handle the query"""
        # Default implementation - can be overridden by specific agents
        return True
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities"""
        return ["general_qa"]
    
    def __str__(self):
        return f"{self.name} ({self.role})"
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>" 