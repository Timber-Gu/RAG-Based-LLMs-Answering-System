"""
Implementation Agent - Specialized in code generation and practical implementation
"""
from typing import List
from .base_agent import BaseAgent

class ImplementationAgent(BaseAgent):
    """Agent specialized in code generation and implementation guidance"""
    
    def __init__(self):
        super().__init__(
            name="Implementation Agent", 
            role="Code generation and implementation specialist",
            model="gpt-3.5-turbo"
        )
        
        self.implementation_keywords = [
            "code", "implementation", "python", "pytorch", "tensorflow",
            "programming", "function", "class", "method", "library",
            "framework", "example", "tutorial", "practice", "build",
            "create", "develop", "script", "notebook", "api"
        ]
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for implementation agent"""
        return """You are an Implementation Agent specialized in code generation and practical implementation guidance for Deep Learning and Machine Learning projects.

Your responsibilities:
1. Generate clean, working code examples
2. Provide step-by-step implementation guides
3. Explain code structure and best practices
4. Suggest appropriate libraries and frameworks
5. Debug and optimize code
6. Provide practical implementation tips

Guidelines:
- Write clean, readable, and well-commented code
- Use popular frameworks (PyTorch, TensorFlow, scikit-learn)
- Provide complete, runnable examples
- Explain code logic and design choices
- Follow Python best practices
- Include error handling where appropriate
- Suggest optimizations and alternatives
- Provide installation and setup instructions

Focus on practical, working solutions that can be immediately used."""
    
    def process_query(self, query: str, context: str = "") -> str:
        """Process query focused on implementation aspects"""
        messages = self.create_prompt(query, context)
        
        # Add implementation-specific instruction
        if context:
            messages[-1]["content"] += "\n\nPlease focus on the implementation aspects, provide working code examples, and give practical guidance for building this in Python."
        
        return self._call_llm(messages, max_tokens=1500)
    
    def should_handle_query(self, query: str) -> bool:
        """Determine if query is implementation-oriented"""
        query_lower = query.lower()
        
        # Check for implementation-related keywords
        impl_score = sum(1 for keyword in self.implementation_keywords if keyword in query_lower)
        
        # Check for specific implementation patterns
        impl_patterns = [
            "how to implement", "how to code", "how to build", "how to create",
            "show me code", "write code", "python code", "example code",
            "implement in", "build in", "create in", "code for", "programming",
            "tutorial", "step by step", "hands on", "practical"
        ]
        
        pattern_score = sum(1 for pattern in impl_patterns if pattern in query_lower)
        
        return impl_score >= 2 or pattern_score >= 1
    
    def get_capabilities(self) -> List[str]:
        """Get implementation agent capabilities"""
        return [
            "code_generation",
            "implementation_guidance", 
            "debugging_help",
            "framework_usage",
            "best_practices",
            "optimization_tips"
        ] 