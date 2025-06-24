"""
Theory Agent - Specialized in mathematical derivations and conceptual explanations
"""
from typing import List
from .base_agent import BaseAgent

class TheoryAgent(BaseAgent):
    """Agent specialized in theoretical explanations and mathematical concepts"""
    
    def __init__(self):
        super().__init__(
            name="Theory Agent",
            role="Mathematical and theoretical concepts specialist",
            model="gpt-3.5-turbo"
        )
        
        self.theory_keywords = [
            "theory", "mathematical", "derivation", "proof", "equation",
            "formula", "theorem", "algorithm", "optimization", "loss function",
            "gradient", "backpropagation", "calculus", "linear algebra",
            "probability", "statistics", "convergence", "complexity"
        ]
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for theory agent"""
        return """You are a Theory Agent specialized in explaining mathematical concepts and theoretical foundations of Deep Learning and Machine Learning.

Your responsibilities:
1. Explain mathematical concepts clearly and step-by-step
2. Provide mathematical derivations and proofs
3. Break down complex algorithms into understandable steps
4. Explain optimization techniques and loss functions
5. Clarify theoretical foundations and assumptions
6. Connect mathematical concepts to practical applications

Guidelines:
- Use clear mathematical notation
- Provide step-by-step explanations
- Explain intuition behind mathematical concepts
- Give concrete examples when possible
- Clarify assumptions and limitations
- Connect theory to practical implementations
- Use analogies to make concepts accessible

Focus on making theoretical concepts accessible while maintaining mathematical rigor."""
    
    def process_query(self, query: str, context: str = "") -> str:
        """Process query focused on theoretical aspects"""
        messages = self.create_prompt(query, context)
        
        # Add theory-specific instruction
        if context:
            messages[-1]["content"] += "\n\nPlease focus on the theoretical and mathematical aspects, provide derivations where appropriate, and explain the underlying concepts clearly."
        
        return self._call_llm(messages, max_tokens=1500)
    
    def should_handle_query(self, query: str) -> bool:
        """Determine if query is theory-oriented"""
        query_lower = query.lower()
        
        # Check for theory-related keywords
        theory_score = sum(1 for keyword in self.theory_keywords if keyword in query_lower)
        
        # Check for specific theory patterns
        theory_patterns = [
            "how does", "why does", "mathematical", "derive", "proof",
            "equation", "formula", "algorithm", "optimization", "loss",
            "gradient", "backprop", "theoretical", "math behind",
            "explain the theory", "mathematical foundation"
        ]
        
        pattern_score = sum(1 for pattern in theory_patterns if pattern in query_lower)
        
        # Check for question words that often indicate theoretical queries
        question_indicators = ["how", "why", "what is", "explain"]
        question_score = sum(1 for indicator in question_indicators if indicator in query_lower)
        
        return theory_score >= 2 or pattern_score >= 1 or (question_score >= 1 and theory_score >= 1)
    
    def get_capabilities(self) -> List[str]:
        """Get theory agent capabilities"""
        return [
            "mathematical_derivation",
            "concept_explanation",
            "algorithm_breakdown",
            "optimization_theory",
            "theoretical_foundations",
            "mathematical_proofs"
        ] 