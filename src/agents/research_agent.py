"""
Research Agent - Specialized in literature retrieval and academic synthesis
"""
from typing import List
from .base_agent import BaseAgent

class ResearchAgent(BaseAgent):
    """Agent specialized in research paper analysis and academic synthesis"""
    
    def __init__(self):
        super().__init__(
            name="Research Agent",
            role="Literature retrieval and academic synthesis specialist",
            model="gpt-3.5-turbo"
        )
        
        self.research_keywords = [
            "paper", "research", "study", "literature", "citation", "author",
            "publication", "journal", "conference", "arxiv", "theory",
            "methodology", "experiment", "results", "findings", "analysis"
        ]
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for research agent"""
        return """You are a Research Agent specialized in analyzing academic literature and research papers in Deep Learning and Machine Learning.

Your responsibilities:
1. Analyze and synthesize information from research papers
2. Provide accurate citations and references
3. Explain research methodologies and experimental setups
4. Compare different approaches and findings
5. Highlight key contributions and limitations
6. Maintain academic rigor and accuracy

Guidelines:
- Always cite sources when mentioning specific research
- Explain technical concepts clearly
- Provide context about the research field
- Mention related work and comparisons
- Be precise about claims and limitations
- Use academic language appropriately

Focus on providing comprehensive, well-researched answers based on the academic literature."""
    
    def process_query(self, query: str, context: str = "") -> str:
        """Process query focused on research aspects"""
        messages = self.create_prompt(query, context)
        
        # Add research-specific instruction
        if context:
            messages[-1]["content"] += "\n\nPlease focus on the research aspects, cite relevant papers from the context, and provide academic insights."
        
        return self._call_llm(messages, max_tokens=1200)
    
    def should_handle_query(self, query: str) -> bool:
        """Determine if query is research-oriented"""
        query_lower = query.lower()
        
        # Check for research-related keywords
        research_score = sum(1 for keyword in self.research_keywords if keyword in query_lower)
        
        # Check for specific research patterns
        research_patterns = [
            "what paper", "which study", "research on", "literature review",
            "citations", "authors", "methodology", "experiment", "findings",
            "state of the art", "sota", "recent work", "survey"
        ]
        
        pattern_score = sum(1 for pattern in research_patterns if pattern in query_lower)
        
        return research_score >= 2 or pattern_score >= 1
    
    def get_capabilities(self) -> List[str]:
        """Get research agent capabilities"""
        return [
            "literature_search",
            "citation_analysis", 
            "research_synthesis",
            "methodology_explanation",
            "comparative_analysis",
            "academic_writing"
        ] 