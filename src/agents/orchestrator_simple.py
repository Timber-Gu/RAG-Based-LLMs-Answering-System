"""
Simplified Agent Orchestrator - Coordinates multi-agent responses without RAG
"""
from typing import List, Dict, Any, Optional
from .research_agent import ResearchAgent
from .theory_agent import TheoryAgent
from .implementation_agent import ImplementationAgent

class SimpleAgentOrchestrator:
    """Orchestrates multiple agents to provide comprehensive responses (without RAG)"""
    
    def __init__(self):
        self.agents = {
            'research': ResearchAgent(),
            'theory': TheoryAgent(),
            'implementation': ImplementationAgent()
        }
    
    def route_query(self, query: str) -> List[str]:
        """Determine which agents should handle the query"""
        selected_agents = []
        
        # Check each agent's suitability
        for agent_name, agent in self.agents.items():
            if agent.should_handle_query(query):
                selected_agents.append(agent_name)
        
        # If no agents are selected, use theory as default
        if not selected_agents:
            selected_agents = ['theory']
        
        # Limit to top 2 agents to avoid overwhelming responses
        return selected_agents[:2]
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query using appropriate agents (without RAG)"""
        result = {
            'query': query,
            'agents_used': [],
            'responses': {},
            'context': 'RAG not available in simplified mode',
            'final_response': ''
        }
        
        # Route query to appropriate agents
        selected_agents = self.route_query(query)
        result['agents_used'] = selected_agents
        
        # Get responses from selected agents
        for agent_name in selected_agents:
            try:
                agent = self.agents[agent_name]
                response = agent.process_query(query, "")  # No context for now
                result['responses'][agent_name] = response
            except Exception as e:
                print(f"Error getting response from {agent_name}: {e}")
                result['responses'][agent_name] = f"Error: {str(e)}"
        
        # Combine responses
        result['final_response'] = self._combine_responses(result['responses'], query)
        
        return result
    
    def _combine_responses(self, responses: Dict[str, str], query: str) -> str:
        """Combine multiple agent responses into a coherent answer"""
        if not responses:
            return "I'm sorry, I couldn't generate a response for your query."
        
        if len(responses) == 1:
            return list(responses.values())[0]
        
        # For multiple responses, create a structured answer
        combined = f"Here's a comprehensive answer to your query: '{query}'\n\n"
        
        agent_labels = {
            'research': 'Research Perspective',
            'theory': 'Theoretical Explanation', 
            'implementation': 'Implementation Guide'
        }
        
        for agent_name, response in responses.items():
            label = agent_labels.get(agent_name, agent_name.title())
            combined += f"## {label}\n\n{response}\n\n"
        
        return combined.strip()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about available agents"""
        info = {}
        for name, agent in self.agents.items():
            info[name] = {
                'name': agent.name,
                'role': agent.role,
                'capabilities': agent.get_capabilities()
            }
        return info
    
    def health_check(self) -> Dict[str, bool]:
        """Check if all components are working"""
        status = {}
        
        # RAG pipeline not available in simplified mode
        status['rag_pipeline'] = False
        
        # Check agents
        for name, agent in self.agents.items():
            try:
                # Simple test
                test_response = agent.process_query("test", "")
                status[f'agent_{name}'] = bool(test_response and not test_response.startswith("Error:"))
            except:
                status[f'agent_{name}'] = False
        
        return status

if __name__ == "__main__":
    # Test the simple orchestrator
    orchestrator = SimpleAgentOrchestrator()
    
    # Test queries
    test_queries = [
        "What are convolutional neural networks?",
        "How do I implement a CNN in PyTorch?",
        "What papers discuss attention mechanisms?",
        "Explain the mathematical derivation of backpropagation"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        result = orchestrator.process_query(query)
        print(f"Agents used: {result['agents_used']}")
        print(f"Response:\n{result['final_response']}") 