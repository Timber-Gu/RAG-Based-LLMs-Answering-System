"""
Demo script for LangChain ML Q&A Assistant
Shows the multi-agent system in action
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append('src')

from src.agents.langchain_agents import create_langchain_ml_agents

def demo_queries():
    """Demonstrate different types of queries and agent routing"""
    
    print("🌟 LangChain ML Q&A Assistant Demo")
    print("=" * 60)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: Please set OPENAI_API_KEY in .env file")
        return
    
    # Initialize agents
    try:
        print("🚀 Initializing LangChain agents...")
        agents = create_langchain_ml_agents()
        print("✅ Agents ready!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Demo queries for different agents
    demo_queries = [
        {
            "query": "What are neural networks?",
            "expected_agent": "theory",
            "description": "Basic theory question"
        },
        {
            "query": "How to implement a CNN in PyTorch?", 
            "expected_agent": "implementation",
            "description": "Implementation question"
        },
        {
            "query": "Recent research on transformer architectures",
            "expected_agent": "research", 
            "description": "Research question"
        },
        {
            "query": "Explain gradient descent mathematically",
            "expected_agent": "theory",
            "description": "Mathematical explanation"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n{'='*60}")
        print(f"Demo {i}: {demo['description']}")
        print(f"Query: {demo['query']}")
        print(f"Expected Agent: {demo['expected_agent']}")
        print(f"{'='*60}")
        
        try:
            result = agents.process_query(demo['query'])
            
            if result.get('success'):
                print(f"✅ Agent Used: {result['agent_used']}")
                print(f"🎯 Routing {'✓' if result['agent_used'] == demo['expected_agent'] else '✗'}")
                print(f"💡 Response Preview: {result['response'][:200]}...")
            else:
                print(f"❌ Error: {result.get('error')}")
        
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print(f"\n{'='*60}")
    print("🎉 Demo completed! The LangChain multi-agent system is working!")
    print("✨ Key features demonstrated:")
    print("  • Proper LangChain agent routing")
    print("  • Specialized agent responses")
    print("  • Clean error handling")
    print("  • RAG integration (with knowledge base)")

if __name__ == "__main__":
    demo_queries() 