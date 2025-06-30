"""
Main entry point for LangChain ML Q&A Assistant
Simple interface to test the multi-agent system
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append('src')

from src.agents.langchain_agents import create_langchain_ml_agents

def main():
    """Main function to test LangChain agents"""
    
    print("🌟 LangChain ML Q&A Assistant")
    print("=" * 50)
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_key_here")
        return
    
    # Initialize agents
    try:
        print("🚀 Initializing LangChain agents...")
        agents = create_langchain_ml_agents()
        print("✅ Agents initialized successfully!")
        print(f"📋 Available agents: {', '.join(agents.get_available_agents())}")
    except Exception as e:
        print(f"❌ Error initializing agents: {e}")
        return
    
    # Health check
    health = agents.health_check()
    print(f"🔍 Health check: {health}")
    
    # Interactive loop
    print("\n💬 Interactive Q&A Mode")
    print("Type 'quit' to exit, 'agents' to see available agents")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n🤔 Your question: ").strip()
            
            if query.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'agents':
                agents_info = agents.get_available_agents()
                print(f"📋 Available agents: {', '.join(agents_info)}")
                continue
            
            if not query:
                print("⚠️ Please enter a question")
                continue
            
            print(f"🔄 Processing query...")
            result = agents.process_query(query)
            
            if result.get('success'):
                print(f"🤖 Agent used: {result['agent_used']}")
                print(f"💡 Response:\n{result['response']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 