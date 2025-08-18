"""
Main entry point for LangChain ML Q&A Assistant
Simple interface to test the multi-agent system
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

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
    print("Available commands:")
    print("  • 'quit' - Exit the application")
    print("  • 'agents' - Show available agents")
    print("  • 'thinking on/off' - Toggle thinking process display")
    print("  • 'history' - Show chat history")
    print("  • 'clear history' - Clear chat history")
    print("  • 'save history' - Save chat history to file")
    print("  • 'load history' - Load chat history from file")
    print("-" * 50)
    
    show_thinking = True  # Default to showing thinking process
    
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
            
            if query.lower().startswith('thinking'):
                if 'on' in query.lower():
                    show_thinking = True
                    print("🧠 Thinking process display: ON")
                elif 'off' in query.lower():
                    show_thinking = False
                    print("🧠 Thinking process display: OFF")
                else:
                    print(f"🧠 Thinking process display: {'ON' if show_thinking else 'OFF'}")
                continue
            
            # Handle chat history commands
            if query.lower() == 'history':
                history_summary = agents.get_chat_history_summary()
                print(f"📊 Chat History Summary:")
                print(f"   Total messages: {history_summary['total_messages']}")
                print(f"   Human messages: {history_summary['human_messages']}")
                print(f"   AI messages: {history_summary['ai_messages']}")
                print(f"   Max history length: {history_summary['max_history_length']}")
                
                if history_summary['total_messages'] > 0:
                    print("\n📜 Recent Chat History:")
                    history = agents.get_chat_history()
                    for entry in history[-10:]:  # Show last 10 messages
                        msg_type = "🤔 You" if entry['type'] == 'human' else "🤖 AI"
                        content = entry['content'][:100] + "..." if len(entry['content']) > 100 else entry['content']
                        print(f"   {msg_type}: {content}")
                else:
                    print("   No chat history available")
                continue
            
            if query.lower() == 'clear history':
                agents.clear_chat_history()
                continue
            
            if query.lower() == 'save history':
                success = agents.save_chat_history_to_file()
                if success:
                    print("✅ Chat history saved successfully")
                else:
                    print("❌ Failed to save chat history")
                continue
            
            if query.lower() == 'load history':
                success = agents.load_chat_history_from_file()
                if success:
                    summary = agents.get_chat_history_summary()
                    print(f"📊 Loaded {summary['total_messages']} messages")
                else:
                    print("❌ Failed to load chat history")
                continue
            
            if not query:
                print("⚠️ Please enter a question")
                continue
            
            print(f"🔄 Processing query...")
            result = agents.process_query(query, show_thinking=show_thinking)
            
            if result.get('success'):
                print(f"🤖 Agent used: {result['agent_used']}")
                
                # Display thinking process if available and enabled
                if show_thinking and result.get('has_thinking') and result.get('thinking_process'):
                    print("\n🧠 Agent Thinking Process:")
                    print("=" * 40)
                    for step in result['thinking_process']:
                        print(f"\nStep {step['step_number']}: {step['description']}")
                        if step.get('result_summary'):
                            print(f"   Result: {step['result_summary']}")
                    print("=" * 40)
                
                print(f"\n💡 Final Response:\n{result['response']}")
                
                # Show chat history context info
                history_summary = agents.get_chat_history_summary()
                print(f"\n📊 Chat context: {history_summary['total_messages']} messages in history")
                
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 