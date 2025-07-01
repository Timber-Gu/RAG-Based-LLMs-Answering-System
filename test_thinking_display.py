#!/usr/bin/env python3
"""
Test script for thinking process display functionality
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add src to path
sys.path.append('src')

from src.agents.langchain_agents import LangChainMLAgents

def test_thinking_display():
    """Test the thinking process display functionality"""
    print("🧪 Testing Agent Thinking Process Display")
    print("=" * 60)
    
    try:
        # Initialize agents
        agents = LangChainMLAgents()
        
        # Test query that should trigger multiple tools
        test_query = "Explain the mathematical foundation of gradient descent"
        
        print(f"📝 Query: {test_query}")
        print("-" * 60)
        
        # Process with thinking display enabled
        print("\n🧠 WITH thinking process display:")
        print("=" * 40)
        result_with_thinking = agents.process_query(test_query, show_thinking=True)
        
        if result_with_thinking.get('success'):
            print(f"✅ Agent used: {result_with_thinking['agent_used']}")
            print(f"🧠 Has thinking process: {result_with_thinking.get('has_thinking')}")
            
            if result_with_thinking.get('thinking_process'):
                print("\n🧠 Thinking Steps:")
                for step in result_with_thinking['thinking_process']:
                    print(f"  Step {step['step_number']}: {step['description']}")
                    print(f"    Tool: {step['tool_name']}")
                    print(f"    Input: {step['tool_input']}")
                    print(f"    Result: {step['result_summary']}")
                    print()
            
            print(f"💡 Response: {result_with_thinking['response'][:200]}...")
        else:
            print(f"❌ Error: {result_with_thinking.get('error')}")
        
        print("\n" + "=" * 60)
        
        # Process without thinking display
        print("\n🤐 WITHOUT thinking process display:")
        print("=" * 40)
        result_without_thinking = agents.process_query(test_query, show_thinking=False)
        
        if result_without_thinking.get('success'):
            print(f"✅ Agent used: {result_without_thinking['agent_used']}")
            print(f"🧠 Has thinking process: {result_without_thinking.get('has_thinking')}")
            print(f"💡 Response: {result_without_thinking['response'][:200]}...")
        else:
            print(f"❌ Error: {result_without_thinking.get('error')}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_thinking_display() 