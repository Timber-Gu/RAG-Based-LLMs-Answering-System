#!/usr/bin/env python3
"""
Test script for Chain of Thoughts functionality in Theory Agent
"""

import os
import sys
sys.path.append('src')

from agents.langchain_agents import LangChainMLAgents

def test_theory_agent_cot():
    """Test Theory Agent with Chain of Thoughts"""
    print("🧪 Testing Theory Agent with Chain of Thoughts...")
    print("=" * 60)
    
    try:
        # Initialize the agents
        agents = LangChainMLAgents()
        
        # Test query for mathematical explanation
        test_query = "Explain the mathematical foundation of backpropagation in neural networks"
        
        print(f"📝 Query: {test_query}")
        print("-" * 60)
        
        # Process the query
        result = agents.process_query(test_query)
        
        if result.get('success'):
            print(f"✅ Agent used: {result['agent_used']}")
            print(f"📖 Response:\n{result['response']}")
        else:
            print(f"❌ Error: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_cot_tool_directly():
    """Test the CoT tool directly"""
    print("\n🔧 Testing CoT Tool Directly...")
    print("=" * 60)
    
    try:
        agents = LangChainMLAgents()
        cot_tool = agents._create_cot_tool()
        
        test_problem = "How does gradient descent optimize neural network weights?"
        result = cot_tool.func(test_problem)
        
        print(f"📝 Problem: {test_problem}")
        print("-" * 60)
        print(f"🧠 CoT Analysis:\n{result}")
        
    except Exception as e:
        print(f"❌ CoT tool test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test CoT tool directly first
    test_cot_tool_directly()
    
    # Then test full Theory Agent
    test_theory_agent_cot() 