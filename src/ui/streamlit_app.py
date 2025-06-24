"""
Streamlit UI for ML Q&A Assistant
"""
import streamlit as st
import requests
import json
from typing import Dict, Any

# Configure page
st.set_page_config(
    page_title="ML Q&A Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"

def call_api(endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Call the API"""
    try:
        if data:
            response = requests.post(f"{API_BASE_URL}/{endpoint}", json=data)
        else:
            response = requests.get(f"{API_BASE_URL}/{endpoint}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {}

def main():
    """Main Streamlit app"""
    
    # Title and description
    st.title("🤖 ML Q&A Assistant")
    st.markdown("**Advanced Deep Learning Q&A with Multi-Agent System and RAG**")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # RAG toggle
        use_rag = st.checkbox("Use RAG (Retrieval-Augmented Generation)", value=True)
        
        # System status
        st.header("System Status")
        if st.button("Check Health"):
            health = call_api("health")
            if health:
                status = health.get("status", "unknown")
                if status == "healthy":
                    st.success("System is healthy ✅")
                else:
                    st.warning("System is degraded ⚠️")
                
                # Show component status
                components = health.get("components", {})
                for component, status in components.items():
                    if status:
                        st.text(f"✅ {component}")
                    else:
                        st.text(f"❌ {component}")
        
        # Agents info
        st.header("Available Agents")
        agents = call_api("agents")
        if agents:
            for agent_name, agent_info in agents.items():
                with st.expander(f"{agent_info['name']}"):
                    st.write(f"**Role:** {agent_info['role']}")
                    st.write("**Capabilities:**")
                    for capability in agent_info['capabilities']:
                        st.write(f"- {capability}")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        query = st.text_area(
            "Ask a question about Deep Learning, Machine Learning, or Neural Networks:",
            height=100,
            placeholder="e.g., What are convolutional neural networks? How do I implement a CNN in PyTorch?"
        )
        
        # Submit button
        if st.button("Submit Query", type="primary"):
            if query.strip():
                with st.spinner("Processing your query..."):
                    result = call_api("query", {"query": query, "use_rag": use_rag})
                    
                    if result:
                        # Display results
                        st.header("Response")
                        
                        # Show which agents were used
                        agents_used = result.get("agents_used", [])
                        if agents_used:
                            st.info(f"**Agents used:** {', '.join(agents_used)}")
                        
                        # Main response
                        response = result.get("final_response", "No response generated")
                        st.markdown(response)
                        
                        # Context (if available)
                        context = result.get("context", "")
                        if context and use_rag:
                            with st.expander("Retrieved Context"):
                                st.text(context)
            else:
                st.warning("Please enter a query.")
    
    with col2:
        # Example queries
        st.header("Example Queries")
        
        example_queries = [
            "What are convolutional neural networks?",
            "How do I implement a CNN in PyTorch?",
            "Explain the attention mechanism",
            "What is backpropagation?",
            "Show me a GAN implementation",
            "What are the latest papers on transformers?"
        ]
        
        for example in example_queries:
            if st.button(example, key=f"example_{hash(example)}"):
                st.experimental_set_query_params(query=example)
                st.experimental_rerun()

if __name__ == "__main__":
    main() 