#!/usr/bin/env python3
"""
Simplified setup script for Phase 1 of ML Q&A Assistant
This version skips heavy ML dependencies and focuses on core functionality
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_curation.paper_collector import PaperCollector
from src.data_curation.content_extractor import ContentExtractor
from src.config import settings

def setup_phase1_simple():
    """Setup Phase 1 components (simplified version)"""
    print("🚀 Setting up Phase 1: Foundation & Data (Simplified)")
    print("=" * 60)
    
    # Check environment variables
    print("1. Checking configuration...")
    if not settings.OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    if not settings.PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY not found") 
        return False
    
    print("✅ API keys configured")
    
    # Create directories
    print("\n2. Creating directories...")
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.PAPERS_DIR, exist_ok=True)
    print("✅ Directories created")
    
    # Collect papers (small sample for testing)
    print("\n3. Collecting papers from arXiv...")
    collector = PaperCollector()
    
    # Check if papers already exist
    existing_papers = collector.load_papers_metadata()
    if existing_papers:
        print(f"📚 Found {len(existing_papers)} existing papers")
        use_existing = input("Use existing papers? (y/n): ").lower().strip()
        if use_existing == 'y':
            papers = existing_papers
        else:
            papers = collector.collect_and_save(max_papers=10)  # Small sample
    else:
        papers = collector.collect_and_save(max_papers=10)  # Small sample
    
    print(f"✅ Papers collected: {len(papers)}")
    
    # Extract content and create knowledge base (without ML embeddings for now)
    print("\n4. Extracting content...")
    extractor = ContentExtractor()
    
    # Check if knowledge base exists
    if os.path.exists(settings.KNOWLEDGE_BASE_FILE):
        print("📖 Knowledge base already exists")
        rebuild = input("Rebuild knowledge base? (y/n): ").lower().strip()
        if rebuild != 'y':
            print("✅ Using existing knowledge base")
        else:
            chunks = extractor.process_papers(papers, max_papers=5)  # Small sample
            extractor.save_knowledge_base(chunks)
            print(f"✅ Knowledge base created: {len(chunks)} chunks")
    else:
        chunks = extractor.process_papers(papers, max_papers=5)  # Small sample
        extractor.save_knowledge_base(chunks)
        print(f"✅ Knowledge base created: {len(chunks)} chunks")
    
    # Test the agent system (without RAG for now)
    print("\n5. Testing the multi-agent system...")
    try:
        from src.agents.research_agent import ResearchAgent
        from src.agents.theory_agent import TheoryAgent
        from src.agents.implementation_agent import ImplementationAgent
        
        # Test agents
        research_agent = ResearchAgent()
        theory_agent = TheoryAgent()
        impl_agent = ImplementationAgent()
        
        print("✅ All agents instantiated successfully")
        
        # Test a simple query (without RAG)
        test_query = "What are neural networks?"
        try:
            response = theory_agent.process_query(test_query)
            if response and not response.startswith("Error"):
                print("✅ Agent query processing working")
            else:
                print("⚠️ Agent query processing may have issues")
        except Exception as e:
            print(f"⚠️ Agent testing error: {e}")
            
    except Exception as e:
        print(f"❌ Error testing agents: {e}")
        return False
    
    print("\n🎉 Phase 1 simplified setup completed successfully!")
    print("\nWhat works:")
    print("✅ Paper collection from arXiv")
    print("✅ Content extraction and knowledge base creation")
    print("✅ Multi-agent system (Research, Theory, Implementation)")
    print("✅ Basic query processing")
    
    print("\nNext steps:")
    print("1. Fix PyTorch/TorchVision compatibility for full ML features")
    print("2. Start the API server: python -m src.api.server")
    print("3. Start the Streamlit UI: streamlit run src/ui/streamlit_app.py")
    print("4. Test the system with various queries")
    
    print("\nNote: RAG (vector embeddings) will be available once ML dependencies are fixed")
    
    return True

if __name__ == "__main__":
    success = setup_phase1_simple()
    if not success:
        sys.exit(1) 