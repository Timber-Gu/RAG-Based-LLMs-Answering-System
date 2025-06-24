#!/usr/bin/env python3
"""
Setup script for Phase 1 of ML Q&A Assistant
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_curation.paper_collector import PaperCollector
from src.data_curation.content_extractor import ContentExtractor
from src.rag.vector_store import RAGPipeline
from src.config import settings

def setup_phase1():
    """Setup Phase 1 components"""
    print("🚀 Setting up Phase 1: Foundation & Data")
    print("=" * 50)
    
    # Check environment variables
    print("1. Checking environment variables...")
    if not settings.OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found. Please set it in .env file")
        return False
    
    if not settings.PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY not found. Please set it in .env file")
        return False
    
    print("✅ Environment variables configured")
    
    # Create directories
    print("\n2. Creating directories...")
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.PAPERS_DIR, exist_ok=True)
    print("✅ Directories created")
    
    # Collect papers
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
            papers = collector.collect_and_save(max_papers=20)  # Start with 20 for testing
    else:
        papers = collector.collect_and_save(max_papers=20)  # Start with 20 for testing
    
    print(f"✅ Papers collected: {len(papers)}")
    
    # Extract content and create knowledge base
    print("\n4. Extracting content and creating knowledge base...")
    extractor = ContentExtractor()
    
    # Check if knowledge base exists
    if os.path.exists(settings.KNOWLEDGE_BASE_FILE):
        print("📖 Knowledge base already exists")
        rebuild = input("Rebuild knowledge base? (y/n): ").lower().strip()
        if rebuild != 'y':
            print("✅ Using existing knowledge base")
        else:
            chunks = extractor.process_papers(papers, max_papers=10)  # Process 10 for testing
            extractor.save_knowledge_base(chunks)
            print(f"✅ Knowledge base created: {len(chunks)} chunks")
    else:
        chunks = extractor.process_papers(papers, max_papers=10)  # Process 10 for testing
        extractor.save_knowledge_base(chunks)
        print(f"✅ Knowledge base created: {len(chunks)} chunks")
    
    # Build RAG pipeline
    print("\n5. Building RAG pipeline...")
    try:
        pipeline = RAGPipeline()
        pipeline.build_knowledge_base()
        print("✅ RAG pipeline built successfully")
    except Exception as e:
        print(f"❌ Error building RAG pipeline: {e}")
        return False
    
    # Test the system
    print("\n6. Testing the system...")
    try:
        from src.agents.orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator()
        
        # Test query
        test_query = "What are neural networks?"
        result = orchestrator.process_query(test_query, use_rag=False)  # Test without RAG first
        
        if result['final_response'] and not result['final_response'].startswith("Error"):
            print("✅ Multi-agent system working")
        else:
            print("❌ Multi-agent system error")
            return False
        
        # Test with RAG
        result_rag = orchestrator.process_query(test_query, use_rag=True)
        if result_rag['final_response']:
            print("✅ RAG system working")
        else:
            print("⚠️ RAG system may have issues")
            
    except Exception as e:
        print(f"❌ Error testing system: {e}")
        return False
    
    print("\n🎉 Phase 1 setup completed successfully!")
    print("\nNext steps:")
    print("1. Start the API server: python -m src.api.server")
    print("2. Start the Streamlit UI: streamlit run src/ui/streamlit_app.py")
    print("3. Test the system with various queries")
    
    return True

if __name__ == "__main__":
    success = setup_phase1()
    if not success:
        sys.exit(1) 