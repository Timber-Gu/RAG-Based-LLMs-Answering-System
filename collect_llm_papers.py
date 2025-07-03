#!/usr/bin/env python3
"""
CLI tool to collect LLM papers and update the knowledge base
"""
import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_curation.llm_paper_collector import update_knowledge_base
from src.config import settings

def main():
    """Main function for LLM paper collector CLI"""
    parser = argparse.ArgumentParser(description="Collect LLM papers and update knowledge base")
    parser.add_argument(
        "--max-papers", 
        type=int, 
        default=settings.MAX_PAPERS,
        help=f"Maximum number of papers to collect (default: {settings.MAX_PAPERS})"
    )
    parser.add_argument(
        "--upload", 
        action="store_true",
        help="Upload to Pinecone after collecting (default: False)"
    )
    
    args = parser.parse_args()
    
    # Override settings
    settings.MAX_PAPERS = args.max_papers
    
    print("📚 LLM Knowledge Base Updater")
    print("=" * 50)
    print(f"Max papers: {settings.MAX_PAPERS}")
    
    # Check OpenAI API key (needed for LangChain agent initialization if --upload)
    if args.upload and not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key for uploading to Pinecone")
        return
    
    # Check Pinecone API key if uploading
    if args.upload and not os.getenv('PINECONE_API_KEY'):
        print("❌ Error: PINECONE_API_KEY not found in environment variables")
        print("Please create a .env file with your Pinecone API key")
        return
    
    try:
        # Update knowledge base
        print("🔄 Collecting LLM papers...")
        update_knowledge_base()
        print("✅ Knowledge base updated successfully!")
        
        # Upload to Pinecone if requested
        if args.upload:
            print("\n🔄 Uploading knowledge base to Pinecone...")
            try:
                from src.agents.langchain_agents import LangChainMLAgents
                agents = LangChainMLAgents()
                success = agents.upsert_knowledge_base_to_pinecone()
                if success:
                    print("✅ Knowledge base uploaded to Pinecone!")
                else:
                    print("❌ Failed to upload to Pinecone. Check your configuration.")
            except Exception as e:
                print(f"❌ Error during Pinecone upload: {e}")
            
        print("\n✨ Done! Your knowledge base is now ready to use.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 