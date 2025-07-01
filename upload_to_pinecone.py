#!/usr/bin/env python3
"""
Standalone script to upload knowledge base to Pinecone
Supports both traditional and LLM-chunked content
"""
import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_curation.llm_paper_collector import upload_to_pinecone
from src.config import settings

def main():
    """Main function for Pinecone upload"""
    parser = argparse.ArgumentParser(description="Upload knowledge base to Pinecone")
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force upload even if vectors already exist"
    )
    parser.add_argument(
        "--clear", 
        action="store_true",
        help="Clear existing vectors before uploading"
    )
    
    args = parser.parse_args()
    
    print("🔄 Pinecone Knowledge Base Uploader")
    print("=" * 50)
    
    # Check required environment variables
    if not settings.PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY not found in environment variables")
        print("Please create a .env file with your Pinecone API key")
        return 1
    
    if not settings.PINECONE_INDEX_NAME:
        print("❌ Error: PINECONE_INDEX_NAME not found in environment variables")
        print("Please set your Pinecone index name in the .env file")
        return 1
    
    # Check if knowledge base exists
    if not os.path.exists(settings.KNOWLEDGE_BASE_FILE):
        print(f"❌ Error: Knowledge base file not found: {settings.KNOWLEDGE_BASE_FILE}")
        print("Please run 'python collect_llm_papers.py' first to create the knowledge base")
        return 1
    
    # Clear existing vectors if requested
    if args.clear:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(settings.PINECONE_INDEX_NAME)
            
            print("🗑️ Clearing existing vectors...")
            index.delete(delete_all=True, namespace="__default__")
            print("✅ Existing vectors cleared")
        except Exception as e:
            print(f"⚠️ Warning: Could not clear existing vectors: {e}")
    
    try:
        # Upload to Pinecone using LangChain-compatible approach
        print(f"📤 Uploading knowledge base to Pinecone index: {settings.PINECONE_INDEX_NAME}")
        print(f"📍 Namespace: __default__")
        print("🔄 Using LangChain-compatible upserting method...")
        
        success = upload_to_pinecone()
        
        if success:
            print("\n✅ Upload completed successfully!")
            print("🎉 Your knowledge base is now available in Pinecone")
            print("📋 LLM-based chunking strategy has been applied to the uploaded content")
            
            # Show some stats
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=settings.PINECONE_API_KEY)
                index = pc.Index(settings.PINECONE_INDEX_NAME)
                stats = index.describe_index_stats()
                
                print("\n📊 Index Statistics:")
                print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
                print(f"   Index dimension: {stats.get('dimension', 'Unknown')}")
                
                if 'namespaces' in stats:
                    default_ns = stats['namespaces'].get('__default__', {})
                    if default_ns:
                        print(f"   Vectors in __default__ namespace: {default_ns.get('vector_count', 0)}")
                
            except Exception as e:
                print(f"⚠️ Could not retrieve index stats: {e}")
        else:
            print("\n❌ Upload failed!")
            print("Please check your Pinecone configuration and try again")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 