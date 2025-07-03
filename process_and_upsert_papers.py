#!/usr/bin/env python3
"""
Process existing PDF papers with intelligent chunking (paragraph-based by default) and upsert to Pinecone
"""

import os
import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
sys.path.append('.')

from src.config import settings
from src.data_curation.llm_paper_collector import (
    LLMSemanticChunker, 
    extract_text_from_pdf
)

def load_existing_metadata() -> Dict[str, Dict]:
    """Load existing papers metadata"""
    metadata_file = os.path.join(settings.DATA_DIR, 'papers_metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
            # Convert to dict keyed by arxiv_id for easy lookup
            return {meta['arxiv_id']: meta for meta in metadata_list if 'arxiv_id' in meta}
    return {}

def load_existing_knowledge_base() -> List[Dict]:
    """Load existing knowledge base"""
    if os.path.exists(settings.KNOWLEDGE_BASE_FILE):
        with open(settings.KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def clean_unicode_text(text: str) -> str:
    """Clean text to remove problematic Unicode characters"""
    import unicodedata
    # Normalize Unicode and remove surrogate characters
    text = unicodedata.normalize('NFKD', text)
    # Remove surrogates and other problematic characters
    cleaned = ''.join(char for char in text if ord(char) < 0xD800 or ord(char) > 0xDFFF)
    return cleaned

def create_paragraph_chunks(text: str, title: str, paper_id: str, 
                          source: str, authors: List[str], categories: List[str],
                          max_chunk_size: int = 1500, overlap_size: int = 200) -> List[Dict]:
    """
    Create chunks based on paragraph boundaries with intelligent merging
    
    Args:
        text: The full text to chunk
        title: Paper title
        paper_id: Paper ID
        source: Source URL
        authors: List of authors
        categories: List of categories
        max_chunk_size: Maximum size of each chunk in characters
        overlap_size: Overlap between chunks in characters
        
    Returns:
        List of chunk entries for knowledge base
    """
    # Use sentence-aware chunking for better content integrity
    # First split by paragraphs, then by sentences within paragraphs
    import re
    
    # Split text into sentences (more sophisticated sentence detection)
    sentence_endings = r'[.!?]+(?:\s|$)'
    sentences = re.split(sentence_endings, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    chunk_count = 0
    
    for sentence in sentences:
        # Add sentence ending back (except for last sentence)
        if sentence != sentences[-1] and not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        
        # Check if adding this sentence would exceed max size
        test_chunk = current_chunk + (' ' if current_chunk else '') + sentence
        
        if len(test_chunk) > max_chunk_size and current_chunk:
            # Current chunk is complete, save it
            chunks.append(current_chunk.strip())
            chunk_count += 1
            
            # Start new chunk with overlap if specified
            if overlap_size > 0 and len(current_chunk) > overlap_size:
                # Find good overlap point (complete sentences only)
                words = current_chunk.split()
                overlap_text = ""
                for i in range(len(words) - 1, 0, -1):
                    candidate = ' '.join(words[i:])
                    if len(candidate) <= overlap_size and candidate.endswith(('.', '!', '?')):
                        overlap_text = candidate
                        break
                current_chunk = overlap_text + (' ' if overlap_text else '') + sentence
            else:
                current_chunk = sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += ' ' + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        chunk_count += 1
    
    # Create chunk entries
    chunk_entries = []
    for i, chunk in enumerate(chunks):
        if chunk.strip():  # Only add non-empty chunks
            chunk_entry = {
                "id": f"{paper_id}_chunk_{i+1}",
                "title": f"{title} (Part {i+1}/{len(chunks)})",
                "content": chunk.strip(),
                "source": source,
                "authors": authors,
                "categories": categories,
                "type": "chunk",
                "chunk_index": i + 1,
                "total_chunks": len(chunks),
                "parent_paper_id": paper_id
            }
            chunk_entries.append(chunk_entry)
    
    return chunk_entries

def process_existing_pdfs_with_chunking():
    """
    Process all existing PDF files with intelligent chunking and prepare for Pinecone upload
    Supports multiple chunking methods: paragraph-based (default), LLM-based, and traditional
    """
    print("🔄 Processing existing PDF papers with intelligent chunking...")
    
    # Load existing data
    papers_metadata = load_existing_metadata()
    knowledge_base = load_existing_knowledge_base()
    
    print(f"📚 Found {len(papers_metadata)} papers in metadata")
    print(f"📖 Found {len(knowledge_base)} entries in knowledge base")
    
    # Find all PDF files
    pdf_dir = Path(settings.PAPERS_DIR)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF files to process")
    
    if not pdf_files:
        print("❌ No PDF files found in papers directory")
        return False
    
    # Choose chunking method:
    # "paragraph" - Fast paragraph-based chunking (recommended)
    # "llm" - Intelligent LLM-based semantic chunking (slower but smarter)
    # "traditional" - Basic recursive character splitting
    chunking_method = "paragraph"  # Change this to your preferred method
    
    if chunking_method == "llm":
        # Initialize LLM chunker with Llama model
        # Popular Llama models: "llama3.2", "llama3.1", "llama3", "codellama"
        # To use OpenAI instead, change to: "gpt-3.5-turbo" or "gpt-4"
        llm_model = "llama3.2"  # Change this to your preferred model
        print(f"🦙 Using LLM-based semantic chunking with {llm_model}")
        
        chunker = LLMSemanticChunker(
            llm_model=llm_model,
            max_chunk_size=settings.CHUNK_SIZE
        )
    elif chunking_method == "paragraph":
        print("📄 Using paragraph-based chunking (fast and smart)")
        # Will be handled in the processing loop
        chunker = None
    else:  # traditional
        print("⚡ Using traditional recursive character chunking")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    # Remove existing chunk entries to avoid duplicates
    existing_ids = {entry['id'] for entry in knowledge_base}
    knowledge_base = [entry for entry in knowledge_base if entry.get('type') != 'chunk']
    print(f"🧹 Removed existing chunks, keeping {len(knowledge_base)} non-chunk entries")
    
    # Process each PDF
    processed_count = 0
    chunk_count = 0
    
    for pdf_file in pdf_files:
        arxiv_id = pdf_file.stem  # Get filename without extension
        
        print(f"\n📄 Processing: {arxiv_id}")
        
        # Load PDF and extract text
        try:
            with open(pdf_file, 'rb') as f:
                pdf_data = f.read()
            
            content = extract_text_from_pdf(pdf_data)
            if not content or len(content.strip()) < 100:
                print(f"  ⚠️  Skipping {arxiv_id}: No meaningful content extracted")
                continue
            
            # Clean Unicode issues
            content = clean_unicode_text(content)
            
            # Get metadata for this paper
            metadata = papers_metadata.get(arxiv_id, {})
            title = metadata.get('title', f'Unknown Paper ({arxiv_id})')
            authors = metadata.get('authors', [])
            categories = metadata.get('categories', ['cs.CL'])
            source = metadata.get('url', f'arXiv:{arxiv_id}')
            
            print(f"  📊 Content length: {len(content):,} characters")
            print(f"  🏷️  Title: {title[:80]}...")
            
            # Create chunks using selected method
            if chunking_method == "llm":
                print(f"  🦙 Creating semantic chunks with LLM...")
                chunk_entries = chunker.create_semantic_chunks(
                    text=content,
                    title=title,
                    paper_id=arxiv_id,
                    source=source,
                    authors=authors,
                    categories=categories
                )
            elif chunking_method == "paragraph":
                print(f"  📄 Creating paragraph-based chunks...")
                chunk_entries = create_paragraph_chunks(
                    text=content,
                    title=title,
                    paper_id=arxiv_id,
                    source=source,
                    authors=authors,
                    categories=categories,
                    max_chunk_size=settings.CHUNK_SIZE,
                    overlap_size=settings.CHUNK_OVERLAP
                )
            else:  # traditional
                print(f"  ⚡ Creating traditional recursive chunks...")
                # Use traditional chunking
                text_chunks = chunker.split_text(content)
                chunk_entries = []
                
                for i, chunk in enumerate(text_chunks):
                    if chunk.strip():  # Only add non-empty chunks
                        chunk_entry = {
                            "id": f"{arxiv_id}_chunk_{i+1}",
                            "title": f"{title} (Part {i+1}/{len(text_chunks)})",
                            "content": chunk.strip(),
                            "source": source,
                            "authors": authors,
                            "categories": categories,
                            "type": "chunk",
                            "chunk_index": i + 1,
                            "total_chunks": len(text_chunks),
                            "parent_paper_id": arxiv_id
                        }
                        chunk_entries.append(chunk_entry)
            
            # Add chunks to knowledge base
            for chunk_entry in chunk_entries:
                if chunk_entry['id'] not in existing_ids:
                    # Clean Unicode in chunk content
                    chunk_entry['content'] = clean_unicode_text(chunk_entry['content'])
                    chunk_entry['title'] = clean_unicode_text(chunk_entry['title'])
                    
                    knowledge_base.append(chunk_entry)
                    existing_ids.add(chunk_entry['id'])
                    chunk_count += 1
            
            processed_count += 1
            print(f"  ✅ Created {len(chunk_entries)} chunks for {title[:50]}...")
            
        except Exception as e:
            print(f"  ❌ Error processing {arxiv_id}: {e}")
            continue
    
    print(f"\n🎉 Processing Complete!")
    print(f"  📄 Processed {processed_count} PDF files")
    print(f"  🧩 Created {chunk_count} semantic chunks")
    print(f"  📚 Total knowledge base entries: {len(knowledge_base)}")
    
    # Save updated knowledge base with Unicode handling
    try:
        with open(settings.KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=True)
        print(f"✅ Saved updated knowledge base with {len(knowledge_base)} entries")
    except Exception as e:
        print(f"❌ Error saving knowledge base: {e}")
        return False
    
    return True

def upsert_to_pinecone():
    """
    Upsert the processed knowledge base to Pinecone using LangChain agents
    """
    print("\n🚀 Upserting knowledge base to Pinecone...")
    
    try:
        # Import the LangChain agents class
        from src.agents.langchain_agents import LangChainMLAgents
        
        print("🔄 Initializing LangChain agents...")
        
        # Create agent instance for Pinecone functionality
        agents = LangChainMLAgents()
        
        # Use the unified upsert method
        print("📤 Starting Pinecone upsert...")
        success = agents.upsert_knowledge_base_to_pinecone()
        
        if success:
            print("✅ Successfully upserted knowledge base to Pinecone!")
            return True
        else:
            print("❌ Failed to upsert to Pinecone")
            return False
            
    except Exception as e:
        print(f"❌ Error during Pinecone upsert: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Process PDFs with intelligent chunking (paragraph/LLM/traditional) and upsert to Pinecone')
    parser.add_argument('--chunk-only', action='store_true', help='Only create chunks, do not upsert to Pinecone')
    parser.add_argument('--upsert-only', action='store_true', help='Only upsert existing knowledge base to Pinecone')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if chunks exist')
    
    args = parser.parse_args()
    
    print("🔧 Intelligent PDF Processor & Pinecone Upserter")
    print("=" * 60)
    
    success = True
    
    if not args.upsert_only:
        # Process PDFs with intelligent chunking
        print("📋 Step 1: Processing PDFs with intelligent chunking...")
        success = process_existing_pdfs_with_chunking()
        
        if not success:
            print("❌ PDF processing failed. Stopping.")
            return
    
    if not args.chunk_only and success:
        # Upsert to Pinecone
        print("\n📋 Step 2: Upserting to Pinecone...")
        success = upsert_to_pinecone()
    
    if success:
        print("\n🎊 All operations completed successfully!")
        print("🔍 Your papers are now processed with semantic chunks and available in Pinecone!")
    else:
        print("\n⚠️  Some operations failed. Check the logs above.")

if __name__ == "__main__":
    main() 