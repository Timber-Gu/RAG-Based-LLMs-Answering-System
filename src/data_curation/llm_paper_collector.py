"""
LLM Paper Collector
Fetches recent LLM research papers from arXiv and processes them for the knowledge base.
"""
import os
import json
import arxiv
import requests
import PyPDF2
import io
import time
import random
import hashlib
from typing import Dict, List, Optional, Tuple

# Add src to path for relative imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import settings

# Import LangChain components for LLM-based chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

class LLMSemanticChunker:
    """
    LLM-based semantic text chunker that intelligently splits documents
    based on semantic boundaries and content structure.
    """
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo", max_chunk_size: int = 1500):
        """
        Initialize the LLM-based chunker.
        
        Args:
            llm_model: The LLM model to use for semantic analysis
            max_chunk_size: Maximum size of each chunk in characters
        """
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,  # Use deterministic output for consistent chunking
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.max_chunk_size = max_chunk_size
        
        # Fallback to traditional chunking if LLM fails
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Prompt template for semantic boundary detection
        self.boundary_prompt = PromptTemplate(
            input_variables=["text", "max_chunk_size"],
            template="""
You are an expert at analyzing academic papers and identifying semantic boundaries.
Your task is to identify the best places to split the following text into meaningful chunks.

Rules:
1. Each chunk should contain a complete thought or concept
2. Prefer splitting at section boundaries, paragraph breaks, or natural topic transitions
3. Avoid splitting in the middle of equations, code blocks, or tables
4. Each chunk should be roughly {max_chunk_size} characters, but prioritize semantic coherence
5. Return the split positions as character indices

Text to analyze:
{text}

Please identify the optimal split positions (as character indices) that would create semantically coherent chunks.
Return only the indices as a comma-separated list (e.g., "150, 320, 480").
If no good split points exist, return "NO_SPLIT".
"""
        )
    
    def chunk_text(self, text: str, title: str = "") -> List[str]:
        """
        Split text into semantically coherent chunks using LLM analysis.
        
        Args:
            text: The text to chunk
            title: Optional title for context
            
        Returns:
            List of text chunks
        """
        # If text is small enough, return as single chunk
        if len(text) <= self.max_chunk_size:
            return [text]
        
        try:
            return self._llm_based_chunking(text, title)
        except Exception as e:
            print(f"LLM chunking failed: {e}, falling back to traditional chunking")
            return self._fallback_chunking(text)
    
    def _llm_based_chunking(self, text: str, title: str = "") -> List[str]:
        """
        Perform LLM-based semantic chunking.
        """
        chunks = []
        remaining_text = text
        
        while len(remaining_text) > self.max_chunk_size:
            # Take a section of text that's larger than max_chunk_size for analysis
            analysis_section = remaining_text[:self.max_chunk_size * 2]
            
            # Get LLM's suggestion for split points
            prompt = self.boundary_prompt.format(
                text=analysis_section,
                max_chunk_size=self.max_chunk_size
            )
            
            response = self.llm.invoke(prompt)
            split_suggestions = response.content.strip()
            
            if split_suggestions == "NO_SPLIT" or not split_suggestions:
                # If no good split point, use fallback
                chunk = remaining_text[:self.max_chunk_size]
                chunks.append(chunk)
                remaining_text = remaining_text[self.max_chunk_size:]
            else:
                # Parse split positions
                try:
                    positions = [int(pos.strip()) for pos in split_suggestions.split(",")]
                    # Find the best position within our target range
                    best_position = None
                    for pos in positions:
                        if self.max_chunk_size * 0.7 <= pos <= self.max_chunk_size * 1.3:
                            best_position = pos
                            break
                    
                    if best_position:
                        chunk = remaining_text[:best_position]
                        chunks.append(chunk)
                        remaining_text = remaining_text[best_position:]
                    else:
                        # No position in good range, use traditional split
                        chunk = remaining_text[:self.max_chunk_size]
                        chunks.append(chunk)
                        remaining_text = remaining_text[self.max_chunk_size:]
                        
                except (ValueError, IndexError):
                    # If parsing fails, use traditional split
                    chunk = remaining_text[:self.max_chunk_size]
                    chunks.append(chunk)
                    remaining_text = remaining_text[self.max_chunk_size:]
        
        # Add remaining text as final chunk
        if remaining_text.strip():
            chunks.append(remaining_text)
        
        return chunks
    
    def _fallback_chunking(self, text: str) -> List[str]:
        """
        Fallback to traditional recursive character text splitting.
        """
        return self.fallback_splitter.split_text(text)
    
    def create_semantic_chunks(self, text: str, title: str, paper_id: str, 
                             source: str, authors: List[str], categories: List[str]) -> List[Dict]:
        """
        Create knowledge base entries from semantically chunked text.
        
        Args:
            text: The full text to chunk
            title: Paper title
            paper_id: Paper ID
            source: Source URL
            authors: List of authors
            categories: List of categories
            
        Returns:
            List of chunk entries for knowledge base
        """
        chunks = self.chunk_text(text, title)
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

# LLM-related search queries
LLM_SEARCH_QUERIES = [
    "large language models",
    "transformer architectures",
    "attention mechanisms",
    "llama model",
    "gpt architecture",
    "claude ai",
    "mistral ai",
    "prompt engineering",
    "in-context learning",
    "chain-of-thought",
    "self-attention",
    "retrieval augmented generation",
    "instruction tuning",
    "constitutional ai",
    "alignment techniques",
    "language model evaluation",
    "inference optimization",
    "quantization methods",
    "parameter efficient fine-tuning",
    "foundation models",
    "multimodal llm",
    "vision language models",
    "llm reasoning",
    "tokenization techniques",
    "transformer scaling",
    "sparse attention",
    "mixture of experts",
    "distillation llm",
    "language model alignment",
    "efficient transformers",
    "token merging",
    "neural scaling laws",
    "llm interpretability",
    "emergent abilities",
    "context window extension",
    "rotary positional encoding",
    "flash attention algorithm",
    "chatbot architecture",
    "agent llm systems",
    "language model safety",
    "llm evaluation benchmarks",
    "multilingual language models",
    "code generation models",
    "sequence modeling",
    "transformer optimization"
]

# Key LLM concepts and algorithms
LLM_CONCEPTS = [
    "Transformer Architecture",
    "Self-Attention Mechanism",
    "Multi-head Attention",
    "Positional Encoding",
    "Masked Language Modeling",
    "Next-Token Prediction",
    "Transfer Learning",
    "Fine-tuning",
    "Zero-shot Learning",
    "Few-shot Learning",
    "In-context Learning",
    "Chain-of-Thought Prompting",
    "Retrieval Augmented Generation",
    "Reinforcement Learning from Human Feedback",
    "Constitutional AI",
    "Direct Preference Optimization",
    "Mixture of Experts",
    "Flash Attention",
    "Key-Value Cache",
    "Rotary Position Embedding",
    "Grouped Query Attention",
    "Sliding Window Attention",
    "Parameter-Efficient Fine-Tuning",
    "Low-Rank Adaptation",
    "Quantization",
    "Knowledge Distillation",
    "Sparse Attention",
    "Continuous Batching",
    # Additional concepts with detailed descriptions
    {
        "name": "GPT (Generative Pre-trained Transformer)",
        "description": "A family of large language models developed by OpenAI. GPT models use the transformer architecture and are trained using a combination of unsupervised pre-training on large text corpora and supervised fine-tuning. GPT models generate text by predicting the next token given previous context."
    },
    {
        "name": "LLaMA (Large Language Model Meta AI)",
        "description": "A series of foundation language models developed by Meta AI, known for their efficiency and strong performance despite smaller parameter counts. LLaMA models have been widely used as open-weight models that serve as the foundation for many fine-tuned variants."
    },
    {
        "name": "Claude",
        "description": "A family of large language models developed by Anthropic, focused on helpful, harmless, and honest AI. Claude models are trained using Constitutional AI and RLHF techniques to align with human values and reduce harmful outputs."
    },
    {
        "name": "RLHF (Reinforcement Learning from Human Feedback)",
        "description": "A technique used to align language models with human preferences. RLHF involves collecting human feedback on model outputs, training a reward model based on this feedback, and then using reinforcement learning to optimize the language model against the reward model."
    },
    {
        "name": "RAG (Retrieval-Augmented Generation)",
        "description": "A technique that enhances language model outputs by retrieving relevant documents from an external knowledge base. RAG combines the knowledge access of information retrieval systems with the fluent text generation capabilities of large language models."
    },
    {
        "name": "LoRA (Low-Rank Adaptation)",
        "description": "A parameter-efficient fine-tuning method that freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks."
    },
    {
        "name": "QLoRA (Quantized Low-Rank Adaptation)",
        "description": "An extension of LoRA that quantizes the pre-trained language model to 4-bit precision while maintaining high performance. QLoRA enables fine-tuning of models on consumer GPUs that would otherwise be too large to fit in memory."
    },
    {
        "name": "PEFT (Parameter-Efficient Fine-Tuning)",
        "description": "A family of techniques that allow adaptation of pre-trained language models to specific tasks using only a small number of trainable parameters. PEFT methods include adapter layers, prefix tuning, prompt tuning, and LoRA."
    }
]

def get_arxiv_papers(query: str, max_results: int = 10, start_date: str = None, end_date: str = None) -> List[arxiv.Result]:
    """
    Fetch papers from arXiv based on query
    """
    search_query = query
    
    # Add date filtering if provided
    if start_date and end_date:
        search_query += f" AND submittedDate:[{start_date} TO {end_date}]"
    elif start_date:
        search_query += f" AND submittedDate:[{start_date} TO *]"
    elif end_date:
        search_query += f" AND submittedDate:[* TO {end_date}]"
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    try:
        results = list(client.results(search))
        print(f"Found {len(results)} papers for query: {search_query}")
        return results
    except Exception as e:
        print(f"Error fetching papers for query '{search_query}': {e}")
        return []

def extract_text_from_pdf(pdf_data: bytes) -> str:
    """
    Extract text from PDF data
    """
    try:
        pdf_file = io.BytesIO(pdf_data)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        # Extract text from all pages
        max_pages = len(reader.pages)
        for i in range(max_pages):
            text += reader.pages[i].extract_text() + "\n\n"
        
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def download_pdf(url: str) -> Optional[bytes]:
    """
    Download PDF from URL
    """
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.content
        else:
            print(f"Failed to download PDF: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None

def process_paper(paper: arxiv.Result, use_llm_chunking: bool = True) -> Tuple[Dict, List[Dict]]:
    """
    Process paper and extract content for knowledge base with intelligent chunking
    Returns: (metadata_entry, knowledge_base_entries)
    """
    # Create metadata entry
    metadata = {
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "abstract": paper.summary,
        "url": paper.entry_id,
        "pdf_url": paper.pdf_url,
        "published": paper.published.isoformat(),
        "categories": paper.categories,
        "arxiv_id": paper.get_short_id()
    }
    
    # Create abstract entry for knowledge base
    abstract_entry = {
        "id": f"{paper.get_short_id()}_abstract",
        "title": paper.title,
        "content": f"Title: {paper.title}\n\nAbstract: {paper.summary}",
        "source": paper.entry_id,
        "authors": [author.name for author in paper.authors],
        "categories": paper.categories,
        "type": "abstract"
    }
    
    # Download and extract content from PDF
    knowledge_entries = [abstract_entry]
    
    # Try to download the PDF and extract content
    pdf_data = download_pdf(paper.pdf_url)
    if pdf_data:
        content = extract_text_from_pdf(pdf_data)
        if content:
            # Save PDF to papers directory
            pdf_path = os.path.join(settings.PAPERS_DIR, f"{paper.get_short_id()}.pdf")
            try:
                with open(pdf_path, 'wb') as f:
                    f.write(pdf_data)
                print(f"Saved PDF to {pdf_path}")
            except Exception as e:
                print(f"Error saving PDF: {e}")
            
            # Decide on chunking strategy
            if use_llm_chunking and settings.OPENAI_API_KEY:
                try:
                    print(f"Using LLM-based semantic chunking for: {paper.title}")
                    # Initialize LLM chunker
                    chunker = LLMSemanticChunker(
                        llm_model="gpt-3.5-turbo",
                        max_chunk_size=settings.CHUNK_SIZE
                    )
                    
                    # Create semantic chunks
                    chunk_entries = chunker.create_semantic_chunks(
                        text=content,
                        title=paper.title,
                        paper_id=paper.get_short_id(),
                        source=paper.entry_id,
                        authors=[author.name for author in paper.authors],
                        categories=paper.categories
                    )
                    
                    knowledge_entries.extend(chunk_entries)
                    print(f"Created {len(chunk_entries)} semantic chunks for {paper.title}")
                    
                except Exception as e:
                    print(f"LLM chunking failed for {paper.title}: {e}")
                    # Fallback to traditional approach
                    content_entry = {
                        "id": f"{paper.get_short_id()}_content",
                        "title": paper.title,
                        "content": content,
                        "source": paper.entry_id,
                        "authors": [author.name for author in paper.authors],
                        "categories": paper.categories,
                        "type": "content"
                    }
                    knowledge_entries.append(content_entry)
            else:
                # Traditional approach - single content entry
                print(f"Using traditional content storage for: {paper.title}")
                content_entry = {
                    "id": f"{paper.get_short_id()}_content",
                    "title": paper.title,
                    "content": content,
                    "source": paper.entry_id,
                    "authors": [author.name for author in paper.authors],
                    "categories": paper.categories,
                    "type": "content"
                }
                knowledge_entries.append(content_entry)
    
    return metadata, knowledge_entries

def create_concept_entries() -> List[Dict]:
    """
    Create knowledge base entries for LLM concepts and algorithms
    """
    concept_entries = []
    
    for concept in LLM_CONCEPTS:
        if isinstance(concept, dict):
            # Handle detailed concept entry
            concept_id = hashlib.md5(concept["name"].encode()).hexdigest()[:10]
            
            entry = {
                "id": f"llm_concept_{concept_id}",
                "title": concept["name"],
                "content": f"Title: {concept['name']}\n\n{concept['description']}",
                "source": "LLM concept database",
                "authors": [],
                "categories": ["cs.CL", "cs.AI", "cs.LG"],
                "type": "concept"
            }
        else:
            # Handle simple concept name
            concept_id = hashlib.md5(concept.encode()).hexdigest()[:10]
            
            entry = {
                "id": f"llm_concept_{concept_id}",
                "title": concept,
                "content": f"Title: {concept}\n\nThis entry represents an important concept in Large Language Models.",
                "source": "LLM concept database",
                "authors": [],
                "categories": ["cs.CL", "cs.AI", "cs.LG"],
                "type": "concept"
            }
        
        concept_entries.append(entry)
    
    return concept_entries

def update_knowledge_base():
    """
    Main function to update knowledge base with LLM papers
    """
    # Create necessary directories
    os.makedirs(settings.PAPERS_DIR, exist_ok=True)
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    
    # Load existing knowledge base if it exists
    knowledge_base = []
    papers_metadata = []
    
    if os.path.exists(settings.KNOWLEDGE_BASE_FILE):
        with open(settings.KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
        print(f"Loaded {len(knowledge_base)} entries from existing knowledge base")
    
    if os.path.exists(os.path.join(settings.DATA_DIR, 'papers_metadata.json')):
        with open(os.path.join(settings.DATA_DIR, 'papers_metadata.json'), 'r', encoding='utf-8') as f:
            papers_metadata = json.load(f)
        print(f"Loaded metadata for {len(papers_metadata)} papers")
    
    # Track existing entries to avoid duplicates
    existing_ids = {entry['id'] for entry in knowledge_base}
    existing_arxiv_ids = {meta.get('arxiv_id') for meta in papers_metadata if meta.get('arxiv_id')}
    
    # Add concept entries
    concept_entries = create_concept_entries()
    for entry in concept_entries:
        if entry['id'] not in existing_ids:
            knowledge_base.append(entry)
            existing_ids.add(entry['id'])
    
    # Fetch papers for each query
    new_papers_count = 0
    
    # First pass: use high max_results for most important topics
    top_queries = [
        "large language models",
        "transformer architectures",
        "attention mechanisms",
        "llama model",
        "gpt architecture",
        "prompt engineering",
        "in-context learning",
        "retrieval augmented generation",
        "instruction tuning",
        "parameter efficient fine-tuning",
        "foundation models"
    ]
    
    for query in top_queries:
        if new_papers_count >= settings.MAX_PAPERS:
            break
            
        print(f"Searching for papers on: {query} (priority query)")
        papers = get_arxiv_papers(query, max_results=10)  # Increase max_results for important topics
        
        for paper in papers:
            arxiv_id = paper.get_short_id()
            
            # Skip if already processed
            if arxiv_id in existing_arxiv_ids:
                print(f"Skipping already processed paper: {arxiv_id}")
                continue
            
            print(f"Processing paper: {paper.title} ({arxiv_id})")
            
            # Process paper with LLM chunking if enabled
            metadata, entries = process_paper(paper, use_llm_chunking=settings.USE_LLM_CHUNKING)
            
            # Add to collections
            papers_metadata.append(metadata)
            existing_arxiv_ids.add(arxiv_id)
            
            for entry in entries:
                if entry['id'] not in existing_ids:
                    knowledge_base.append(entry)
                    existing_ids.add(entry['id'])
            
            new_papers_count += 1
            
            # Add delay to avoid rate limiting
            time.sleep(random.uniform(1, 2))
            
            # Limit total number of new papers
            if new_papers_count >= settings.MAX_PAPERS:
                print(f"Reached maximum number of papers ({settings.MAX_PAPERS})")
                break
    
    # Second pass: use lower max_results for remaining topics
    remaining_queries = [q for q in LLM_SEARCH_QUERIES if q not in top_queries]
    
    for query in remaining_queries:
        if new_papers_count >= settings.MAX_PAPERS:
            break
            
        print(f"Searching for papers on: {query}")
        papers = get_arxiv_papers(query, max_results=5)  # Lower max_results for less important topics
        
        for paper in papers:
            arxiv_id = paper.get_short_id()
            
            # Skip if already processed
            if arxiv_id in existing_arxiv_ids:
                print(f"Skipping already processed paper: {arxiv_id}")
                continue
            
            print(f"Processing paper: {paper.title} ({arxiv_id})")
            
            # Process paper with LLM chunking if enabled
            metadata, entries = process_paper(paper, use_llm_chunking=settings.USE_LLM_CHUNKING)
            
            # Add to collections
            papers_metadata.append(metadata)
            existing_arxiv_ids.add(arxiv_id)
            
            for entry in entries:
                if entry['id'] not in existing_ids:
                    knowledge_base.append(entry)
                    existing_ids.add(entry['id'])
            
            new_papers_count += 1
            
            # Add delay to avoid rate limiting
            time.sleep(random.uniform(1, 2))
            
            # Limit total number of new papers
            if new_papers_count >= settings.MAX_PAPERS:
                print(f"Reached maximum number of papers ({settings.MAX_PAPERS})")
                break
    
    # Save updated knowledge base
    with open(settings.KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(knowledge_base)} entries to knowledge base")
    
    # Save updated papers metadata
    with open(os.path.join(settings.DATA_DIR, 'papers_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(papers_metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata for {len(papers_metadata)} papers")

def upload_to_pinecone():
    """
    Upload knowledge base to Pinecone using LangChain-compatible approach
    This function now uses the unified LangChain agents upserting method
    """
    try:
        # Import the LangChain agents class
        from ..agents.langchain_agents import LangChainMLAgents
        
        print("🔄 Initializing LangChain agents for Pinecone upload...")
        
        # Create a minimal agent instance just for the Pinecone functionality
        # We don't need the full agent setup, just the Pinecone integration
        agents = LangChainMLAgents()
        
        # Use the unified upsert method
        success = agents.upsert_knowledge_base_to_pinecone()
        
        return success
        
    except Exception as e:
        print(f"❌ Error uploading to Pinecone: {e}")
        return False

if __name__ == "__main__":
    print("📚 LLM Paper Collector")
    print("=" * 50)
    
    # Update knowledge base
    update_knowledge_base()
    
    print("\n✅ Knowledge base updated successfully")
    print("To use this data with Pinecone, run the main application") 