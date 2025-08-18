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

# Import Ollama for Llama models
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama not installed. Install with: pip install langchain-ollama ollama")

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
                      OpenAI models: "gpt-3.5-turbo", "gpt-4", etc.
                      Ollama models: "llama3.2", "llama3.1", "codellama", etc.
            max_chunk_size: Maximum size of each chunk in characters
        """
        self.model_name = llm_model
        self.max_chunk_size = max_chunk_size
        
        # Check if this is an Ollama model (common Llama model names)
        ollama_models = [
            "llama3.2", "llama3.1", "llama3", "llama2", 
            "codellama", "wizardlm", "vicuna", "orca",
            "mistral", "mixtral", "phi", "gemma"
        ]
        
        is_ollama = any(model in llm_model.lower() for model in ollama_models)
        
        if is_ollama and OLLAMA_AVAILABLE:
            print(f"🦙 Using Ollama model: {llm_model}")
            self.llm = ChatOllama(
                model=llm_model,
                temperature=0,  # Use deterministic output for consistent chunking
            )
        else:
            print(f"🤖 Using OpenAI model: {llm_model}")
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key required for OpenAI models. Set OPENAI_API_KEY environment variable.")
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=0,  # Use deterministic output for consistent chunking
                openai_api_key=settings.OPENAI_API_KEY
            )
        
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

# LLM-related search queries - Expanded for comprehensive coverage
LLM_SEARCH_QUERIES = [
    # Core LLM Architectures
    "large language models",
    "transformer architectures", 
    "attention mechanisms",
    "self-attention",
    "multi-head attention",
    "cross-attention",
    "flash attention",
    "sparse attention",
    "local attention",
    "sliding window attention",
    "rotary positional encoding",
    "positional embeddings",
    
    # Popular LLM Models
    "gpt models",
    "gpt-4",
    "gpt-3.5",
    "llama model",
    "llama2",
    "llama3",
    "claude ai",
    "claude 3",
    "mistral ai",
    "mixtral",
    "gemini model",
    "palm model",
    "bert model",
    "t5 model",
    "bart model",
    "pegasus model",
    "electra model",
    "roberta model",
    "deberta model",
    "albert model",
    
    # Training Techniques
    "pre-training language models",
    "instruction tuning",
    "fine-tuning llm",
    "parameter efficient fine-tuning",
    "low-rank adaptation",
    "lora",
    "qlora",
    "adapter layers",
    "prefix tuning",
    "prompt tuning",
    "p-tuning",
    "delta tuning",
    "bitfit",
    "compacter",
    
    # Alignment and Safety
    "reinforcement learning from human feedback",
    "rlhf",
    "constitutional ai",
    "direct preference optimization",
    "dpo",
    "proximal policy optimization",
    "reward modeling",
    "human preference learning",
    "ai alignment",
    "ai safety",
    "harmless ai",
    "truthful ai",
    "red teaming llm",
    "adversarial prompts",
    "jailbreaking llm",
    
    # Prompting and In-Context Learning
    "prompt engineering",
    "in-context learning",
    "few-shot learning",
    "zero-shot learning",
    "chain-of-thought",
    "tree of thought",
    "step-by-step reasoning",
    "chain-of-thought prompting",
    "instruction following",
    "prompt optimization",
    "automatic prompt generation",
    "soft prompts",
    "discrete prompts",
    
    # Retrieval and Knowledge
    "retrieval augmented generation",
    "rag",
    "knowledge grounding",
    "external memory",
    "vector databases",
    "semantic search",
    "dense retrieval",
    "knowledge injection",
    "fact checking llm",
    "knowledge editing",
    "memory augmented llm",
    
    # Multimodal and Specialized Models
    "multimodal llm",
    "vision language models",
    "vlm",
    "text-to-image generation",
    "image-to-text generation",
    "clip model",
    "dalle",
    "stable diffusion",
    "midjourney",
    "flamingo model",
    "blip model",
    "instructblip",
    "llava model",
    "gpt-4v",
    "code generation models",
    "codex",
    "github copilot",
    "code llama",
    "starcoder",
    "codegen",
    
    # Efficiency and Optimization
    "llm inference optimization",
    "model compression",
    "quantization methods",
    "pruning llm",
    "knowledge distillation",
    "efficient transformers",
    "linear attention",
    "approximate attention",
    "low-rank approximation",
    "mixture of experts",
    "moe",
    "switch transformer",
    "glam model",
    "palm-2",
    
    # Scaling and Architecture
    "transformer scaling",
    "neural scaling laws",
    "emergent abilities",
    "scaling laws llm",
    "chinchilla scaling",
    "compute optimal training",
    "model parallelism",
    "pipeline parallelism",
    "tensor parallelism",
    "gradient accumulation",
    "mixed precision training",
    "bf16 training",
    "fp16 training",
    
    # Context and Memory
    "long context llm",
    "context window extension",
    "longformer",
    "big bird model",
    "linformer",
    "performer model",
    "synthesizer model",
    "memory efficient attention",
    "gradient checkpointing",
    "activation recomputation",
    
    # Training Infrastructure
    "distributed training llm",
    "federated learning llm",
    "continual learning llm",
    "catastrophic forgetting",
    "elastic weight consolidation",
    "progressive networks",
    "meta-learning llm",
    "transfer learning nlp",
    
    # Evaluation and Benchmarks
    "llm evaluation",
    "language model benchmarks",
    "glue benchmark",
    "superglue benchmark",
    "big-bench",
    "mmlu benchmark",
    "hellaswag",
    "arc benchmark",
    "truthfulqa",
    "gsm8k",
    "humaneval",
    "code evaluation",
    "reasoning evaluation",
    "common sense reasoning",
    
    # Interpretability and Analysis
    "llm interpretability",
    "attention visualization",
    "probing llm",
    "mechanistic interpretability",
    "concept bottleneck models",
    "feature attribution",
    "gradient-based explanations",
    "lime explanations",
    "shap explanations",
    "adversarial examples nlp",
    
    # Applications and Domains
    "conversational ai",
    "chatbot architecture",
    "dialogue systems",
    "question answering",
    "text summarization",
    "machine translation",
    "sentiment analysis",
    "named entity recognition",
    "information extraction",
    "text classification",
    "natural language inference",
    "reading comprehension",
    "story generation",
    "creative writing ai",
    "scientific writing ai",
    "code documentation",
    "code explanation",
    "automated debugging",
    
    # Advanced Techniques
    "gradient-free optimization",
    "evolutionary strategies llm",
    "genetic algorithms nlp",
    "neural architecture search",
    "automl nlp",
    "hyperparameter optimization",
    "early stopping",
    "learning rate scheduling",
    "optimizer comparison",
    "adam optimizer",
    "adamw optimizer",
    "rmsprop optimizer",
    
    # Tokenization and Preprocessing
    "tokenization techniques",
    "subword tokenization",
    "byte-pair encoding",
    "sentencepiece",
    "wordpiece",
    "unigram tokenization",
    "character-level models",
    "token merging",
    "vocabulary optimization",
    "multilingual tokenization",
    
    # Multilingual and Cross-lingual
    "multilingual language models",
    "cross-lingual transfer",
    "zero-shot cross-lingual",
    "language-agnostic models",
    "xlm model",
    "xlm-r",
    "mbert",
    "language adaptation",
    "code-switching nlp",
    
    # Domain-Specific Applications
    "biomedical llm",
    "legal llm",
    "financial llm",
    "scientific llm",
    "mathematical reasoning",
    "logical reasoning",
    "commonsense reasoning",
    "spatial reasoning",
    "temporal reasoning",
    "causal reasoning",
    
    # Recent Innovations
    "mixture of depths",
    "mamba model",
    "state space models",
    "selective state spaces",
    "structured state spaces",
    "liquid neural networks",
    "neural ode",
    "continuous time models",
    "diffusion language models",
    "score-based generative models",
    "flow-based models",
    "autoregressive flows",
    
    # Agent Systems
    "llm agents",
    "autonomous agents",
    "tool-using llm",
    "function calling",
    "api integration llm",
    "planning with llm",
    "reasoning agents",
    "multi-agent systems",
    "collaborative ai",
    "swarm intelligence nlp"
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
    
    # Fetch papers for each query with comprehensive coverage strategy
    new_papers_count = 0
    total_queries = len(LLM_SEARCH_QUERIES)
    
    print(f"🔍 Starting comprehensive paper collection across {total_queries} research areas...")
    
    # Dynamically adjust papers per query based on total target
    papers_per_query = max(1, settings.MAX_PAPERS // (total_queries // 3))  # Distribute papers across queries
    
    # Organize queries by priority for optimal coverage
    high_priority_queries = [
        "large language models", "transformer architectures", "attention mechanisms",
        "gpt models", "llama model", "claude ai", "mistral ai", "retrieval augmented generation",
        "instruction tuning", "parameter efficient fine-tuning", "rlhf", "constitutional ai",
        "prompt engineering", "in-context learning", "chain-of-thought", "multimodal llm"
    ]
    
    medium_priority_queries = [
        "quantization methods", "knowledge distillation", "mixture of experts", 
        "flash attention", "long context llm", "code generation models", "llm evaluation",
        "neural scaling laws", "emergent abilities", "llm interpretability"
    ]
    
    # Process high priority queries first
    for i, query in enumerate(high_priority_queries):
        if new_papers_count >= settings.MAX_PAPERS:
            break
            
        print(f"🎯 [{i+1}/{len(high_priority_queries)}] High Priority: {query}")
        papers = get_arxiv_papers(query, max_results=papers_per_query + 5)  # Extra papers for high priority
        
        query_papers_added = 0
        for paper in papers:
            if new_papers_count >= settings.MAX_PAPERS:
                break
                
            arxiv_id = paper.get_short_id()
            
            # Skip if already processed
            if arxiv_id in existing_arxiv_ids:
                continue
            
            print(f"  ✅ Processing: {paper.title[:80]}...")
            
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
            query_papers_added += 1
            
            # Add delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 1.5))
        
        print(f"    📊 Added {query_papers_added} papers from this query")
    
    # Process medium priority queries
    for i, query in enumerate(medium_priority_queries):
        if new_papers_count >= settings.MAX_PAPERS:
            break
            
        print(f"📋 [{i+1}/{len(medium_priority_queries)}] Medium Priority: {query}")
        papers = get_arxiv_papers(query, max_results=papers_per_query)
        
        query_papers_added = 0
        for paper in papers:
            if new_papers_count >= settings.MAX_PAPERS:
                break
                
            arxiv_id = paper.get_short_id()
            
            if arxiv_id in existing_arxiv_ids:
                continue
            
            print(f"  ✅ Processing: {paper.title[:80]}...")
            
            metadata, entries = process_paper(paper, use_llm_chunking=settings.USE_LLM_CHUNKING)
            
            papers_metadata.append(metadata)
            existing_arxiv_ids.add(arxiv_id)
            
            for entry in entries:
                if entry['id'] not in existing_ids:
                    knowledge_base.append(entry)
                    existing_ids.add(entry['id'])
            
            new_papers_count += 1
            query_papers_added += 1
            
            time.sleep(random.uniform(0.5, 1.5))
        
        print(f"    📊 Added {query_papers_added} papers from this query")
    
    # Process remaining queries with lower priority
    remaining_queries = [q for q in LLM_SEARCH_QUERIES if q not in high_priority_queries + medium_priority_queries]
    
    for i, query in enumerate(remaining_queries):
        if new_papers_count >= settings.MAX_PAPERS:
            break
            
        print(f"📄 [{i+1}/{len(remaining_queries)}] Comprehensive Coverage: {query}")
        papers = get_arxiv_papers(query, max_results=max(2, papers_per_query // 2))  # Fewer papers for remaining queries
        
        query_papers_added = 0
        for paper in papers:
            if new_papers_count >= settings.MAX_PAPERS:
                break
                
            arxiv_id = paper.get_short_id()
            
            if arxiv_id in existing_arxiv_ids:
                continue
            
            print(f"  ✅ Processing: {paper.title[:80]}...")
            
            metadata, entries = process_paper(paper, use_llm_chunking=settings.USE_LLM_CHUNKING)
            
            papers_metadata.append(metadata)
            existing_arxiv_ids.add(arxiv_id)
            
            for entry in entries:
                if entry['id'] not in existing_ids:
                    knowledge_base.append(entry)
                    existing_ids.add(entry['id'])
            
            new_papers_count += 1
            query_papers_added += 1
            
            time.sleep(random.uniform(0.5, 1.5))
        
        if query_papers_added > 0:
            print(f"    📊 Added {query_papers_added} papers from this query")
    
    print(f"\n🎉 Collection Summary:")
    print(f"   📄 Total new papers processed: {new_papers_count}")
    print(f"   🔍 Total queries executed: {min(len(LLM_SEARCH_QUERIES), new_papers_count)}")
    print(f"   📚 Total knowledge base entries: {len(knowledge_base)}")
    print(f"   📋 Total paper metadata records: {len(papers_metadata)}")
    
    # Save updated knowledge base with proper Unicode handling
    try:
        with open(settings.KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=True)  # Use ensure_ascii=True to avoid Unicode issues
        print(f"Saved {len(knowledge_base)} entries to knowledge base")
    except UnicodeEncodeError as e:
        print(f"Unicode error when saving knowledge base: {e}")
        # Try with ASCII encoding as fallback
        with open(settings.KNOWLEDGE_BASE_FILE, 'w', encoding='ascii', errors='ignore') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=True)
        print(f"Saved {len(knowledge_base)} entries to knowledge base (with Unicode fallback)")
    
    # Save updated papers metadata with proper Unicode handling
    try:
        with open(os.path.join(settings.DATA_DIR, 'papers_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(papers_metadata, f, indent=2, ensure_ascii=True)
        print(f"Saved metadata for {len(papers_metadata)} papers")
    except UnicodeEncodeError as e:
        print(f"Unicode error when saving metadata: {e}")
        with open(os.path.join(settings.DATA_DIR, 'papers_metadata.json'), 'w', encoding='ascii', errors='ignore') as f:
            json.dump(papers_metadata, f, indent=2, ensure_ascii=True)
        print(f"Saved metadata for {len(papers_metadata)} papers (with Unicode fallback)")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Paper Collector - Comprehensive Research Paper Database')
    parser.add_argument('--max-papers', type=int, default=100, help='Maximum number of papers to collect (default: 100)')
    parser.add_argument('--use-llm-chunking', action='store_true', help='Enable LLM for semantic chunking')
    parser.add_argument('--no-llm-chunking', action='store_true', help='Disable LLM semantic chunking (faster collection)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Override settings with command line arguments
    settings.MAX_PAPERS = args.max_papers
    
    # Handle LLM chunking logic (default to True unless explicitly disabled)
    if args.no_llm_chunking:
        settings.USE_LLM_CHUNKING = False
    elif args.use_llm_chunking:
        settings.USE_LLM_CHUNKING = True
    # If neither flag is provided, keep the default from settings (True)
    
    print("📚 LLM Paper Collector - Comprehensive Database Builder")
    print("=" * 60)
    print(f"🎯 Target papers: {settings.MAX_PAPERS}")
    print(f"🧠 LLM chunking: {'Enabled' if settings.USE_LLM_CHUNKING else 'Disabled'}")
    print(f"🔍 Search queries: {len(LLM_SEARCH_QUERIES)} topics")
    print("=" * 60)
    
    # Update knowledge base
    update_knowledge_base()
    
    print("\n✅ Knowledge base updated successfully!")
    print(f"📊 Database contains comprehensive coverage of {len(LLM_SEARCH_QUERIES)} LLM research areas")
    print("🚀 Ready for semantic search and retrieval!") 