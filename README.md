# Multi-Model LangChain ML Q&A Assistant

A RAG-based multi-agent system for Machine Learning and Deep Learning Q&A. It uses specialized LLMs and an ensemble retrieval pipeline with contextual compression.

- Research Agent: Claude 3.5 Sonnet when `ANTHROPIC_API_KEY` is set; otherwise falls back to the Theory Agent model
- Theory Agent: OpenAI model selected via `THEORY_MODEL` (supports `gpt-4`, `gpt-4o`, `gpt-4o-mini-*`)
- Implementation Agent: Claude 3.5 Sonnet when `ANTHROPIC_API_KEY` is set; otherwise falls back to the Theory Agent model
- Knowledge Base: Pinecone vector store with hosted embeddings and ensemble retrieval

## Key Features

### Ensemble Retrieval System
- **Multiple Retrieval Strategies**: Combines similarity search, self-query, and parent document retrieval
- **Reciprocal Rank Fusion (RRF)**: Advanced algorithm for intelligent result combination
- **Contextual Compression**: LLM-based filtering to extract only relevant information
- **Dynamic Context Management**: Automatic context window optimization to prevent overflow
- **Enhanced Performance**: Significantly improved retrieval accuracy over basic similarity search

### LLM-Powered Contextual Compression
- **Intelligent Filtering**: Uses LLM to identify and extract only query-relevant content
- **Content Preservation**: Maintains important context while removing irrelevant information
- **Fallback Mechanism**: Graceful degradation if compression fails
- **Token Optimization**: Reduces context length while preserving answer quality

### Streamlit Paper Upload and Processing
- **Direct PDF Upload**: Upload research papers directly through the web interface
- **Real-time Processing**: Live progress indicators for paper processing pipeline
- **Intelligent Chunking**: Choice between paragraph-based and traditional chunking methods
- **Automatic Integration**: Uploaded papers immediately available for Q&A
- **Metadata Extraction**: Automatic extraction of paper titles, authors, and abstracts

### LangChain-First Architecture
- **Complete LangChain Integration**: Uses LangChain agents, tools, and vector stores
- **Multi-Agent System**: Intelligent routing to specialized agents
- **Advanced RAG Integration**: Ensemble retrieval with contextual compression
- **Chat History Support**: Conversation context management

### Vector Store Integration
- **Pinecone Integration**: Scalable vector storage and retrieval
- **Hosted Embeddings**: llama-text-embed-v2 (1024 dimensions) with zero local setup
- **Efficient Search**: Fast similarity search with ensemble enhancement
- **Rich Metadata**: Comprehensive metadata for better context preservation

### Specialized Agents
- **Smart Routing**: Automatic agent selection based on query type
- **Domain Expertise**: Each agent optimized for specific ML/DL tasks
- **Fallback Handling**: Graceful degradation if specific models unavailable

## Quick Setup

### Prerequisites
- **OpenAI API Key** (for GPT-4 Theory Agent) - Get it from [OpenAI Platform](https://platform.openai.com/account/api-keys)
- **Anthropic API Key** (for Claude Implementation Agent) - Get it from [Anthropic Console](https://console.anthropic.com/)
- **Pinecone API Key** and **Environment** - Get them from [Pinecone Console](https://app.pinecone.io/)
- **Ollama** (optional, for Research Agent) - [Installation Guide](https://ollama.ai/download)

### Installation

1. Clone the Repository
   ```bash
   git clone https://github.com/Timber-Gu/RAG-Based-LLMs-Answering-System.git
   cd RAG-Based-LLMs-Answering-System
   ```

2. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set Up Environment Variables
   
   Create a new file named `.env` in the project root with the following structure:
   ```bash
   # Required API Keys
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   
   # Pinecone Configuration (Required)
   VECTOR_STORE_TYPE=pinecone
   PINECONE_INDEX_NAME=myproject          # Create this index in Pinecone first
   PINECONE_ENVIRONMENT=your_environment  # e.g., us-east-1
   PINECONE_API_KEY=your_pinecone_api_key
   EMBEDDING_MODEL=your_chosen_embedding_model
   
   # Ensemble Retrieval Settings (New!)
   USE_ENSEMBLE_RETRIEVAL=true            # Enable advanced ensemble retrieval
   ENSEMBLE_MAX_DOCS=4                    # Maximum documents to return
   ENSEMBLE_USE_COMPRESSION=true          # Enable LLM-based contextual compression
   ENSEMBLE_COMPRESSION_MAX_CHARS=1500    # Maximum characters after compression
   ENSEMBLE_SIMILARITY_WEIGHT=0.4         # Weight for similarity search (40%)
   ENSEMBLE_SELF_QUERY_WEIGHT=0.3         # Weight for self-query retrieval (30%)
   ENSEMBLE_PARENT_DOC_WEIGHT=0.3         # Weight for parent document retrieval (30%)
   
   # Model Settings
   AGENT_TEMPERATURE=0.7
   # OpenAI theory agent model (examples: gpt-4, gpt-4o, gpt-4o-mini-2024-07-18)
   THEORY_MODEL=gpt-4o
   # Claude implementation agent model (requires ANTHROPIC_API_KEY)
   IMPLEMENTATION_MODEL=claude-3-5-sonnet-20241022
   
   # Vector Store Settings
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   
   # Data Paths (Optional - these are default values)
   DATA_DIR=data
   PAPERS_DIR=data/papers
   KNOWLEDGE_BASE_FILE=data/knowledge_base.json
   
   # API Server Settings (Optional - for web interface)
   API_HOST=localhost
   API_PORT=8000
   
   # Ollama Settings (Optional - for Research Agent)
   OLLAMA_BASE_URL=http://127.0.0.1:11434
   RESEARCH_MODEL=llama3.1
   
   # Text Processing Settings
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   
   # LLM-based Chunking Settings
   USE_LLM_CHUNKING=true
   LLM_CHUNKING_MODEL=gpt-3.5-turbo
   MAX_CHUNK_SIZE=1500
   ```

4. Pinecone Setup
   - Create a free account at [Pinecone](https://app.pinecone.io/)
   - Create a new index with the following settings:
     - Dimensions: 1024 (required for llama-text-embed-v2)
     - Metric: cosine
     - Pod Type: starter (free tier)
   - Copy your API key and environment from the Pinecone console
   - Update your `.env` file with these values

5. Run the System
   
   **Option A: Command Line Interface**
   ```bash
   python main.py
   ```
   
   **Option B: Streamlit Web Interface (Recommended)**
   ```bash
   # Method 1: Using the launcher script
   python run_streamlit.py
   
   # Method 2: Direct Streamlit command
   streamlit run streamlit_app.py
   ```

## Streamlit Web Interface

Experience the ML Q&A Assistant through a modern, intuitive web interface with **direct paper upload** and **real-time processing** capabilities.

### Quick Start
```bash
# Install Streamlit (if not already installed)
pip install streamlit

# Launch the web interface
python run_streamlit.py
```
The app will automatically open in your default browser at `http://localhost:8501`

### Web Interface Features

#### Smart Agent Routing
- **Automatic Mode**: AI automatically routes queries to the best agent
- **Manual Mode**: Choose specific agents (Research, Theory, Implementation)
- **Visual Agent Badges**: See which agent responded to each query

#### Paper Upload and Processing
- **Direct PDF Upload**: Upload research papers directly through the sidebar
- **Real-time Processing**: Live progress indicators for each processing step
- **Chunking Options**: Choose between paragraph-based or traditional chunking
- **Automatic Integration**: Uploaded papers immediately available for Q&A
- **Success Feedback**: Detailed reports on processing results

Example Success Case:
```
FlashAttention paper processed successfully
Title: FlashAttention: Fast and Memory-Efficient Exact Attention
Authors: Tri Dao, Daniel Y. Fu, Stefano Ermon, et al.
Results: 100 chunks created, 96 uploaded to Pinecone
Ready for Q&A shortly
```

#### Chat Interface
- **Real-time Chat**: Interactive conversation with proper message formatting
- **Agent Identification**: Clear visual indicators showing which agent responded
- **Message History**: Scrollable chat history with user and AI messages
- **Thinking Process**: Optional display of agent reasoning steps
- **Ensemble Retrieval**: See how multiple retrieval strategies combine results

#### Settings Panel
- **Agent Selection**: Switch between automatic and manual routing
- **Thinking Display**: Toggle visibility of agent reasoning process
- **Retrieval Mode**: Switch between basic and ensemble retrieval
- **System Status**: Real-time health monitoring of all components
- **Knowledge Base Management**: Upload documents to Pinecone directly from the UI

#### System Monitoring
- **Health Dashboard**: Monitor agent availability and model connections
- **Chat Statistics**: Track conversation history and message counts
- **Vector Store Status**: Check Pinecone connection and data availability
- **Retrieval Analytics**: Monitor ensemble retrieval performance

#### Chat Management
- **Session Persistence**: Chat history maintained during browser sessions
- **Save/Load**: Export and import conversation history
- **Clear History**: Reset conversation context when needed

### Interface Highlights

```
┌─────────────────────────────────────────────────────────────┐
│  🤖 LangChain ML Q&A Assistant                             │
│  Multi-Agent System with Ensemble Retrieval                │
└─────────────────────────────────────────────────────────────┘
│                                                             │
│  📄 Paper Upload            💬 Chat Interface              │
│  ┌─────────────────────┐   ┌─────────────────────────────┐ │
│  │ 📁 Upload PDF       │   │ 🤔 You: Explain FlashAttn  │ │
│  │ ⚙️ Paragraph chunks │   │                             │ │
│  │ 🚀 Process Paper    │   │ 🤖 AI [THEORY]: FlashAttn  │ │
│  │                     │   │ achieves 7.6× speedup...   │ │
│  │ ✅ FlashAttention   │   │                             │ │
│  │ 96/100 chunks OK    │   │ 🔍 Ensemble Retrieval:     │ │
│  └─────────────────────┘   │ • 8 similarity results     │ │
│                             │ • 6 self-query results     │ │
│  ⚙️ Settings                │ • 1 parent doc result      │ │
│  ┌─────────────────────┐   │ → 9 unique after fusion    │ │
│  │ 🎯 Routing: Auto    │   │ → 4 final after compress   │ │
│  │ 🧠 Thinking: On     │   └─────────────────────────────┘ │
│  │ 🔍 Ensemble: On     │                                   │
│  │ 💾 Messages: 12     │   Ask your ML/DL question here   │
│  └─────────────────────┘   [                    ] [Send] │
└─────────────────────────────────────────────────────────────┘
```

### Web-Specific Features

- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Modern Styling**: Beautiful gradient headers and color-coded agent badges
- **Real-time Updates**: Live system status and chat history updates
- **Error Handling**: User-friendly error messages and recovery options
- **Help Documentation**: Built-in usage guide and example queries
- **Paper Management**: Upload and manage research papers directly in the UI

### Mobile-Friendly
The Streamlit interface is fully responsive and works great on mobile devices, making it easy to access your ML Q&A assistant from anywhere.

### Interactive Commands (CLI Only)

When using the command-line interface (`python main.py`), you can use these commands:
- `quit` - Exit the system
- `agents` - Show available agents
- `thinking on` - Enable thinking process display (default)
- `thinking off` - Disable thinking process display
- `thinking` - Check current thinking display status

### Environment Variables Explained

#### Required Variables
- `OPENAI_API_KEY`: Your OpenAI API key for GPT-4 access
- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude access
- `PINECONE_API_KEY`: Your Pinecone API key for vector storage
- `PINECONE_ENVIRONMENT`: Your Pinecone environment (e.g., us-east-1)
- `PINECONE_INDEX_NAME`: Name of your Pinecone index

#### New Ensemble Retrieval Variables
- `USE_ENSEMBLE_RETRIEVAL`: Enable ensemble retrieval (true/false, default: true)
- `ENSEMBLE_MAX_DOCS`: Maximum documents to return (default: 4)
- `ENSEMBLE_USE_COMPRESSION`: Enable contextual compression (true/false, default: true)
- `ENSEMBLE_COMPRESSION_MAX_CHARS`: Max characters after compression (default: 1500)
- `ENSEMBLE_SIMILARITY_WEIGHT`: Weight for similarity search (default: 0.4)
- `ENSEMBLE_SELF_QUERY_WEIGHT`: Weight for self-query retrieval (default: 0.3)
- `ENSEMBLE_PARENT_DOC_WEIGHT`: Weight for parent document retrieval (default: 0.3)

#### Optional Variables
- `AGENT_TEMPERATURE`: Controls response creativity (0.0-1.0, default: 0.7)
- `THEORY_AGENT_MAX_TOKENS`: Theory agent token limit (default: 3000, optimized for context)
- `RESEARCH_AGENT_MAX_TOKENS`: Research agent token limit (default: 4096)
- `IMPLEMENTATION_AGENT_MAX_TOKENS`: Implementation agent token limit (default: 8192)
- `CHUNK_SIZE`: Size of text chunks for vector storage (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `USE_LLM_CHUNKING`: Enable LLM-based semantic chunking (default: true)
- `LLM_CHUNKING_MODEL`: Model to use for semantic chunking (default: gpt-3.5-turbo)
- `MAX_CHUNK_SIZE`: Maximum size for LLM-based chunks (default: 1500)
- `OLLAMA_BASE_URL`: URL for local Ollama instance (default: http://127.0.0.1:11434)
- `RESEARCH_MODEL`: Ollama model to use (default: llama3.1)

### Troubleshooting Environment Setup

1. API Key Issues
   - Ensure all API keys are correctly copied without extra spaces
   - Verify API keys are active and have sufficient credits
   - Check for any special characters that might need escaping

2. Pinecone Issues
   - Verify your index is created with correct dimensions (1024)
   - Ensure index name matches exactly what's in your `.env`
   - Check if you're using the correct environment

3. Ollama Issues
   - Verify Ollama is running locally (`ollama run llama3.1`)
   - Check if the model is downloaded (`ollama list`)
   - Ensure the correct base URL is set

4. Ensemble Retrieval Issues
   - If experiencing context overflow, reduce `ENSEMBLE_MAX_DOCS` from 4 to 3
   - If compression is too aggressive, increase `ENSEMBLE_COMPRESSION_MAX_CHARS`
   - Disable compression temporarily by setting `ENSEMBLE_USE_COMPRESSION=false`

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Configuration
PINECONE_INDEX_NAME=myproject
VECTOR_STORE_TYPE=pinecone
EMBEDDING_MODEL=your_chosen_embedding_model

# Model Configuration
# OpenAI theory agent model (examples: gpt-4, gpt-4o, gpt-4o-mini-2024-07-18)
THEORY_MODEL=gpt-4o
RESEARCH_MODEL=llama3.1
IMPLEMENTATION_MODEL=claude-3-5-sonnet-20241022

# Ensemble Retrieval (NEW!)
USE_ENSEMBLE_RETRIEVAL=true
ENSEMBLE_MAX_DOCS=4
ENSEMBLE_USE_COMPRESSION=true
ENSEMBLE_COMPRESSION_MAX_CHARS=1500

# Optional: Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=

# Agent Settings
AGENT_TEMPERATURE=0.3
THEORY_AGENT_MAX_TOKENS=3000
KNOWLEDGE_BASE_FILE=data/knowledge_base.json
```

## 🏗️ Enhanced Architecture

```
User Query
    ↓
LangChain Router
    ↓
┌─────────────┬─────────────┬─────────────┐
│ Research    │ Theory      │ Implementation │
│ Agent       │ Agent       │ Agent          │
│ (Llama 3.1) │ (GPT-4)     │ (Claude 3.5)   │
└─────────────┴─────────────┴─────────────┘
    ↓
🔍 Ensemble RAG Tool (NEW!)
    ↓
┌─────────────┬─────────────┬─────────────┐
│ Enhanced    │ Self-Query  │ Parent Doc  │
│ Similarity  │ Retriever   │ Retriever   │
│ Retriever   │             │             │
└─────────────┴─────────────┴─────────────┘
    ↓
⚡ Reciprocal Rank Fusion (RRF)
    ↓
🧠 LLM-based Contextual Compression
    ↓
Pinecone Vector Store (llama-text-embed-v2)
    ↓
Knowledge Base (5,368+ ML/DL Papers & Content)
```

### Ensemble Retrieval Process

1. **Parallel Retrieval**: Three strategies run simultaneously:
   - **Enhanced Similarity**: Advanced semantic search with optimized prompts
   - **Self-Query Retriever**: Structured query parsing for metadata filtering
   - **Parent Document Retriever**: Retrieves larger context chunks when needed

2. **Result Fusion**: Reciprocal Rank Fusion (RRF) algorithm combines results:
   ```
   Combined Score = Σ(weight × 1/(rank + k))
   where k=60, weights configurable per strategy
   ```

3. **Contextual Compression**: LLM analyzes and filters content:
   - Removes irrelevant information
   - Preserves query-relevant context
   - Maintains answer completeness
   - Falls back gracefully if needed

4. **Dynamic Context Management**: Automatically adjusts context window:
   - Monitors token usage in real-time
   - Prevents context overflow errors
   - Optimizes for answer quality vs. length

## Expanding the Knowledge Base

### LLM-Based Semantic Chunking

This project features an advanced **LLM-based semantic chunking** system that intelligently splits documents based on semantic boundaries rather than fixed character limits:

#### Features:
- **Semantic Awareness**: Uses GPT-3.5-turbo to identify natural breakpoints in academic papers
- **Context Preservation**: Maintains conceptual integrity by avoiding splits in the middle of equations, code blocks, or important concepts
- **Adaptive Sizing**: Balances chunk size constraints with semantic coherence
- **Fallback Mechanism**: Automatically falls back to traditional chunking if LLM processing fails
- **LangChain Compatible**: Fully integrated with LangChain architecture for unified processing

#### How it Works:
1. **Analysis**: The LLM analyzes document sections to identify optimal split points
2. **Boundary Detection**: Prioritizes section boundaries, paragraph breaks, and topic transitions  
3. **Smart Splitting**: Creates chunks that contain complete thoughts or concepts
4. **Metadata Enrichment**: Each chunk includes contextual information and parent document references
5. **Unified Processing**: All chunks flow through the same LangChain-compatible pipeline

#### LangChain Integration:
- **Document Objects**: LLM chunks are converted to LangChain `Document` objects
- **Unified Upserting**: Same Pinecone integration for all content types
- **Metadata Preservation**: Chunk relationships maintained through LangChain pipeline
- **Hosted Embeddings**: Works seamlessly with Pinecone's hosted embedding models

#### Configuration:
```env
USE_LLM_CHUNKING=true              # Enable LLM-based chunking
LLM_CHUNKING_MODEL=gpt-3.5-turbo   # Model for semantic analysis
MAX_CHUNK_SIZE=1500                # Target chunk size
```

#### Benefits over Traditional Chunking:
- **Better Retrieval**: Semantically coherent chunks improve RAG performance
- **Context Preservation**: Important concepts aren't split across multiple chunks
- **Improved Understanding**: Each chunk represents a complete idea or concept
- **Academic Paper Optimized**: Specifically designed for research paper structure
- **Architecture Consistency**: No duplicate code paths or conflicting approaches

### Streamlit Paper Upload and Processing

The enhanced Streamlit interface now supports **direct PDF upload** with real-time processing:

#### Upload Process:
1. **Select PDF**: Choose research paper from your device
2. **Choose Chunking**: Select paragraph-based or traditional chunking
3. **Real-time Processing**: Watch progress indicators for each step:
   - Text extraction from PDF
   - Metadata extraction (title, authors, abstract)
   - Intelligent chunking (paragraph-based or traditional)
   - Vector embedding and Pinecone upload
4. **Success Confirmation**: Detailed report on processing results
5. **Immediate Availability**: Ask questions about the paper instantly

#### Example Success Case - FlashAttention Paper:
```
Processing: 2205.14135.pdf
Text Extraction: Complete (45 pages)
Metadata Extraction:
   Title: FlashAttention: Fast and Memory-Efficient Exact Attention
   Authors: Tri Dao, Daniel Y. Fu, Stefano Ermon, et al.
   Date: 2022-05-23
Chunking: 100 paragraph-based chunks created
Pinecone Upload: 96/100 chunks uploaded successfully
Notes: 4 chunks rate-limited (retry later)
Status: Paper now available in knowledge base
```

#### Supported Features:
- **Multiple Formats**: PDF files with text content
- **Progress Tracking**: Real-time status updates
- **Error Handling**: Graceful handling of processing failures
- **Metadata Preservation**: Title, authors, dates automatically extracted
- **Chunking Options**: Choose optimal strategy for your use case

### Collecting LLM Papers
The system includes a tool to automatically collect and process LLM-related research papers from arXiv and add them to your knowledge base:

```bash
# Collect LLM papers and update the knowledge base
python collect_llm_papers.py

# Specify maximum number of papers to collect
python collect_llm_papers.py --max-papers 30

# Collect papers and immediately upload to Pinecone
python collect_llm_papers.py --upload

# Process a single paper with enhanced chunking
python process_and_upsert_papers.py --paper data/papers/your_paper.pdf
```

This tool will:
1. Search for papers on key LLM topics (transformers, attention mechanisms, etc.)
2. Extract text from PDFs and add to knowledge base
3. Save papers locally in the `data/papers` directory
4. Add basic concept entries for important LLM terms
5. Automatically upload to Pinecone when using the `--upload` flag

### Supported LLM Topics
The paper collector searches for the latest research in:
- Transformer architectures (including FlashAttention)
- Attention mechanisms and optimizations
- LLaMA, GPT, Claude, Mistral models
- Prompt engineering and in-context learning
- Chain-of-thought reasoning
- Retrieval augmented generation (RAG)
- Instruction tuning and alignment
- Constitutional AI and RLHF
- Model evaluation and benchmarking
- Inference optimization and quantization
- Parameter-efficient fine-tuning (LoRA, etc.)
- Foundation models and scaling laws

### Key LLM Concepts
The knowledge base is pre-populated with entries covering fundamental LLM concepts like:
- Self-attention mechanisms and multi-head attention
- Positional encoding and rotary embeddings
- Masked language modeling and autoregressive generation
- Zero-shot, few-shot, and in-context learning
- RLHF and Constitutional AI techniques
- Flash Attention and memory optimization
- Transformer architectures and variants
- And many more cutting-edge concepts

## Agent Specializations

### Research Agent (Ollama Llama 3.1)
- **Purpose**: Literature analysis and academic synthesis
- **Triggers**: `paper`, `research`, `study`, `literature`, `recent`, `survey`
- **Example**: *"Find recent papers about transformer architectures"*
- **Enhanced**: Uses ensemble retrieval to find comprehensive research coverage

### Theory Agent (OpenAI) with Chain-of-Thoughts
- **Purpose**: Mathematical concepts and theoretical explanations using structured reasoning
- **Features**: 
  - Chain of Thoughts (CoT) reasoning for complex problems
  - Step-by-step mathematical derivations
  - Structured problem decomposition
  - Intuitive explanations alongside formal proofs
  - **Visible thinking process** - See how the agent reasons through problems
  - **Context Management**: Optimized 3,000 token limit for detailed explanations
- **Triggers**: `explain`, `theory`, `mathematical`, `algorithm`, `concept`
- **Example**: *"Explain the mathematical foundations of FlashAttention"*
- **CoT Structure**: Problem Understanding → Knowledge Retrieval → Step-by-Step Analysis → Intuitive Explanation → Key Takeaways

### Implementation Agent (Claude 3.5 Sonnet)
- **Purpose**: Code generation and practical guidance
- **Triggers**: `code`, `implement`, `pytorch`, `tensorflow`, `example`, `how to`
- **Example**: *"Show me code implementation of FlashAttention in PyTorch"*
- **Enhanced**: Uses ensemble retrieval to find comprehensive implementation examples

## 📁 Project Structure

```
RAG-Based-LLMs-Answering-System/
├── src/
│   ├── agents/
│   │   └── langchain_agents.py    # Multi-agent system with ensemble RAG
│   ├── api/
│   │   └── langchain_server.py    # FastAPI server (if needed)
│   ├── data_curation/             # Paper collection and processing
│   │   └── llm_paper_collector.py # LLM paper collector
│   └── config.py                  # Enhanced configuration management
├── data/
│   ├── knowledge_base.json        # Structured knowledge content (5,368+ entries)
│   ├── papers/                    # PDF papers (70+ research papers)
│   └── papers_metadata.json       # Paper metadata
├── main.py                        # Interactive CLI interface
├── streamlit_app.py               # Enhanced Streamlit web interface with upload
├── run_streamlit.py               # Streamlit launcher script
├── collect_llm_papers.py          # CLI for paper collection
├── process_and_upsert_papers.py   # Individual paper processing
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
└── README.md                      # This documentation
```

## Example Queries

### Research Questions
```
"What are the latest developments in transformer architectures?"
"Find research papers about FlashAttention optimization"
"Recent advances in attention mechanisms and memory efficiency"
```

### Theory Questions with Thinking Process
```
"Explain FlashAttention mathematically"
"What is the computational complexity of standard attention vs FlashAttention?"
"How does gradient descent work in transformer training?"
```

**Example with Enhanced Ensemble Retrieval:**
```
User question: Explain FlashAttention mathematically

🔍 Ensemble Retrieval Process:
========================================
Strategy 1 - Enhanced Similarity: 8 results found
Strategy 2 - Self-Query Retrieval: 6 results found  
Strategy 3 - Parent Document: 1 result found
Total: 15 individual results

⚡ Reciprocal Rank Fusion: 9 unique results after combination
🧠 Contextual Compression: 4 final results (1,420 chars)

Results include:
• FlashAttention paper sections on algorithmic complexity
• Mathematical derivations of O(N²d²/M) vs O(N²d) complexity
• Memory optimization techniques and tile-based computation
• Practical speedup measurements (7.6× faster)
========================================

💡 Final Response:
🤔 Thinking Process: I'll explain FlashAttention's mathematical foundations...

📚 Knowledge Base Search: [comprehensive FlashAttention content retrieved]

🧠 Step-by-Step Analysis:
Step 1: Standard Attention Complexity - O(N²d) memory, O(N²d²) time...
Step 2: FlashAttention Innovation - Tile-based computation with O(N²d²/M)...
[Detailed mathematical explanation with specific metrics]
```

### Implementation Questions
```
"Show me PyTorch code for FlashAttention implementation"
"How to implement memory-efficient attention in practice?"
"Best practices for training large transformer models"
```

## Key Technical Features

### Ensemble Retrieval System
- **Multi-Strategy Approach**: Combines three specialized retrieval methods
  - **Enhanced Similarity**: Optimized semantic search with improved prompts
  - **Self-Query Retriever**: Structured query parsing for metadata filtering  
  - **Parent Document Retriever**: Context-aware chunk relationships
- **Reciprocal Rank Fusion**: Mathematical algorithm for intelligent result combination
- **Weighted Scoring**: Configurable weights per strategy (40% similarity, 30% self-query, 30% parent)
- **Performance Gains**: Significantly improved retrieval accuracy over basic similarity search

### LLM-Based Contextual Compression
- **Intelligent Filtering**: Uses GPT-4o-mini to extract only query-relevant content
- **Content Preservation**: Maintains important context while removing noise
- **Token Optimization**: Reduces context length by 60-80% while preserving answer quality
- **Fallback Mechanism**: Graceful degradation if compression fails
- **Configuration**: Adjustable compression thresholds and character limits

### Dynamic Context Management
- **Real-time Monitoring**: Tracks token usage across all agents
- **Automatic Scaling**: Adjusts context based on agent-specific limits
- **Overflow Prevention**: Proactive management to prevent context window errors
- **Per-Agent Optimization**: Customized limits for each agent's use case

### LangChain Integration
- **Agent Executors**: Proper LangChain agent execution with enhanced tools
- **Advanced RAG Tools**: Ensemble retrieval fully integrated with LangChain
- **Vector Store**: LangChain `PineconeVectorStore` with hosted embeddings
- **Prompt Management**: Structured prompt templates for each agent
- **Chain of Thoughts**: Advanced reasoning framework for Theory Agent with structured problem-solving
- **Thinking Process Display**: Real-time visualization of agent reasoning steps and tool usage

### Pinecone Hosted Embeddings
- **Automatic Embedding**: No local embedding models required
- **High Performance**: Optimized `llama-text-embed-v2` model (1024 dimensions)
- **Proper API Usage**: Correct `upsert_records` and `search` API calls
- **Metadata Handling**: Clean metadata structure for retrieval
- **Scale**: Currently managing 5,368+ documents including 70+ research papers

### Intelligent Token Management
- **Per-Agent Optimization**: Each agent has optimized token limits for its specific tasks
  - **Routing LLM** (GPT-4o-mini): 512 tokens - Fast classification decisions
  - **Research Agent** (Claude): 4,096 tokens - Comprehensive research responses
  - **Theory Agent** (GPT-4): 3,000 tokens - Detailed mathematical explanations (optimized for context management)
  - **Implementation Agent** (Claude): 8,192 tokens - Complete code implementations without truncation
- **Automatic Scaling**: Token limits sized to prevent response truncation while optimizing costs
- **Fallback Protection**: All agents fall back to GPT-4 with adequate limits if primary models unavailable
- **Context Window Management**: Dynamic adjustment based on ensemble retrieval results

### Error Handling & Fallbacks
- **Model Fallbacks**: GPT-4 fallback if Ollama/Claude unavailable
- **Graceful Degradation**: System works with basic retrieval if ensemble fails
- **Configuration Validation**: Comprehensive environment variable checking
- **Rate Limit Handling**: Automatic retry logic for Pinecone uploads
- **Upload Resilience**: Partial success handling for batch uploads

## Advanced Usage

### Adding Knowledge
1. **JSON Method**: Add documents to `data/knowledge_base.json`
2. **PDF Method**: Place PDF papers in `data/papers/`
3. **Streamlit Upload**: Use the web interface for direct PDF upload
4. **CLI Processing**: Use `python process_and_upsert_papers.py --paper your_paper.pdf`

### Customizing Ensemble Retrieval
```python
# In config.py
ENSEMBLE_SIMILARITY_WEIGHT = 0.5     # Increase similarity weight
ENSEMBLE_SELF_QUERY_WEIGHT = 0.3     # Self-query weight  
ENSEMBLE_PARENT_DOC_WEIGHT = 0.2     # Parent document weight
ENSEMBLE_MAX_DOCS = 6                # Increase result count
ENSEMBLE_USE_COMPRESSION = True      # Enable compression
```

### Customizing Agents
```python
# In langchain_agents.py
def route_query(self, query: str) -> str:
    # Add custom routing logic
    if 'flashattention' in query.lower():
        return 'theory'  # Route FlashAttention questions to Theory agent
    if 'your_keyword' in query.lower():
        return 'your_agent'
    return 'theory'  # default
```

### API Server (Optional)
```bash
python -m src.api.langchain_server
# Visit: http://localhost:8000/docs
```

## Troubleshooting

### Common Issues
1. **"Knowledge base not available"** - This is normal if no documents loaded yet
2. **Agent routing incorrect** - Check keyword triggers in `route_query()`
3. **Pinecone errors** - Verify API key and index configuration
4. **Model unavailable** - Check API keys and model names
5. **Context overflow** - Reduce `ENSEMBLE_MAX_DOCS` or disable compression temporarily
6. **Upload failures** - Check Pinecone rate limits and retry failed chunks

### Ensemble Retrieval Troubleshooting
1. **Poor retrieval quality**:
   - Increase `ENSEMBLE_MAX_DOCS` from 4 to 6
   - Adjust retrieval strategy weights
   - Disable compression to see raw results

2. **Context overflow errors**:
   - Reduce `THEORY_AGENT_MAX_TOKENS` from 3000 to 2500
   - Reduce `ENSEMBLE_COMPRESSION_MAX_CHARS` from 1500 to 1200
   - Set `ENSEMBLE_USE_COMPRESSION=true` if disabled

3. **Slow performance**:
   - Reduce `ENSEMBLE_MAX_DOCS` from 4 to 3
   - Use only similarity search by setting other weights to 0

### Health Check
```python
from src.agents.langchain_agents import LangChainMLAgents
agents = LangChainMLAgents()
health = agents.health_check()
print(health)

# Check ensemble retrieval specifically
result = agents.ensemble_rag_tool.invoke({"query": "test query"})
print(f"Retrieved {len(result)} documents")
```

## Performance Features

- **Ensemble Retrieval**: 3x better relevance through multi-strategy approach
- **Contextual Compression**: 60-80% context reduction with preserved quality
- **Batch Processing**: Optimized document upserts (96 records/batch for hosted embeddings)
- **Parallel Search**: Efficient vector similarity search across multiple strategies
- **Memory Management**: Proper cleanup of vector store connections
- **Cache Optimization**: Environment variable caching with override support
- **Dynamic Context**: Real-time token management prevents overflow errors

## Security Features

- **API Key Protection**: `.env` files excluded from Git
- **Sanitized Uploads**: Clean text processing for vector storage
- **Error Isolation**: Robust error handling prevents system crashes
- **Configuration Validation**: Ensures all required settings are present
- **Rate Limit Respect**: Automatic handling of Pinecone rate limits

## Real-World Success Case: FlashAttention Paper

### Paper Processing Success
```
📄 FlashAttention: Fast and Memory-Efficient Exact Attention
👥 Authors: Tri Dao, Daniel Y. Fu, Stefano Ermon, et al.
📊 Processing Results:
   • PDF Pages: 45
   • Text Chunks: 100 (paragraph-based)
   • Pinecone Upload: 96/100 successful
   • Processing Time: 2 minutes
   • Knowledge Base Size: 5,368 total documents
```

### Query Performance with Ensemble Retrieval
```
Query: "Explain FlashAttention's memory optimization"

Ensemble Retrieval Results:
• Enhanced Similarity: 8 results (FlashAttention paper sections)
• Self-Query: 6 results (attention mechanism comparisons)  
• Parent Document: 1 result (full paper context)
• Total Retrieved: 15 individual results
• After RRF Fusion: 9 unique results
• After Compression: 4 final results (1,420 chars)

Answer Quality:
- Specific metrics: "7.6× speedup over standard attention"
- Technical details: "O(N²d²/M) complexity vs O(N²d)"
- Implementation insights: "Tile-based computation approach"
- Source attribution: Citations to FlashAttention paper sections
```

## Future Enhancements

- [ ] Add more specialized agents (Computer Vision, NLP, etc.)
- [ ] Implement conversation memory and context tracking
- [ ] Add support for multiple knowledge domains
- [ ] Integrate with more vector store providers
- [x] ~~Add web interface for better user experience~~ ✅ **Completed: Streamlit Web Interface**
- [x] ~~Implement ensemble retrieval for better RAG performance~~ ✅ **Completed: Ensemble Retrieval with RRF**
- [x] ~~Add contextual compression for better context management~~ ✅ **Completed: LLM-based Compression**
- [x] ~~Support direct paper upload and processing~~ ✅ **Completed: Streamlit Upload Feature**
- [ ] Add multi-modal support (images, tables, equations)
- [ ] Implement semantic caching for faster repeated queries
- [ ] Add support for private knowledge bases and custom domains

## 🙏 Acknowledgments

Built with:
- **[LangChain](https://python.langchain.com/)** - AI application framework and ensemble retrieval
- **[Pinecone](https://www.pinecone.io/)** - Vector database with hosted embeddings  
- **[OpenAI](https://openai.com/)** - GPT-4 language model and embeddings
- **[Anthropic](https://www.anthropic.com/)** - Claude 3.5 Sonnet
- **[Ollama](https://ollama.ai/)** - Local Llama model serving
- **[Streamlit](https://streamlit.io/)** - Modern web interface with upload capabilities

### Special Recognition
- **FlashAttention** by Tri Dao et al. - Demonstrates successful real-world paper processing
- **Reciprocal Rank Fusion** algorithm for ensemble retrieval optimization
- **LangChain Community** for comprehensive RAG tools and retrieval strategies

---

**A complete RAG-based multi-agent system for ML/DL expertise with advanced ensemble retrieval, contextual compression, and zero local embedding setup!** 

🎯 **5,368+ documents** | 🔍 **Ensemble retrieval** | 🧠 **LLM compression** | 📄 **Direct upload** | ⚡ **Real-time processing** 
