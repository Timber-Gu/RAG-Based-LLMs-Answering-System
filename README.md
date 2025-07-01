# Multi-Model LangChain ML Q&A Assistant with Pinecone Hosted Embeddings

A **RAG-based multi-agent system** for Machine Learning and Deep Learning Q&A, powered by different specialized LLMs and **Pinecone's hosted embeddings**:

- **Research Agent**: Ollama (Llama 3.1) - For literature reviews and research findings
- **Theory Agent**: GPT-4 - For mathematical explanations and theory
- **Implementation Agent**: Claude 3.5 Sonnet - For code generation and practical examples
- **Knowledge Base**: Pinecone with hosted `llama-text-embed-v2` - For semantic search

## 🚀 Key Features

### LangChain-First Architecture
- **Complete LangChain Integration**: Uses LangChain agents, tools, and vector stores
- **Multi-Agent System**: Intelligent routing to specialized agents
- **RAG Integration**: Seamless knowledge retrieval using LangChain tools
- **Chat History Support**: Conversation context management

### Pinecone Hosted Embeddings
- **No Local Setup Required**: Uses Pinecone's hosted `llama-text-embed-v2` model
- **High Performance**: 1024-dimensional vectors with optimized search
- **Automatic Embedding**: Text is automatically embedded during upsert and search
- **Scalable**: Cloud-native vector storage and retrieval

### Specialized Agents
- **Smart Routing**: Automatic agent selection based on query type
- **Domain Expertise**: Each agent optimized for specific ML/DL tasks
- **Fallback Handling**: Graceful degradation if specific models unavailable

## 🛠️ Quick Setup

### Prerequisites
- **OpenAI API Key** (for GPT-4 Theory Agent) - Get it from [OpenAI Platform](https://platform.openai.com/account/api-keys)
- **Anthropic API Key** (for Claude Implementation Agent) - Get it from [Anthropic Console](https://console.anthropic.com/)
- **Pinecone API Key** and **Environment** - Get them from [Pinecone Console](https://app.pinecone.io/)
- **Ollama** (optional, for Research Agent) - [Installation Guide](https://ollama.ai/download)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Timber-Gu/RAG-Based-LLMs-Answering-System.git
   cd RAG-Based-LLMs-Answering-System
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**
   
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
   EMBEDDING_MODEL=llama-text-embed-v2
   
   # Model Settings (Optional - these are default values)
   AGENT_TEMPERATURE=0.7
   AGENT_MAX_TOKENS=1000
   
   # Vector Store Settings (Optional - these are default values)
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

4. **Pinecone Setup**
   - Create a free account at [Pinecone](https://app.pinecone.io/)
   - Create a new index with the following settings:
     - Dimensions: 1024 (required for llama-text-embed-v2)
     - Metric: cosine
     - Pod Type: starter (free tier)
   - Copy your API key and environment from the Pinecone console
   - Update your `.env` file with these values

5. **Run the System**
   ```bash
   python main.py
   ```

### Interactive Commands

Once the system is running, you can use these commands:
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

#### Optional Variables
- `AGENT_TEMPERATURE`: Controls response creativity (0.0-1.0, default: 0.7)
- `AGENT_MAX_TOKENS`: Maximum response length (default: 1000)
- `CHUNK_SIZE`: Size of text chunks for vector storage (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `USE_LLM_CHUNKING`: Enable LLM-based semantic chunking (default: true)
- `LLM_CHUNKING_MODEL`: Model to use for semantic chunking (default: gpt-3.5-turbo)
- `MAX_CHUNK_SIZE`: Maximum size for LLM-based chunks (default: 1500)
- `OLLAMA_BASE_URL`: URL for local Ollama instance (default: http://127.0.0.1:11434)
- `RESEARCH_MODEL`: Ollama model to use (default: llama3.1)

### Troubleshooting Environment Setup

1. **API Key Issues**
   - Ensure all API keys are correctly copied without extra spaces
   - Verify API keys are active and have sufficient credits
   - Check for any special characters that might need escaping

2. **Pinecone Issues**
   - Verify your index is created with correct dimensions (1024)
   - Ensure index name matches exactly what's in your `.env`
   - Check if you're using the correct environment

3. **Ollama Issues**
   - Verify Ollama is running locally (`ollama run llama3.1`)
   - Check if the model is downloaded (`ollama list`)
   - Ensure the correct base URL is set

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
EMBEDDING_MODEL=llama-text-embed-v2

# Model Configuration
THEORY_MODEL=gpt-4
RESEARCH_MODEL=llama3.1
IMPLEMENTATION_MODEL=claude-3-5-sonnet-20241022

# Optional: Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=

# Agent Settings
AGENT_TEMPERATURE=0.3
KNOWLEDGE_BASE_FILE=data/knowledge_base.json
```

## 🏗️ Architecture

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
RAG Tool (LangChain + Pinecone)
    ↓
Pinecone Vector Store (Hosted llama-text-embed-v2)
    ↓
Knowledge Base (ML/DL Papers & Content)
```

## 📚 Expanding the Knowledge Base

### 🧠 LLM-Based Semantic Chunking

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

### Collecting LLM Papers
The system includes a tool to automatically collect and process LLM-related research papers from arXiv and add them to your knowledge base:

```bash
# Collect LLM papers and update the knowledge base
python collect_llm_papers.py

# Specify maximum number of papers to collect
python collect_llm_papers.py --max-papers 30

# Collect papers and immediately upload to Pinecone
python collect_llm_papers.py --upload

# Or upload existing knowledge base to Pinecone separately
python upload_to_pinecone.py

# Upload with options
python upload_to_pinecone.py --clear  # Clear existing vectors first
python upload_to_pinecone.py --force  # Force upload even if vectors exist

# Test unified LangChain-compatible chunking
python test_unified_chunking.py
```

This tool will:
1. Search for papers on key LLM topics (transformers, attention mechanisms, etc.)
2. Extract text from PDFs and add to knowledge base
3. Save papers locally in the `data/papers` directory
4. Add basic concept entries for important LLM terms
5. Automatically upload to Pinecone when using the `--upload` flag

### Supported LLM Topics
The paper collector searches for the latest research in:
- Transformer architectures
- Attention mechanisms
- LLaMA, GPT, Claude, Mistral models
- Prompt engineering
- In-context learning
- Chain-of-thought
- Retrieval augmented generation
- Instruction tuning
- Constitutional AI
- Alignment techniques
- Model evaluation
- Inference optimization
- Quantization methods
- Parameter-efficient fine-tuning
- Foundation models

### Key LLM Concepts
The knowledge base is pre-populated with entries covering fundamental LLM concepts like:
- Self-attention mechanisms
- Multi-head attention
- Positional encoding
- Masked language modeling
- Zero-shot and few-shot learning
- RLHF and Constitutional AI
- Flash Attention
- And many more

## 🤖 Agent Specializations

### Research Agent (Ollama Llama 3.1)
- **Purpose**: Literature analysis and academic synthesis
- **Triggers**: `paper`, `research`, `study`, `literature`, `recent`, `survey`
- **Example**: *"Find recent papers about transformer architectures"*

### 📐 Theory Agent (GPT-4) with Chain of Thoughts
- **Purpose**: Mathematical concepts and theoretical explanations using structured reasoning
- **Features**: 
  - Chain of Thoughts (CoT) reasoning for complex problems
  - Step-by-step mathematical derivations
  - Structured problem decomposition
  - Intuitive explanations alongside formal proofs
  - **Visible thinking process** - See how the agent reasons through problems
- **Triggers**: `explain`, `theory`, `mathematical`, `algorithm`, `concept`
- **Example**: *"Explain the mathematical foundations of neural networks"*
- **CoT Structure**: Problem Understanding → Knowledge Retrieval → Step-by-Step Analysis → Intuitive Explanation → Key Takeaways

### Implementation Agent (Claude 3.5 Sonnet)
- **Purpose**: Code generation and practical guidance
- **Triggers**: `code`, `implement`, `pytorch`, `tensorflow`, `example`, `how to`
- **Example**: *"Show me code implementation of a transformer architecture"*

## 📁 Project Structure

```
RAG-Based-LLMs-Answering-System/
├── src/
│   ├── agents/
│   │   └── langchain_agents.py    # Multi-agent system with RAG
│   ├── api/
│   │   └── langchain_server.py    # FastAPI server (if needed)
│   ├── data_curation/             # Paper collection and processing
│   │   └── llm_paper_collector.py # LLM paper collector
│   └── config.py                  # Configuration management
├── data/
│   ├── knowledge_base.json        # Structured knowledge content
│   ├── papers/                    # PDF papers (optional)
│   └── papers_metadata.json       # Paper metadata
├── main.py                        # Interactive CLI interface
├── collect_llm_papers.py          # CLI for paper collection
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
└── README.md                      # This documentation
```

## 🧪 Example Queries

### Research Questions
```
"What are the latest developments in transformer architectures?"
"Find research papers about attention mechanisms"
"Recent advances in computer vision models"
```

### Theory Questions with Thinking Process
```
"Explain backpropagation mathematically"
"What is the mathematical foundation of neural networks?"
"How does gradient descent work in deep learning?"
```

**Example with Thinking Process Display:**
```
🤔 Your question: Explain gradient descent mathematically

🧠 Agent Thinking Process:
========================================

Step 1: 🔍 Searching knowledge base for: gradient descent mathematical
   Result: Found relevant mathematical content about optimization...

Step 2: 🧠 Applying Chain of Thoughts reasoning to: gradient descent
   Result: Generated structured reasoning framework

========================================

💡 Final Response:
🤔 Thinking Process: I'll explain gradient descent using a structured mathematical approach...

📚 Knowledge Base Search: [searches for relevant mathematical content]

🧠 Step-by-Step Analysis:
Step 1: Problem Understanding - Gradient descent is an optimization algorithm...
[Detailed mathematical explanation follows]
```

### Implementation Questions
```
"Show me PyTorch code for a transformer model"
"How to implement attention mechanism in Python?"
"Best practices for training deep learning models"
```

## 🔧 Key Technical Features

### LangChain Integration
- **Agent Executors**: Proper LangChain agent execution with tools
- **RAG Tools**: LangChain-compatible knowledge retrieval tools
- **Vector Store**: LangChain `PineconeVectorStore` with hosted embeddings
- **Prompt Management**: Structured prompt templates for each agent
- **Chain of Thoughts**: Advanced reasoning framework for Theory Agent with structured problem-solving
- **Thinking Process Display**: Real-time visualization of agent reasoning steps and tool usage

### Pinecone Hosted Embeddings
- **Automatic Embedding**: No local embedding models required
- **High Performance**: Optimized `llama-text-embed-v2` model (1024 dimensions)
- **Proper API Usage**: Correct `upsert_records` and `search` API calls
- **Metadata Handling**: Clean metadata structure for retrieval

### Error Handling & Fallbacks
- **Model Fallbacks**: GPT-4 fallback if Ollama/Claude unavailable
- **Graceful Degradation**: System works even without RAG knowledge base
- **Configuration Validation**: Comprehensive environment variable checking

## Advanced Usage

### Adding Knowledge
1. Add documents to `data/knowledge_base.json`
2. Or place PDF papers in `data/papers/`
3. Restart system to reload vector store

### Customizing Agents
```python
# In langchain_agents.py
def route_query(self, query: str) -> str:
    # Add custom routing logic
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

### Health Check
```python
from src.agents.langchain_agents import LangChainMLAgents
agents = LangChainMLAgents()
health = agents.health_check()
print(health)
```

## Performance Features

- **Batch Processing**: Optimized document upserts (96 records/batch for hosted embeddings)
- **Parallel Search**: Efficient vector similarity search
- **Memory Management**: Proper cleanup of vector store connections
- **Cache Optimization**: Environment variable caching with override support

## Security Features

- **API Key Protection**: `.env` files excluded from Git
- **Sanitized Uploads**: Clean text processing for vector storage
- **Error Isolation**: Robust error handling prevents system crashes
- **Configuration Validation**: Ensures all required settings are present

## Future Enhancements

- [ ] Add more specialized agents (Computer Vision, NLP, etc.)
- [ ] Implement conversation memory and context tracking
- [ ] Add support for multiple knowledge domains
- [ ] Integrate with more vector store providers
- [ ] Add web interface for better user experience

## 🙏 Acknowledgments

Built with:
- **[LangChain](https://python.langchain.com/)** - AI application framework
- **[Pinecone](https://www.pinecone.io/)** - Vector database with hosted embeddings  
- **[OpenAI](https://openai.com/)** - GPT-4 language model
- **[Anthropic](https://www.anthropic.com/)** - Claude 3.5 Sonnet
- **[Ollama](https://ollama.ai/)** - Local Llama model serving

---

** A complete RAG-based multi-agent system for ML/DL expertise with zero local embedding setup!** 
