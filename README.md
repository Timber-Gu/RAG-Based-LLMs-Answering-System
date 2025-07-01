# Multi-Model LangChain ML Q&A Assistant with Pinecone Hosted Embeddings

A sophisticated **RAG-based multi-agent system** for Machine Learning and Deep Learning Q&A, powered by different specialized LLMs and **Pinecone's hosted embeddings**:

- **Research Agent**: Ollama (Llama 3.1) - For literature reviews and research findings
- **Theory Agent**: GPT-4 - For mathematical explanations and theory
- **Implementation Agent**: Claude 3.5 Sonnet - For code generation and practical examples
- **Knowledge Base**: Pinecone with hosted `llama-text-embed-v2` - For semantic search

## 🚀 Key Features

### 🎯 LangChain-First Architecture
- **Complete LangChain Integration**: Uses LangChain agents, tools, and vector stores
- **Multi-Agent System**: Intelligent routing to specialized agents
- **RAG Integration**: Seamless knowledge retrieval using LangChain tools
- **Chat History Support**: Conversation context management

### 🌟 Pinecone Hosted Embeddings
- **No Local Setup Required**: Uses Pinecone's hosted `llama-text-embed-v2` model
- **High Performance**: 1024-dimensional vectors with optimized search
- **Automatic Embedding**: Text is automatically embedded during upsert and search
- **Scalable**: Cloud-native vector storage and retrieval

### 🤖 Specialized Agents
- **Smart Routing**: Automatic agent selection based on query type
- **Domain Expertise**: Each agent optimized for specific ML/DL tasks
- **Fallback Handling**: Graceful degradation if specific models unavailable

## 🛠️ Quick Setup

### Prerequisites
- **OpenAI API Key** (for GPT-4 Theory Agent)
- **Anthropic API Key** (for Claude Implementation Agent)  
- **Pinecone API Key** (for hosted embeddings and vector storage)
- **Ollama** (optional, for Research Agent)

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

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

4. **Run the System**
   ```bash
   python main.py
   ```

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

## 🤖 Agent Specializations

### 🔬 Research Agent (Ollama Llama 3.1)
- **Purpose**: Literature analysis and academic synthesis
- **Triggers**: `paper`, `research`, `study`, `literature`, `recent`, `survey`
- **Example**: *"Find recent papers about transformer architectures"*

### 📐 Theory Agent (GPT-4)
- **Purpose**: Mathematical concepts and theoretical explanations
- **Triggers**: `explain`, `theory`, `mathematical`, `algorithm`, `concept`
- **Example**: *"Explain the mathematical foundations of neural networks"*

### 💻 Implementation Agent (Claude 3.5 Sonnet)
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
│   └── config.py                  # Configuration management
├── data/
│   ├── knowledge_base.json        # Structured knowledge content
│   ├── papers/                    # PDF papers (optional)
│   └── papers_metadata.json       # Paper metadata
├── main.py                        # Interactive CLI interface
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

### Theory Questions  
```
"Explain backpropagation mathematically"
"What is the mathematical foundation of neural networks?"
"How does gradient descent work in deep learning?"
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

### Pinecone Hosted Embeddings
- **Automatic Embedding**: No local embedding models required
- **High Performance**: Optimized `llama-text-embed-v2` model (1024 dimensions)
- **Proper API Usage**: Correct `upsert_records` and `search` API calls
- **Metadata Handling**: Clean metadata structure for retrieval

### Error Handling & Fallbacks
- **Model Fallbacks**: GPT-4 fallback if Ollama/Claude unavailable
- **Graceful Degradation**: System works even without RAG knowledge base
- **Configuration Validation**: Comprehensive environment variable checking

## 🚀 Advanced Usage

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

## 🔍 Troubleshooting

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

## 📊 Performance Features

- **Batch Processing**: Optimized document upserts (96 records/batch for hosted embeddings)
- **Parallel Search**: Efficient vector similarity search
- **Memory Management**: Proper cleanup of vector store connections
- **Cache Optimization**: Environment variable caching with override support

## 🛡️ Security Features

- **API Key Protection**: `.env` files excluded from Git
- **Sanitized Uploads**: Clean text processing for vector storage
- **Error Isolation**: Robust error handling prevents system crashes
- **Configuration Validation**: Ensures all required settings are present

## 📈 Future Enhancements

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

**🎯 A complete RAG-based multi-agent system for ML/DL expertise with zero local embedding setup!** 