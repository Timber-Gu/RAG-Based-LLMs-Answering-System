# ML Q&A Assistant - Phase 1 Implementation

Advanced ML Q&A system with Multi-Agent System and RAG (Retrieval-Augmented Generation) for Deep Learning and Neural Networks.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# You need: OPENAI_API_KEY and PINECONE_API_KEY
```

### 3. Run Setup Script
```bash
python scripts/setup_phase1.py
```

### 4. Start the System
```bash
# Terminal 1: Start API server
python -m src.api.server

# Terminal 2: Start Streamlit UI
streamlit run src/ui/streamlit_app.py
```

## 📋 Phase 1 Features

### ✅ Completed
- **Data Curation**: Automated collection of 100+ Deep Learning papers from arXiv
- **Content Extraction**: PDF processing and knowledge base creation
- **RAG Pipeline**: Pinecone vector store with semantic search
- **Multi-Agent System**: 3 specialized agents (Research, Theory, Implementation)
- **Agent Orchestration**: Smart routing and response combination
- **Web Interface**: FastAPI backend + Streamlit frontend
- **Knowledge Base**: Processed papers with embeddings

### 🔧 Components

#### 1. Data Curation (`src/data_curation/`)
- `paper_collector.py`: Collects papers from arXiv API
- `content_extractor.py`: Processes PDFs and creates knowledge chunks

#### 2. RAG System (`src/rag/`)
- `vector_store.py`: Pinecone integration with semantic search

#### 3. Multi-Agent System (`src/agents/`)
- `base_agent.py`: Base agent class with OpenAI integration
- `research_agent.py`: Literature analysis and synthesis
- `theory_agent.py`: Mathematical explanations and derivations
- `implementation_agent.py`: Code generation and practical guidance
- `orchestrator.py`: Agent coordination and response combination

#### 4. API & UI (`src/api/`, `src/ui/`)
- `server.py`: FastAPI backend with REST endpoints
- `streamlit_app.py`: Interactive web interface

## 🎯 Usage Examples

### Research Questions
- "What papers discuss attention mechanisms?"
- "Recent work on transformers"
- "State of the art in computer vision"

### Theoretical Questions
- "Explain backpropagation mathematically"
- "How does gradient descent work?"
- "Derive the loss function for neural networks"

### Implementation Questions
- "How to implement a CNN in PyTorch?"
- "Show me GAN code examples"
- "Build a transformer from scratch"

## 🔧 Configuration

Key settings in `src/config.py`:
- `MAX_PAPERS`: Number of papers to collect (default: 100)
- `EMBEDDING_MODEL`: Sentence transformer model
- `PINECONE_INDEX_NAME`: Vector database index name
- `CHUNK_SIZE`: Text chunk size for embeddings

## 📊 System Architecture

```
Web Interface (Streamlit)
    ↓
API Server (FastAPI)
    ↓
Agent Orchestrator
    ↓
┌─────────────┬─────────────┬─────────────┐
│ Research    │ Theory      │ Implementation │
│ Agent       │ Agent       │ Agent          │
└─────────────┴─────────────┴─────────────┘
    ↓
RAG Pipeline (Pinecone + Sentence Transformers)
    ↓
Knowledge Base (Processed Papers)
```

## 🛠️ Development

### Project Structure
```
Project/
├── src/
│   ├── agents/          # Multi-agent system
│   ├── api/             # FastAPI server
│   ├── data_curation/   # Paper collection and processing
│   ├── rag/             # RAG pipeline
│   ├── ui/              # Streamlit interface
│   └── config.py        # Configuration
├── data/                # Data files (papers, knowledge base)
├── scripts/             # Setup and utility scripts
├── requirements.txt     # Dependencies
└── README.md           # This file
```

### Running Tests
```bash
# Test paper collection
python -m src.data_curation.paper_collector

# Test content extraction
python -m src.data_curation.content_extractor

# Test RAG pipeline
python -m src.rag.vector_store

# Test agent orchestrator
python -m src.agents.orchestrator
```

## 🚀 Next Steps (Phase 2)

- [ ] Supervised Fine-Tuning (SFT) of specialized models
- [ ] Expanded dataset with 1500+ Q&A pairs
- [ ] Enhanced web interface with conversation history
- [ ] Performance optimization and caching
- [ ] Advanced prompt engineering techniques

## 📝 Notes

- Start with 20 papers for testing (configurable)
- RAG system requires Pinecone free tier account
- OpenAI API usage is optimized for cost efficiency
- All components are modular and extensible

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure `.env` file has valid API keys
2. **Pinecone Connection**: Check environment and API key
3. **PDF Processing**: Some PDFs may fail to extract text
4. **Memory Usage**: Large knowledge bases may require more RAM

### Getting Help

Check the logs in the console output for detailed error messages. Each component has comprehensive error handling and logging.

---

Built with ❤️ for the ML community 