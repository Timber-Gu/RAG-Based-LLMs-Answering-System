# LangChain ML Q&A Assistant

A clean, well-organized multi-agent system for Machine Learning Q&A using **LangChain framework**. This project demonstrates proper use of LangChain agents with RAG capabilities for ML/DL domain expertise.

## 🎯 What This Project Does

- **Multi-Agent System**: 3 specialized LangChain agents (Research, Theory, Implementation)
- **RAG Integration**: ChromaDB vector store for knowledge retrieval
- **Clean Architecture**: Simple, maintainable code using LangChain best practices
- **No Over-Engineering**: Focused implementation without unnecessary complexity

## 🚀 Quick Start

### 1. Setup
```bash
# Clone or download the project
# Run the setup script
python setup_langchain.py
```

### 2. Configure API Key
Edit `.env` file with your OpenAI API key:
```env
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Run the System

**Interactive Mode:**
```bash
python main.py
```

**API Server:**
```bash
python -m src.api.langchain_server
```
Then visit: http://localhost:8000/docs

## 🏗️ Architecture

```
User Query
    ↓
LangChain Router
    ↓
┌─────────────┬─────────────┬─────────────┐
│ Research    │ Theory      │ Implementation │
│ Agent       │ Agent       │ Agent          │
│ (LangChain) │ (LangChain) │ (LangChain)    │
└─────────────┴─────────────┴─────────────┘
    ↓
RAG Tool (ChromaDB)
    ↓
Knowledge Base
```

## 🤖 Agents

### Research Agent
- **Purpose**: Literature analysis and academic synthesis
- **Keywords**: paper, research, study, literature, recent
- **Example**: "What papers discuss attention mechanisms?"

### Theory Agent  
- **Purpose**: Mathematical concepts and explanations
- **Keywords**: explain, what is, theory, mathematical, algorithm
- **Example**: "Explain backpropagation mathematically"

### Implementation Agent
- **Purpose**: Code generation and practical guidance  
- **Keywords**: code, implement, pytorch, tensorflow, example
- **Example**: "How to implement a CNN in PyTorch?"

## 📁 Project Structure

```
Project/
├── src/
│   ├── agents/
│   │   └── langchain_agents.py    # LangChain multi-agent system
│   ├── api/
│   │   └── langchain_server.py    # FastAPI server
│   └── config.py                  # Configuration
├── data/                          # Knowledge base and papers
├── main.py                        # Interactive interface
├── setup_langchain.py            # Setup script
├── requirements.txt               # Dependencies
└── README_LangChain.md           # This file
```

## 🔧 Key Features

### Proper LangChain Usage
- **LangChain Agents**: Using `create_openai_functions_agent`
- **Agent Executors**: Proper agent execution with tools
- **RAG Tool**: LangChain tool for knowledge retrieval
- **Chat History**: Support for conversation context

### Simple Dependencies
- **No PyTorch conflicts**: Uses ChromaDB instead of Pinecone
- **Clean requirements**: Only essential packages
- **Fast startup**: Minimal initialization time

### Well-Organized Code
- **Single responsibility**: Each component has clear purpose
- **Easy to extend**: Add new agents or tools easily
- **Type hints**: Full typing for better development
- **Error handling**: Robust error handling throughout

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Test Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are neural networks?"}'
```

### Available Agents
```bash
curl http://localhost:8000/agents
```

## 📝 Example Queries

### Research Questions
- "Recent papers on transformer architectures"
- "State of the art in computer vision"
- "What research exists on attention mechanisms?"

### Theory Questions  
- "Explain gradient descent mathematically"
- "How does backpropagation work?"
- "What is a convolutional neural network?"

### Implementation Questions
- "How to implement a neural network in PyTorch?"
- "Show me a CNN code example"
- "Best practices for training deep learning models"

## 🛠️ Customization

### Add New Agent
```python
# In langchain_agents.py
def _setup_agents(self):
    # Add new agent
    new_prompt = ChatPromptTemplate.from_messages([...])
    new_agent = create_openai_functions_agent(...)
    self.agents['new_agent'] = AgentExecutor(...)
```

### Modify Routing Logic
```python
def route_query(self, query: str) -> str:
    # Add custom routing logic
    if 'your_keyword' in query.lower():
        return 'your_agent'
    return 'theory'  # default
```

### Add New Tools
```python
def _create_new_tool(self):
    def tool_function(input: str) -> str:
        # Your tool logic here
        return result
    
    return Tool(
        name="tool_name",
        description="Tool description",
        func=tool_function
    )
```

## 🔍 Troubleshooting

### Common Issues

1. **"Agents not initialized"**
   - Check your OpenAI API key in `.env`
   - Ensure you have internet connection

2. **"Knowledge base not available"**
   - This is normal if you haven't added documents yet
   - Agents will work without knowledge base

3. **Import errors**
   - Run `python setup_langchain.py` to install dependencies
   - Make sure you're in the project root directory

### Getting Help
- Check the health endpoint: `/health`
- Run with verbose mode for detailed logs
- Verify `.env` file configuration

## 🎯 Next Steps

### Add More Data
1. Add papers to `data/papers/` directory
2. Update `data/knowledge_base.json` with structured content
3. Restart the system to reload vector store

### Expand Agents
1. Add domain-specific agents (e.g., Computer Vision, NLP)
2. Create specialized tools for each domain
3. Implement more sophisticated routing logic

### Scale Up
1. Use more powerful OpenAI models (GPT-4)
2. Add more sophisticated RAG techniques
3. Implement conversation memory and history

---

**Built with LangChain ❤️ - Clean, Simple, Effective** 