"""
LangChain-based Multi-Agent System for ML Q&A
Clean implementation using LangChain's agent framework
"""
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import json
from ..config import settings

class LangChainMLAgents:
    """LangChain-based multi-agent system for ML Q&A"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.AGENT_TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.vector_store = None
        self.agents = {}
        self._setup_vector_store()
        self._setup_agents()
    
    def _setup_vector_store(self):
        """Setup ChromaDB vector store for RAG"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL
            )
            
            # Load knowledge base if it exists
            knowledge_path = settings.KNOWLEDGE_BASE_FILE
            if os.path.exists(knowledge_path):
                with open(knowledge_path, 'r') as f:
                    knowledge_data = json.load(f)
                
                # Convert to documents
                documents = []
                for item in knowledge_data:
                    doc = Document(
                        page_content=item.get('content', ''),
                        metadata={
                            'title': item.get('title', ''),
                            'source': item.get('source', ''),
                            'category': item.get('category', 'general')
                        }
                    )
                    documents.append(doc)
                
                # Create vector store
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=settings.VECTOR_STORE_PATH
                )
                print(f"✅ Vector store loaded with {len(documents)} documents")
            else:
                print("⚠️ No knowledge base found. Creating empty vector store.")
                self.vector_store = Chroma(
                    embedding_function=embeddings,
                    persist_directory=settings.VECTOR_STORE_PATH
                )
        except Exception as e:
            print(f"❌ Error setting up vector store: {e}")
            self.vector_store = None
    
    def _create_rag_tool(self):
        """Create RAG tool for knowledge retrieval"""
        def search_knowledge(query: str) -> str:
            """Search the knowledge base for relevant information"""
            if not self.vector_store:
                return "Knowledge base not available"
            
            try:
                docs = self.vector_store.similarity_search(query, k=3)
                if not docs:
                    return "No relevant information found in knowledge base"
                
                results = []
                for doc in docs:
                    content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    title = doc.metadata.get('title', 'Unknown')
                    results.append(f"Source: {title}\nContent: {content}")
                
                return "\n\n".join(results)
            except Exception as e:
                return f"Error searching knowledge base: {e}"
        
        return Tool(
            name="search_knowledge",
            description="Search the ML/DL knowledge base for relevant papers and information",
            func=search_knowledge
        )
    
    def _setup_agents(self):
        """Setup specialized LangChain agents"""
        
        # Create RAG tool
        rag_tool = self._create_rag_tool()
        
        # Research Agent
        research_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Research Agent specializing in Machine Learning and Deep Learning literature.
            Your role is to:
            - Find and synthesize information from academic papers
            - Provide literature reviews and recent research findings
            - Cite relevant papers and studies
            - Explain research trends and developments
            
            Always search the knowledge base first before providing answers."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        research_agent = create_openai_functions_agent(
            llm=self.llm,
            tools=[rag_tool],
            prompt=research_prompt
        )
        
        self.agents['research'] = AgentExecutor(
            agent=research_agent,
            tools=[rag_tool],
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Theory Agent
        theory_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Theory Agent specializing in explaining mathematical concepts in ML/DL.
            Your role is to:
            - Explain mathematical foundations and theory
            - Derive equations and formulas
            - Clarify conceptual understanding
            - Break down complex algorithms step-by-step
            
            Use the knowledge base to find relevant theoretical content."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        theory_agent = create_openai_functions_agent(
            llm=self.llm,
            tools=[rag_tool],
            prompt=theory_prompt
        )
        
        self.agents['theory'] = AgentExecutor(
            agent=theory_agent,
            tools=[rag_tool],
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Implementation Agent
        implementation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Implementation Agent specializing in ML/DL code and practical applications.
            Your role is to:
            - Generate code examples and implementations
            - Provide practical programming guidance
            - Suggest best practices and optimizations
            - Help with debugging and troubleshooting
            
            Search for code examples and implementation details in the knowledge base."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        implementation_agent = create_openai_functions_agent(
            llm=self.llm,
            tools=[rag_tool],
            prompt=implementation_prompt
        )
        
        self.agents['implementation'] = AgentExecutor(
            agent=implementation_agent,
            tools=[rag_tool],
            verbose=True,
            handle_parsing_errors=True
        )
        
        print("✅ LangChain agents initialized successfully")
    
    def route_query(self, query: str) -> str:
        """Simple routing logic to determine best agent"""
        query_lower = query.lower()
        
        # Implementation keywords
        implementation_keywords = ['code', 'implement', 'pytorch', 'tensorflow', 'example', 'how to', 'build']
        if any(keyword in query_lower for keyword in implementation_keywords):
            return 'implementation'
        
        # Research keywords  
        research_keywords = ['paper', 'research', 'study', 'literature', 'recent', 'state of art']
        if any(keyword in query_lower for keyword in research_keywords):
            return 'research'
        
        # Theory keywords (default)
        theory_keywords = ['explain', 'what is', 'how does', 'theory', 'mathematical', 'algorithm']
        if any(keyword in query_lower for keyword in theory_keywords):
            return 'theory'
        
        # Default to theory agent
        return 'theory'
    
    def process_query(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """Process query using appropriate LangChain agent"""
        if chat_history is None:
            chat_history = []
        
        # Route to appropriate agent
        agent_name = self.route_query(query)
        agent = self.agents.get(agent_name)
        
        if not agent:
            return {
                'error': f'Agent {agent_name} not available',
                'query': query,
                'agent_used': agent_name
            }
        
        try:
            # Process with LangChain agent
            result = agent.invoke({
                'input': query,
                'chat_history': chat_history
            })
            
            return {
                'query': query,
                'agent_used': agent_name,
                'response': result['output'],
                'success': True
            }
            
        except Exception as e:
            return {
                'error': f'Error processing query: {str(e)}',
                'query': query,
                'agent_used': agent_name,
                'success': False
            }
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        return list(self.agents.keys())
    
    def health_check(self) -> Dict[str, bool]:
        """Check system health"""
        status = {
            'llm_connection': False,
            'vector_store': self.vector_store is not None,
            'agents_loaded': len(self.agents) > 0
        }
        
        # Test LLM connection
        try:
            response = self.llm.invoke([HumanMessage(content="test")])
            status['llm_connection'] = bool(response.content)
        except:
            status['llm_connection'] = False
        
        # Test each agent
        for agent_name in self.agents:
            status[f'agent_{agent_name}'] = agent_name in self.agents
        
        return status

# Convenience function for easy import
def create_langchain_ml_agents():
    """Factory function to create LangChain ML agents"""
    return LangChainMLAgents() 