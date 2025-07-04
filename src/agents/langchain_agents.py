"""
LangChain-based Multi-Agent System for ML Q&A
Clean implementation using LangChain's agent framework
"""
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_tool_calling_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import json
import time
from datetime import datetime
from ..config import settings

class PineconeHostedEmbeddings:
    """Custom embeddings class using Pinecone's hosted inference"""
    def __init__(self, index_name):
        self.index_name = index_name
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(index_name)
    
    def embed_documents(self, texts):
        """
        For hosted embeddings, we don't actually embed here.
        We return dummy vectors for LangChain compatibility.
        The actual embedding happens when documents are upserted via LangChain.
        """
        return [[0.0] * 1024 for _ in texts]  # 1024 is llama-text-embed-v2 dimension
    
    def embed_query(self, text):
        """
        For hosted embeddings, we don't actually embed here.
        We return a dummy vector for LangChain compatibility.
        The actual embedding happens during search.
        """
        return [0.0] * 1024

class LangChainMLAgents:
    """LangChain-based multi-agent system for ML Q&A"""
    
    def __init__(self, auto_upsert: bool = False):
        """
        Initialize LangChain ML Agents
        
        Args:
            auto_upsert: Whether to automatically upsert documents to Pinecone during initialization
                        Default is False - documents will only be upserted when explicitly requested
        """
        self.auto_upsert = auto_upsert
        
        # Initialize chat history storage using LangChain message format
        self.chat_history = []  # List of HumanMessage and AIMessage objects
        self.max_history_length = 20  # Maximum number of messages to keep in memory
        
        # Initialize routing LLM for semantic query analysis (cheap but effective)
        self.routing_llm = ChatOpenAI(
            model="gpt-4o-mini",  # Very cheap and fast for classification tasks
            temperature=0.0,  # Low temperature for consistent routing decisions
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize multiple LLMs for different agents
        
        # GPT-4 for theory agent
        self.theory_llm = ChatOpenAI(
            model=settings.THEORY_MODEL,
            temperature=settings.AGENT_TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Ollama for research agent (API-based)
        self.research_llm = None
        try:
            # Configure Ollama with API endpoint
            ollama_kwargs = {
                "model": settings.RESEARCH_MODEL,
                "temperature": 0.1,  # Lower temperature for research accuracy
                "base_url": settings.OLLAMA_BASE_URL,
            }
            
            # Add API key if provided (for hosted Ollama services)
            if settings.OLLAMA_API_KEY:
                ollama_kwargs["api_key"] = settings.OLLAMA_API_KEY
            
            self.research_llm = ChatOllama(**ollama_kwargs)
            print(f"✅ Ollama API initialized for research agent (endpoint: {settings.OLLAMA_BASE_URL})")
        except Exception as e:
            print(f"⚠️ Ollama API initialization failed: {e}, using GPT-4 for research agent")
            self.research_llm = self.theory_llm
        
        # Claude for implementation agent
        self.implementation_llm = None
        if settings.ANTHROPIC_API_KEY:
            try:
                self.implementation_llm = ChatAnthropic(
                    model=settings.IMPLEMENTATION_MODEL,
                    temperature=settings.AGENT_TEMPERATURE,
                    anthropic_api_key=settings.ANTHROPIC_API_KEY
                )
                print("✅ Claude 3.5 Sonnet initialized for implementation agent")
            except Exception as e:
                print(f"⚠️ Claude initialization failed: {e}, using GPT-4 for implementation agent")
                self.implementation_llm = self.theory_llm
        else:
            print("⚠️ ANTHROPIC_API_KEY not found, using GPT-4 for implementation agent")
            self.implementation_llm = self.theory_llm
        
        self.vector_store = None
        self.pinecone_index = None
        self.embeddings = None
        self.agents = {}
        self._setup_vector_store()
        self._setup_agents()
    
    def _clean_text_for_pinecone(self, text):
        """Clean text for Pinecone storage"""
        if not text:
            return ""
        return str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
    
    def _setup_vector_store(self):
        """Setup vector store (Pinecone) for RAG"""
        try:
            # Setup Pinecone as the exclusive vector store with hosted embeddings
            if settings.VECTOR_STORE_TYPE.lower() != "pinecone":
                raise ValueError("Configuration error: This project is configured to use 'pinecone' exclusively.")
            
            # Load knowledge base if it exists
            knowledge_path = settings.KNOWLEDGE_BASE_FILE
            documents = []
            
            if os.path.exists(knowledge_path):
                # Enforce UTF-8 encoding to prevent errors on Windows
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    knowledge_data = json.load(f)
                
                # Convert to documents
                for item in knowledge_data:
                    doc = Document(
                        page_content=item.get('content', ''),
                        metadata={
                            'title': item.get('title', ''),
                            'source': item.get('source', ''),
                            'category': item.get('category', 'general'),
                            'type': item.get('type', 'unknown'),
                            'id': item.get('id', ''),
                            'authors': item.get('authors', []),
                            'categories': item.get('categories', []),
                            # Handle chunk-specific metadata
                            'chunk_index': item.get('chunk_index'),
                            'total_chunks': item.get('total_chunks'),
                            'parent_paper_id': item.get('parent_paper_id')
                        }
                    )
                    documents.append(doc)
                print(f"📚 Loaded {len(documents)} documents from knowledge base")
            else:
                print("⚠️ No knowledge base found. Vector store will be empty initially.")
            
            self._setup_pinecone_store(documents)
                
        except Exception as e:
            print(f"❌ Error setting up vector store: {e}")
            print("⚠️ Continuing without vector store - agents will work but without RAG capabilities")
            self.vector_store = None
    
    def _setup_pinecone_store(self, documents):
        """Setup Pinecone vector store with integrated inference using new SDK v7.x API"""
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required for Pinecone vector store. Please set it in your .env file.")
        
        try:
            # Import Pinecone dependencies only when needed
            from pinecone import Pinecone
            from langchain_pinecone import PineconeVectorStore
            
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            
            # Check if index exists, create if not
            index_name = settings.PINECONE_INDEX_NAME
            existing_indexes = [index.name for index in pc.list_indexes()]
            
            if index_name not in existing_indexes:
                print(f"🔄 Creating Pinecone index with integrated embedding: {index_name}")
                
                # Create index with integrated inference using new v7.x API
                pc.create_index_for_model(
                    name=index_name,
                    cloud="aws",
                    region="us-east-1",
                    embed={
                        "model": "llama-text-embed-v2",
                        "field_map": {
                            "text": "text"  # Map the record field to be embedded
                        }
                    }
                )
                print(f"✅ Pinecone index '{index_name}' created with integrated llama-text-embed-v2")
            else:
                print(f"✅ Using existing Pinecone index: {index_name}")
            
            # Create the custom embeddings instance
            self.embeddings = PineconeHostedEmbeddings(index_name)
            self.pinecone_index = pc.Index(index_name)
            
            # Create LangChain vector store using integrated embeddings
            if documents:
                if self.auto_upsert:
                    print(f"🔄 Auto-upsert enabled: Setting up LangChain PineconeVectorStore with {len(documents)} documents...")
                    self.upsert_documents_to_pinecone(documents)
                else:
                    print(f"📚 Found {len(documents)} documents in knowledge base (auto-upsert disabled)")
                    print("💡 To upload documents to Pinecone, call upsert_knowledge_base_to_pinecone() method")
            else:
                print("📝 No documents found in knowledge base - vector store will be empty")
                
            # Create the LangChain vector store pointing to the same index
            self.vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embeddings,
                text_key="text"  # Field containing the text content
            )
            
            print(f"✅ LangChain PineconeVectorStore created with integrated inference")
            
        except ImportError as e:
            print(f"❌ Pinecone dependencies error: {e}")
            print("💡 Please install pinecone dependencies: pip install pinecone langchain-pinecone")
            raise e
    
    def _truncate_content_for_pinecone(self, content: str, max_bytes: int = 35000) -> str:
        """
        Truncate content to fit within Pinecone's metadata size limit
        Leaves room for other metadata fields
        """
        if not content:
            return ""
        
        # Convert to bytes to check actual size
        content_bytes = content.encode('utf-8')
        
        if len(content_bytes) <= max_bytes:
            return content
        
        # Truncate while trying to preserve word boundaries
        truncated = content_bytes[:max_bytes].decode('utf-8', errors='ignore')
        
        # Try to cut at the last complete sentence or paragraph
        for delimiter in ['\n\n', '. ', '\n', ' ']:
            last_pos = truncated.rfind(delimiter)
            if last_pos > max_bytes * 0.8:  # Only if we don't lose too much content
                truncated = truncated[:last_pos + len(delimiter)]
                break
        
        # Add truncation indicator
        if len(content_bytes) > max_bytes:
            truncated += "\n\n[Content truncated due to size limits...]"
        
        return truncated

    def upsert_documents_to_pinecone(self, documents: List[Document], namespace: str = "__default__"):
        """
        Unified method to upsert LangChain Documents to Pinecone with hosted embeddings
        This method maintains LangChain compatibility while handling the hosted embedding format
        """
        if not self.pinecone_index:
            print("❌ Pinecone index not initialized")
            return False
        
        try:
            # Prepare records for upsert in Pinecone hosted embedding format
            records = []
            skipped_count = 0
            
            for i, doc in enumerate(documents):
                # Extract metadata with proper handling of different types
                title = self._clean_text_for_pinecone(str(doc.metadata.get('title', '')))[:80]
                source = self._clean_text_for_pinecone(str(doc.metadata.get('source', '')))[:100]
                entry_type = str(doc.metadata.get('type', 'unknown'))
                entry_id = doc.metadata.get('id', f"doc_{i}")
                authors = doc.metadata.get('authors', [])
                categories = doc.metadata.get('categories', [])
                
                # Truncate content to fit within Pinecone limits
                content = self._truncate_content_for_pinecone(doc.page_content)
                
                # Skip empty content after truncation
                if not content.strip():
                    print(f"⚠️ Skipping document {entry_id} - empty content after processing")
                    skipped_count += 1
                    continue
                
                # Base record structure for hosted embeddings
                record = {
                    "_id": entry_id,
                    "text": content,  # This will be embedded by Pinecone
                    "title": title,
                    "source": source,
                    "type": entry_type,
                    "authors": ', '.join(authors)[:200] if authors else '',  # Limit authors field
                    "categories": ', '.join(categories)[:200] if categories else ''  # Limit categories field
                }
                
                # Add chunk-specific metadata if present (for LLM chunked content)
                if doc.metadata.get('chunk_index') is not None:
                    record.update({
                        "chunk_index": str(doc.metadata.get('chunk_index', 0)),
                        "total_chunks": str(doc.metadata.get('total_chunks', 1)),
                        "parent_paper_id": str(doc.metadata.get('parent_paper_id', ''))[:50]  # Limit parent ID
                    })
                
                # Estimate total record size
                record_size = sum(len(str(v).encode('utf-8')) for v in record.values())
                if record_size > 40000:  # Still too big
                    print(f"⚠️ Skipping document {entry_id} - still too large after truncation ({record_size} bytes)")
                    skipped_count += 1
                    continue
                
                records.append(record)
            
            if skipped_count > 0:
                print(f"⚠️ Skipped {skipped_count} documents due to size constraints")
            
            if not records:
                print("❌ No valid records to upload after processing")
                return False
            
            # Upsert in batches using Pinecone hosted embedding format
            batch_size = 32  # Conservative batch size to avoid rate limits with hosted embeddings
            total_batches = (len(records) - 1) // batch_size + 1
            
            print(f"🔄 Upserting {len(records)} documents to Pinecone in {total_batches} batches...")
            
            successful_batches = 0
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    # Use correct API format for hosted embeddings
                    self.pinecone_index.upsert_records(namespace, batch)
                    print(f"✅ Uploaded batch {batch_num}/{total_batches} ({len(batch)} records)")
                    successful_batches += 1
                    
                    # Add delay to avoid rate limiting (increased for hosted embeddings)
                    time.sleep(2.0)
                    
                except Exception as e:
                    error_str = str(e)
                    print(f"❌ Error uploading batch {batch_num}: {e}")
                    
                    # Handle rate limiting
                    if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                        print(f"⏳ Rate limit hit, waiting 30 seconds before retrying...")
                        time.sleep(30)
                        try:
                            # Retry the batch
                            self.pinecone_index.upsert_records(namespace, batch)
                            print(f"✅ Uploaded batch {batch_num}/{total_batches} ({len(batch)} records) [Retry]")
                            successful_batches += 1
                            time.sleep(2.0)
                        except Exception as retry_e:
                            print(f"❌ Retry failed for batch {batch_num}: {retry_e}")
                            continue
                    # Handle metadata size issues
                    elif "Metadata size" in error_str:
                        print(f"   Batch contains records that are still too large")
                        for j, record in enumerate(batch):
                            record_size = sum(len(str(v).encode('utf-8')) for v in record.values())
                            if record_size > 35000:
                                print(f"   - Record {record['_id']}: {record_size} bytes")
                        continue
                    else:
                        continue
            
            if successful_batches > 0:
                print(f"✅ Successfully uploaded {successful_batches}/{total_batches} batches")
                return True
            else:
                print("❌ No batches were successfully uploaded")
                return False
            
        except Exception as e:
            print(f"❌ Error upserting documents to Pinecone: {e}")
            return False
    
    def upsert_knowledge_base_to_pinecone(self, knowledge_base_path: str = None, namespace: str = "__default__"):
        """
        Upload knowledge base entries to Pinecone using LangChain-compatible approach
        This method converts JSON entries to LangChain Documents and uses the unified upsert method
        """
        if not knowledge_base_path:
            knowledge_base_path = settings.KNOWLEDGE_BASE_FILE
        
        if not os.path.exists(knowledge_base_path):
            print(f"❌ Knowledge base file not found: {knowledge_base_path}")
            return False
        
        try:
            # Load knowledge base
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            
            print(f"📚 Loaded {len(knowledge_base)} entries from knowledge base")
            
            # Convert JSON entries to LangChain Documents
            documents = []
            for entry in knowledge_base:
                doc = Document(
                    page_content=entry.get('content', ''),
                    metadata={
                        'id': entry.get('id', ''),
                        'title': entry.get('title', ''),
                        'source': entry.get('source', ''),
                        'type': entry.get('type', 'unknown'),
                        'authors': entry.get('authors', []),
                        'categories': entry.get('categories', []),
                        # Handle chunk-specific metadata for LLM chunking
                        'chunk_index': entry.get('chunk_index'),
                        'total_chunks': entry.get('total_chunks'),
                        'parent_paper_id': entry.get('parent_paper_id')
                    }
                )
                documents.append(doc)
            
            # Use unified upsert method
            success = self.upsert_documents_to_pinecone(documents, namespace)
            
            if success:
                # Get index stats
                stats = self.pinecone_index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                
                print(f"✅ Successfully uploaded knowledge base to Pinecone!")
                print(f"📊 Total vectors in index: {total_vectors}")
                print(f"📊 Namespace: {namespace}")
                
                return True
            else:
                print("❌ Failed to upload knowledge base to Pinecone")
                return False
                
        except Exception as e:
            print(f"❌ Error uploading knowledge base to Pinecone: {e}")
            return False
    
    def _create_rag_tool(self):
        """Create RAG tool for knowledge retrieval using LangChain's PineconeVectorStore"""
        def search_knowledge(query: str) -> str:
            """Search the knowledge base for relevant information"""
            if not self.vector_store:
                return "Knowledge base not available"
            
            try:
                # For hosted embeddings, we need to use the raw Pinecone API for search
                # because LangChain's similarity_search doesn't support hosted inference yet
                if hasattr(self, 'pinecone_index') and self.pinecone_index:
                    # Use Pinecone's hosted embedding for query
                    query_payload = {
                        "inputs": {
                            "text": query
                        },
                        "top_k": 4
                    }
                    
                    search_results = self.pinecone_index.search(query=query_payload, namespace="__default__")
                    
                    # Handle the correct response format for hosted embeddings
                    hits = search_results.get('result', {}).get('hits', [])
                    if not hits:
                        return "No relevant information found in knowledge base"
                    
                    results = []
                    for hit in hits:
                        # Get metadata from fields (correct format for hosted embeddings)
                        fields = hit.get('fields', {})
                        title = fields.get('title', 'Unknown')
                        source = fields.get('source', '')
                        # Get more comprehensive content for better context
                        text = fields.get('text', '')
                        content = text[:2000] + "..." if len(text) > 2000 else text
                        score = hit.get('_score', 0)
                        results.append(f"Source: {title} (Score: {score:.3f})\nContent: {content}")
                    
                    return "\n\n".join(results)
                else:
                    # Fallback to LangChain's method (though it won't work well with hosted embeddings)
                    docs = self.vector_store.similarity_search(query, k=3)
                    if not docs:
                        return "No relevant information found in knowledge base"
                    
                    results = []
                    for doc in docs:
                        content = doc.page_content[:4000] + "..." if len(doc.page_content) > 4000 else doc.page_content
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
    
    def _create_cot_tool(self):
        """Create Chain of Thoughts reasoning tool for structured thinking"""
        def chain_of_thoughts_reasoning(problem_statement: str) -> str:
            """
            Apply Chain of Thoughts reasoning to break down complex problems
            This tool helps structure thinking for mathematical and theoretical problems
            """
            try:
                # Template for CoT reasoning
                cot_template = f"""
                Chain of Thoughts Analysis for: {problem_statement}
                
                🎯 **Problem Decomposition:**
                - Main question: {problem_statement}
                - Sub-problems to address: [Identify key components]
                - Required knowledge areas: [List relevant ML/DL concepts]
                
                🔍 **Reasoning Strategy:**
                - Approach: [Top-down/Bottom-up/Analogical reasoning]
                - Key assumptions: [List any assumptions made]
                - Potential challenges: [Identify complex aspects]
                
                📊 **Conceptual Hierarchy:**
                - Foundation concepts: [Basic building blocks]
                - Intermediate concepts: [Mid-level understanding]
                - Advanced concepts: [Complex relationships]
                
                🧮 **Mathematical Structure:**
                - Variables and notation: [Define symbols]
                - Key equations: [Relevant formulas]
                - Derivation steps: [Logical progression]
                
                💡 **Insight Generation:**
                - Key insights: [Important realizations]
                - Common misconceptions: [What to avoid]
                - Practical implications: [Real-world applications]
                
                This structured analysis provides a framework for systematic reasoning about: {problem_statement}
                """
                
                return cot_template.strip()
                
            except Exception as e:
                return f"Error in Chain of Thoughts reasoning: {e}"
        
        return Tool(
            name="chain_of_thoughts_reasoning",
            description="Apply structured Chain of Thoughts reasoning to break down complex mathematical and theoretical problems into manageable components",
            func=chain_of_thoughts_reasoning
        )
    
    def _setup_agents(self):
        """Setup specialized LangChain agents"""
        
        # Create RAG tool
        rag_tool = self._create_rag_tool()
        
        # Create Chain of Thoughts tool for Theory Agent
        cot_tool = self._create_cot_tool()
        
        # Research Agent
        research_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Research Agent specializing in Machine Learning and Deep Learning literature.
            Your role is to:
            - Find and synthesize information from academic papers
            - Provide literature reviews and recent research findings
            - Cite relevant papers and studies
            - Explain research trends and developments
            
            **Knowledge Source Strategy:**
            - ALWAYS start by searching the knowledge base using the search_knowledge tool
            - If the knowledge base contains detailed research papers and relevant information → use it as primary source
            - If the knowledge base contains only basic/placeholder content → supplement with your pre-trained knowledge
            - If the knowledge base search fails or returns insufficient results → use your extensive pre-trained knowledge
            - Clearly indicate which knowledge sources you're using in your response
            
            **Guidelines:**
            - Always search the knowledge base first, but don't be limited by insufficient results
            - If you find relevant papers in the knowledge base, use them and cite them appropriately
            - If knowledge base content is minimal, provide comprehensive answers using your pre-trained knowledge
            - Never refuse to answer due to insufficient knowledge base content
            - Provide recent research trends and developments from your training data when knowledge base is limited
                         - Always aim to give comprehensive, helpful research-oriented answers"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Use appropriate agent creation method based on LLM type
        if isinstance(self.research_llm, ChatOllama):
            # For Ollama models, use generic tool calling agent
            research_agent = create_tool_calling_agent(
                llm=self.research_llm,
                tools=[rag_tool],
                prompt=research_prompt
            )
        else:
            # For OpenAI models (fallback case)
            research_agent = create_openai_functions_agent(
                llm=self.research_llm,
                tools=[rag_tool],
                prompt=research_prompt
            )
        
        self.agents['research'] = AgentExecutor(
            agent=research_agent,
            tools=[rag_tool],
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True  # Enable intermediate steps capture
        )
        
        # Theory Agent with Chain of Thoughts
        theory_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Theory Agent specializing in explaining mathematical concepts in ML/DL using Chain of Thoughts reasoning.

            **Knowledge Integration Strategy:**
            - ALWAYS search the knowledge base first using the search_knowledge tool
            - Combine RAG results with your extensive pretrained knowledge for complete answers
            - If RAG provides relevant information: Use it as foundation and enhance with pretrained knowledge
            - If RAG is insufficient or empty: Rely entirely on your pretrained knowledge
            - Never limit your answer to only RAG content - always provide comprehensive explanations

            **Response Approach:**
            1. **Search & Assess:** Use search_knowledge tool and evaluate the results
            2. **Knowledge Synthesis:** Combine the best of RAG + pretrained knowledge
            3. **Structured Explanation:** Use Chain of Thoughts reasoning for clear explanations

            **Response Format:**

            🔍 **Knowledge Search:**
            [Search knowledge base for relevant information]

            📖 **Knowledge Integration:**
            [State your approach: "Combining RAG findings with pretrained knowledge" OR "Using pretrained knowledge (RAG insufficient)"]

            🧠 **Chain of Thoughts Analysis:**

            **Step 1: Problem Understanding**
            [Clearly restate what needs to be explained]

            **Step 2: Fundamental Concepts**
            [Define key terms and foundational knowledge]

            **Step 3: Mathematical Framework** (if applicable)
            [Present equations, formulas, derivations with clear notation]

            **Step 4: Detailed Explanation**
            [Thorough explanation with logical reasoning]

            **Step 5: Intuitive Understanding**
            [Explain the "why" and "how" in accessible terms]

            **Step 6: Connections & Applications**
            [Link to related concepts and practical implications]

            📝 **Key Takeaways:**
            [Summarize the most important insights]

            **Core Principles:**
            - Always provide comprehensive, complete answers
            - Use both knowledge sources intelligently - don't rely solely on either
            - Show mathematical derivations step-by-step when relevant
            - Explain both formal concepts and intuitive understanding
            - Build from simple to complex ideas
            - Use LaTeX notation for mathematical expressions
            - Connect theory to practical ML/DL applications"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        theory_agent = create_openai_functions_agent(
            llm=self.theory_llm,  # Use GPT-4 for theory/math tasks
            tools=[rag_tool, cot_tool],
            prompt=theory_prompt
        )
        
        self.agents['theory'] = AgentExecutor(
            agent=theory_agent,
            tools=[rag_tool, cot_tool],
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True  # Enable intermediate steps capture
        )
        
        # Implementation Agent
        implementation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Implementation Agent specializing in ML/DL code and practical applications.
            Your role is to:
            - Generate code examples and implementations
            - Provide practical programming guidance
            - Suggest best practices and optimizations
            - Help with debugging and troubleshooting
            
            **Knowledge Source Strategy:**
            - ALWAYS start by searching the knowledge base using the search_knowledge tool
            - If the knowledge base contains relevant code examples and implementation details → use them as reference
            - If the knowledge base contains only basic/placeholder content → use your extensive pre-trained knowledge
            - If the knowledge base search fails or returns insufficient results → use your comprehensive programming knowledge
            - Clearly indicate which knowledge sources you're using in your response
            
            **Guidelines:**
            - Always search the knowledge base first, but don't be limited by insufficient results
            - If you find relevant implementation details in the knowledge base, use them and reference them
            - If knowledge base content is minimal, provide comprehensive code examples using your pre-trained knowledge
            - Never refuse to answer due to insufficient knowledge base content
            - Generate practical, working code examples regardless of knowledge base availability
            - Always aim to give complete, runnable implementations with proper explanations"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Use appropriate agent creation method based on LLM type
        if isinstance(self.implementation_llm, ChatAnthropic):
            # For Claude models, use generic tool calling agent
            implementation_agent = create_tool_calling_agent(
                llm=self.implementation_llm,
                tools=[rag_tool],
                prompt=implementation_prompt
            )
        else:
            # For OpenAI models (fallback case)
            implementation_agent = create_openai_functions_agent(
                llm=self.implementation_llm,
                tools=[rag_tool],
                prompt=implementation_prompt
            )
        
        self.agents['implementation'] = AgentExecutor(
            agent=implementation_agent,
            tools=[rag_tool],
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True  # Enable intermediate steps capture
        )
        
        print("✅ LangChain agents initialized successfully")
    
    def _semantic_route_query(self, query: str) -> str:
        """
        Use LLM to semantically analyze query intention and route to appropriate agent
        
        Args:
            query: User question to analyze
            
        Returns:
            Agent name (research, theory, implementation) or None if routing fails
        """
        try:
            # Structured prompt for query classification
            routing_prompt = f"""You are an AI assistant that analyzes user queries and determines which type of expert should handle them.

Available Expert Agents:
1. RESEARCH - For literature review, paper analysis, recent developments, surveys, comparisons
2. THEORY - For mathematical explanations, conceptual understanding, algorithmic details, theoretical foundations  
3. IMPLEMENTATION - For coding, practical examples, debugging, tutorials, how-to guides

Query to analyze: "{query}"

Based on the query's semantic meaning and intent, classify it into ONE of these categories:

Analysis Framework:
- If the query asks about papers, research, recent developments, literature, studies, surveys, or comparisons → RESEARCH
- If the query asks for mathematical explanations, theoretical concepts, algorithmic details, or conceptual understanding → THEORY  
- If the query asks for code, implementation, examples, tutorials, debugging, or practical guidance → IMPLEMENTATION

Respond with ONLY the single word: RESEARCH, THEORY, or IMPLEMENTATION

Classification:"""

            # Get routing decision from LLM
            routing_messages = [HumanMessage(content=routing_prompt)]
            response = self.routing_llm.invoke(routing_messages)
            
            # Extract and validate response
            routing_decision = response.content.strip().upper()
            
            # Map to agent names
            agent_mapping = {
                'RESEARCH': 'research',
                'THEORY': 'theory', 
                'IMPLEMENTATION': 'implementation'
            }
            
            if routing_decision in agent_mapping:
                return agent_mapping[routing_decision]
            else:
                print(f"⚠️ Semantic routing returned unexpected response: {routing_decision}")
                return None
                
        except Exception as e:
            print(f"⚠️ Semantic routing failed: {e}")
            return None
    
    def _keyword_route_query(self, query: str) -> str:
        """
        Fallback keyword-based routing logic (original implementation)
        
        Args:
            query: User question to analyze
            
        Returns:
            Agent name based on keyword matching
        """
        query_lower = query.lower()

        # Implementation keywords - expanded for broader coverage
        implementation_keywords = [
            # Actions
            'code', 'implement', 'implementation', 'write', 'create', 'generate', 'build',
            'debug', 'fix', 'error', 'test', 'optimize', 'refactor', 'run', 'execute',
            'script', 'function', 'class', 'module', 'library', 'package', 'api', 'framework',

            # Libraries/Frameworks
            'pytorch', 'tensorflow', 'keras', 'numpy', 'pandas', 'scikit', 'sklearn',
            'fastapi', 'streamlit', 'docker',

            # Concepts
            'example', 'how to', 'tutorial', 'demo',
            'python', 'jupyter', 'notebook', 'coding', 'programming'
        ]
        if any(keyword in query_lower for keyword in implementation_keywords):
            return 'implementation'

        # Research keywords - expanded for broader coverage
        research_keywords = [
            # Actions
            'find', 'search', 'summarize', 'compare', 'review', 'survey', 'cite',

            # Nouns
            'paper', 'papers', 'study', 'studies', 'literature', 'research', 'publication',
            'journal', 'conference', 'arxiv', 'citation', 'background',

            # Concepts
            'recent', 'state of the art', 'sota', 'advances', 'developments', 'trends'
        ]
        if any(keyword in query_lower for keyword in research_keywords):
            return 'research'

        # Theory keywords (default) - expanded for broader coverage
        theory_keywords = [
            # Actions
            'explain', 'understand', 'define', 'derive', 'prove',

            # Nouns
            'theory', 'mathematical', 'math', 'concept', 'conceptual', 'idea', 'logic',
            'principle', 'foundation', 'formula', 'equation', 'derivation', 'proof',
            'definition', 'intuition', 'algorithm', 'architecture',

            # Questions
            'what is', 'how does', 'why is'
        ]
        if any(keyword in query_lower for keyword in theory_keywords):
            return 'theory'

        # Default to theory agent if no keywords match
        return 'theory'
    
    def route_query(self, query: str) -> str:
        """
        Smart routing logic using semantic analysis with fallback to keyword matching
        
        Args:
            query: User question to analyze
            
        Returns:
            Agent name (research, theory, implementation) 
        """
        # Try semantic routing first using LLM
        semantic_result = self._semantic_route_query(query)
        
        if semantic_result:
            print(f"🧠 Semantic routing: {query[:50]}... → {semantic_result}")
            return semantic_result
        
        # Fallback to keyword-based routing
        keyword_result = self._keyword_route_query(query)
        print(f"🔤 Keyword routing (fallback): {query[:50]}... → {keyword_result}")
        return keyword_result
    
    def process_query(self, query: str, chat_history: List = None, show_thinking: bool = True) -> Dict[str, Any]:
        """
        Process query using appropriate LangChain agent with chat history support
        
        Args:
            query: User question to process
            chat_history: Optional external chat history (if None, uses internal history)
            show_thinking: Whether to display agent thinking process
            
        Returns:
            Dict containing response, agent info, and thinking process
        """
        # Use internal chat history if no external history provided
        if chat_history is None:
            chat_history = self.chat_history.copy()  # Use internal history
        
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
            # Add current query to chat history as HumanMessage
            current_human_message = HumanMessage(content=query)
            
            # Process with LangChain agent using full chat history
            result = agent.invoke({
                'input': query,
                'chat_history': chat_history
            })
            
            # Extract intermediate steps (thinking process)
            thinking_steps = []
            if show_thinking and 'intermediate_steps' in result:
                thinking_steps = self._format_thinking_process(result['intermediate_steps'], agent_name)
            
            # Standardize the response format from different agent types
            raw_response = result.get('output', '')
            
            # Tool-calling agents (Claude/Ollama) may return a list of dicts
            if isinstance(raw_response, list) and raw_response and isinstance(raw_response[0], dict):
                # Handle formats like: [{'text': '...', 'type': 'text'}]
                final_response = " ".join([chunk.get('text', '') for chunk in raw_response if 'text' in chunk])
            elif isinstance(raw_response, str):
                # Standard string output (from OpenAI function-calling agent)
                final_response = raw_response
            else:
                # Fallback for any other unexpected formats
                final_response = str(raw_response)

            response_data = {
                'query': query,
                'agent_used': agent_name,
                'response': final_response.strip(),
                'success': True
            }
            
            # Add thinking process if requested and available
            if show_thinking and thinking_steps:
                response_data['thinking_process'] = thinking_steps
                response_data['has_thinking'] = True
            else:
                response_data['has_thinking'] = False
            
            # Update internal chat history with the conversation
            self._update_chat_history(query, final_response.strip(), agent_name)
            
            return response_data
            
        except Exception as e:
            return {
                'error': f'Error processing query: {str(e)}',
                'query': query,
                'agent_used': agent_name,
                'success': False,
                'has_thinking': False
            }
    
    def _format_thinking_process(self, intermediate_steps: List, agent_name: str) -> List[Dict[str, Any]]:
        """Format the agent's thinking process for display"""
        formatted_steps = []
        
        for i, (action, observation) in enumerate(intermediate_steps):
            step_info = {
                'step_number': i + 1,
                'action_type': 'tool_call',
                'tool_name': getattr(action, 'tool', 'unknown'),
                'tool_input': getattr(action, 'tool_input', {}),
                'observation': observation,
                'timestamp': None  # Could add timestamp if needed
            }
            
            # Format based on tool type
            if step_info['tool_name'] == 'search_knowledge':
                step_info['description'] = f"🔍 Searching knowledge base for: {step_info['tool_input']}"
                step_info['result_summary'] = observation[:200] + "..." if len(observation) > 200 else observation
                
            elif step_info['tool_name'] == 'chain_of_thoughts_reasoning':
                step_info['description'] = f"🧠 Applying Chain of Thoughts reasoning to: {step_info['tool_input']}"
                step_info['result_summary'] = "Generated structured reasoning framework"
                
            else:
                step_info['description'] = f"🔧 Using tool '{step_info['tool_name']}'"
                step_info['result_summary'] = observation[:200] + "..." if len(observation) > 200 else observation
            
            formatted_steps.append(step_info)
        
        return formatted_steps
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        return list(self.agents.keys())
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health and report agent models"""
        status = {
            'vector_store': self.vector_store is not None,
            'vector_store_type': settings.VECTOR_STORE_TYPE,
            'agents_loaded': len(self.agents) > 0,
            'agent_models': {}
        }

        # Report model for each agent
        if 'theory' in self.agents:
            status['agent_models']['theory'] = getattr(self.theory_llm, 'model_name', getattr(self.theory_llm, 'model', 'unknown'))

        if 'research' in self.agents:
            model_name = getattr(self.research_llm, 'model', getattr(self.research_llm, 'model_name', 'unknown'))
            if self.research_llm == self.theory_llm:
                model_name += " (fallback)"
            status['agent_models']['research'] = model_name

        if 'implementation' in self.agents:
            model_name = getattr(self.implementation_llm, 'model', getattr(self.implementation_llm, 'model_name', 'unknown'))
            if self.implementation_llm == self.theory_llm:
                model_name += " (fallback)"
            status['agent_models']['implementation'] = model_name
        
        # Report routing LLM
        status['agent_models']['routing'] = getattr(self.routing_llm, 'model_name', getattr(self.routing_llm, 'model', 'unknown'))

        # Test GPT-4 connection (theory agent)
        try:
            response = self.theory_llm.invoke([HumanMessage(content="test")])
            status['gpt4_connection'] = bool(response.content)
        except:
            status['gpt4_connection'] = False
        
        # Test routing LLM connection (GPT-4o-mini)
        try:
            response = self.routing_llm.invoke([HumanMessage(content="test")])
            status['routing_llm_connection'] = bool(response.content)
        except:
            status['routing_llm_connection'] = False
        
        # Test Ollama connection (research agent)
        if self.research_llm and self.research_llm != self.theory_llm:
            try:
                response = self.research_llm.invoke([HumanMessage(content="test")])
                status['ollama_connection'] = bool(response.content)
            except Exception as e:
                print(f"❌ Ollama connection test failed with an error: {e}")
                status['ollama_connection'] = False
        else:
            status['ollama_connection'] = self.research_llm == self.theory_llm # True if fallback is active
            
        # Test Claude connection (implementation agent)
        if self.implementation_llm and self.implementation_llm != self.theory_llm:
            try:
                response = self.implementation_llm.invoke([HumanMessage(content="test")])
                status['claude_connection'] = bool(response.content)
            except Exception as e:
                print(f"❌ Claude connection test failed with an error: {e}")
                status['claude_connection'] = False
        else:
            status['claude_connection'] = self.implementation_llm == self.theory_llm # True if fallback is active
            
        # Overall LLM health
        status['all_llms_configured'] = status['gpt4_connection'] and status['ollama_connection'] and status['claude_connection']
        
        # Test each agent
        for agent_name in self.agents:
            status[f'agent_{agent_name}'] = agent_name in self.agents
        
        return status
    
    def _update_chat_history(self, query: str, response: str, agent_name: str):
        """
        Update internal chat history with new conversation
        
        Args:
            query: User question
            response: Agent response
            agent_name: Name of the agent that processed the query
        """
        # Add human message
        self.chat_history.append(HumanMessage(content=query))
        
        # Add AI message with agent information
        ai_content = f"[{agent_name.upper()} Agent] {response}"
        self.chat_history.append(AIMessage(content=ai_content))
        
        # Maintain history length limit
        if len(self.chat_history) > self.max_history_length:
            # Remove oldest messages (keep recent conversations)
            self.chat_history = self.chat_history[-self.max_history_length:]
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Get formatted chat history for display
        
        Returns:
            List of formatted chat history entries
        """
        formatted_history = []
        for i, message in enumerate(self.chat_history):
            formatted_history.append({
                'index': i,
                'type': 'human' if isinstance(message, HumanMessage) else 'ai',
                'content': message.content,
                'timestamp': datetime.now().isoformat()  # Note: Real timestamps would need to be stored
            })
        return formatted_history
    
    def clear_chat_history(self):
        """Clear all chat history"""
        self.chat_history = []
        print("✅ Chat history cleared")
    
    def get_chat_history_summary(self) -> Dict[str, Any]:
        """
        Get summary of chat history
        
        Returns:
            Dictionary with chat history statistics
        """
        return {
            'total_messages': len(self.chat_history),
            'human_messages': len([m for m in self.chat_history if isinstance(m, HumanMessage)]),
            'ai_messages': len([m for m in self.chat_history if isinstance(m, AIMessage)]),
            'max_history_length': self.max_history_length,
            'history_full': len(self.chat_history) >= self.max_history_length
        }
    
    def save_chat_history_to_file(self, filepath: str = None):
        """
        Save chat history to JSON file
        
        Args:
            filepath: Path to save file (optional, defaults to chat_history.json)
        """
        if filepath is None:
            filepath = "chat_history.json"
        
        try:
            # Convert messages to serializable format
            history_data = []
            for message in self.chat_history:
                history_data.append({
                    'type': 'human' if isinstance(message, HumanMessage) else 'ai',
                    'content': message.content,
                    'timestamp': datetime.now().isoformat()
                })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Chat history saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving chat history: {e}")
            return False
    
    def load_chat_history_from_file(self, filepath: str = None):
        """
        Load chat history from JSON file
        
        Args:
            filepath: Path to load file (optional, defaults to chat_history.json)
        """
        if filepath is None:
            filepath = "chat_history.json"
        
        if not os.path.exists(filepath):
            print(f"⚠️ Chat history file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            # Convert back to LangChain messages
            self.chat_history = []
            for entry in history_data:
                if entry['type'] == 'human':
                    self.chat_history.append(HumanMessage(content=entry['content']))
                else:
                    self.chat_history.append(AIMessage(content=entry['content']))
            
            print(f"✅ Chat history loaded from {filepath} ({len(self.chat_history)} messages)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading chat history: {e}")
            return False

# Convenience function for easy import
def create_langchain_ml_agents(auto_upsert: bool = False):
    """
    Factory function to create LangChain ML agents
    
    Args:
        auto_upsert: Whether to automatically upsert documents to Pinecone during initialization
                    Default is False - documents will only be upserted when explicitly requested
    """
    return LangChainMLAgents(auto_upsert=auto_upsert) 