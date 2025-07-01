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
from ..config import settings

class LangChainMLAgents:
    """LangChain-based multi-agent system for ML Q&A"""
    
    def __init__(self):
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
        self.agents = {}
        self._setup_vector_store()
        self._setup_agents()
    
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
                            'category': item.get('category', 'general')
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
        """Setup Pinecone vector store with hosted embeddings using LangChain"""
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
                print(f"🔄 Creating Pinecone index with hosted embeddings: {index_name}")
                
                # Create index with integrated inference using llama-text-embed-v2
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
                print(f"✅ Pinecone index '{index_name}' created with hosted llama-text-embed-v2")
            else:
                print(f"✅ Using existing Pinecone index: {index_name}")
            
            # Create a custom embedding class that integrates with Pinecone's hosted inference
            class PineconeHostedEmbeddings:
                """Custom embeddings class using Pinecone's hosted inference"""
                def __init__(self, index_name):
                    self.index_name = index_name
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
            
            # Create the custom embeddings instance
            embeddings = PineconeHostedEmbeddings(index_name)
            
            # Create LangChain vector store using hosted embeddings
            if documents:
                print(f"🔄 Setting up LangChain PineconeVectorStore with {len(documents)} documents...")
                
                # For hosted inference, we need to handle the upsert differently
                # We'll use LangChain's from_documents but with special handling for hosted embeddings
                
                # First, let's manually upsert the documents with proper hosted embedding format
                index = pc.Index(index_name)
                
                # Prepare and upsert documents with hosted embeddings (correct API format)
                records = []
                for i, doc in enumerate(documents):
                    # Clean text for Pinecone
                    def clean_text(text):
                        """Clean text for Pinecone storage"""
                        return text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
                    
                    title = clean_text(str(doc.metadata.get('title', ''))[:80])
                    source = clean_text(str(doc.metadata.get('source', ''))[:100])
                    category = str(doc.metadata.get('category', 'general'))
                    
                    # Correct format: all fields at top level, not nested in metadata
                    record = {
                        "_id": f"doc_{i}",  # Use _id as per Pinecone docs
                        "text": doc.page_content,  # This will be embedded by Pinecone
                        "title": title,  # Top-level metadata fields
                        "source": source,
                        "category": category
                    }
                    records.append(record)
                
                # Upsert in batches using correct API format
                batch_size = 96  # Max for text upserts with hosted embeddings
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    # Correct API: namespace first, then records list
                    index.upsert_records("__default__", batch)
                    print(f"✅ Uploaded batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}")
                
                # Now create the LangChain vector store pointing to the same index
                self.vector_store = PineconeVectorStore(
                    index_name=index_name,
                    embedding=embeddings,
                    text_key="text"  # Field containing the text content
                )
                
                print(f"✅ LangChain PineconeVectorStore created with {len(documents)} documents using hosted embeddings")
            else:
                # Create empty vector store
                self.vector_store = PineconeVectorStore(
                    index_name=index_name,
                    embedding=embeddings,
                    text_key="text"
                )
                print("✅ LangChain PineconeVectorStore created (empty) with hosted embeddings")
            
            # Store the raw index for advanced operations if needed
            self.pinecone_index = pc.Index(index_name)
            
        except ImportError as e:
            print(f"❌ Pinecone dependencies error: {e}")
            print("💡 Please install pinecone dependencies: pip install pinecone langchain-pinecone")
            raise e
    
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
                        "top_k": 3
                    }
                    
                    search_results = self.pinecone_index.search(query=query_payload, namespace="__default__")
                    
                    if not search_results.get('matches'):
                        return "No relevant information found in knowledge base"
                    
                    results = []
                    for match in search_results['matches']:
                        # Get metadata from top-level fields (correct format)
                        title = match.get('title', 'Unknown')
                        source = match.get('source', '')
                        # Get a preview of the content
                        text = match.get('text', '')
                        content = text[:500] + "..." if len(text) > 500 else text
                        score = match.get('score', 0)
                        results.append(f"Source: {title} (Score: {score:.3f})\nContent: {content}")
                    
                    return "\n\n".join(results)
                else:
                    # Fallback to LangChain's method (though it won't work well with hosted embeddings)
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
            
            Always search the knowledge base first before providing answers."""),
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

            Your role is to provide clear, step-by-step explanations using the following structured approach:

            **Chain of Thoughts Framework:**
            1. **Problem Understanding**: First, clearly restate what is being asked
            2. **Knowledge Retrieval**: Search the knowledge base for relevant theoretical content
            3. **Conceptual Foundation**: Establish the fundamental concepts needed
            4. **Step-by-Step Analysis**: Break down the problem into logical steps
            5. **Mathematical Derivation**: Show detailed mathematical work when applicable
            6. **Intuitive Explanation**: Provide intuitive understanding of the concepts
            7. **Connections**: Link to related concepts and broader context
            8. **Summary**: Conclude with key takeaways

            **Response Format:**
            Always structure your responses using this format:

            🤔 **Thinking Process:**
            [Brief overview of your reasoning approach]

            📚 **Knowledge Base Search:**
            [Use the search_knowledge tool to find relevant information]

            🧠 **Step-by-Step Analysis:**
            
            **Step 1: Problem Understanding**
            [Clearly restate the question and identify what needs to be explained]
            
            **Step 2: Fundamental Concepts**
            [Define key terms and establish foundational knowledge]
            
            **Step 3: Mathematical Framework** (if applicable)
            [Present relevant equations, formulas, or mathematical structures]
            
            **Step 4: Detailed Explanation**
            [Provide thorough explanation with reasoning]
            
            **Step 5: Intuitive Understanding**
            [Explain the "why" and "how" in intuitive terms]
            
            **Step 6: Practical Implications**
            [Discuss how this applies in practice]

            🔗 **Connections & Context:**
            [Link to related concepts and broader ML/DL context]

            📝 **Key Takeaways:**
            [Summarize the most important points]

            **Guidelines:**
            - Always search the knowledge base first before providing explanations
            - Show your reasoning process explicitly
            - Use mathematical notation when helpful (LaTeX format)
            - Provide both formal and intuitive explanations
            - Connect abstract concepts to concrete examples
            - Acknowledge limitations or assumptions when relevant
            - Build explanations from simple to complex concepts"""),
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
            
            Search for code examples and implementation details in the knowledge base."""),
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
    
    def route_query(self, query: str) -> str:
        """Simple routing logic to determine best agent"""
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
    
    def process_query(self, query: str, chat_history: List = None, show_thinking: bool = True) -> Dict[str, Any]:
        """Process query using appropriate LangChain agent with optional thinking process display"""
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

        # Test GPT-4 connection (theory agent)
        try:
            response = self.theory_llm.invoke([HumanMessage(content="test")])
            status['gpt4_connection'] = bool(response.content)
        except:
            status['gpt4_connection'] = False
        
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

# Convenience function for easy import
def create_langchain_ml_agents():
    """Factory function to create LangChain ML agents"""
    return LangChainMLAgents() 