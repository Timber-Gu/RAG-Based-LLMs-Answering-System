"""
LangChain-based Multi-Agent System for ML Q&A
Clean implementation using LangChain's agent framework
"""
from typing import List, Dict, Any, Optional, Tuple
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_tool_calling_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

# Enhanced memory imports
from langchain.memory import (
    ConversationSummaryBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    VectorStoreRetrieverMemory
)
from langchain.memory.chat_message_histories import (
    ChatMessageHistory,
    SQLChatMessageHistory
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import TimeWeightedVectorStoreRetriever
import os
import json
import time
import re
from datetime import datetime
from collections import defaultdict, Counter
import sqlite3
from pathlib import Path
import re
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

class AdvancedMemoryManager:
    """
    Advanced memory management system using LangChain components
    Supports multiple memory strategies with automatic optimization
    """
    
    def __init__(self, llm, vector_store=None, session_id: str = "default"):
        self.llm = llm
        self.vector_store = vector_store
        self.session_id = session_id
        self.db_path = Path("data/memory_sessions.db")
        self._ensure_database()
        
        # Initialize different memory types
        self.memory_configs = {
            "summary_buffer": {
                "type": "summary_buffer",
                "max_tokens": 2000,
                "return_messages": True
            },
            "buffer_window": {
                "type": "buffer_window", 
                "k": 10,
                "return_messages": True
            },
            "token_buffer": {
                "type": "token_buffer",
                "max_tokens": 1500,
                "return_messages": True
            }
        }
        
        # Active memory instance
        self.active_memory = None
        self.memory_type = "summary_buffer"  # Default strategy
        
    def _ensure_database(self):
        """Ensure SQLite database exists for persistent memory storage"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    message_content TEXT NOT NULL,
                    agent_name TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    token_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    summary_content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    token_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()
    
    def create_memory(self, memory_type: str = None, **kwargs) -> Any:
        """
        Create and configure memory instance based on type
        
        Args:
            memory_type: Type of memory to create ('summary_buffer', 'buffer_window', 'token_buffer', 'vector_retriever')
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured memory instance
        """
        if memory_type is None:
            memory_type = self.memory_type
            
        # Create persistent chat message history
        chat_history = SQLChatMessageHistory(
            session_id=self.session_id,
            connection_string=f"sqlite:///{self.db_path}"
        )
        
        if memory_type == "summary_buffer":
            max_tokens = kwargs.get("max_tokens", self.memory_configs["summary_buffer"]["max_tokens"])
            return ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=max_tokens,
                chat_memory=chat_history,
                return_messages=kwargs.get("return_messages", True),
                memory_key="chat_history"
            )
            
        elif memory_type == "buffer_window":
            k = kwargs.get("k", self.memory_configs["buffer_window"]["k"])
            return ConversationBufferWindowMemory(
                k=k,
                chat_memory=chat_history,
                return_messages=kwargs.get("return_messages", True),
                memory_key="chat_history"
            )
            
        elif memory_type == "token_buffer":
            max_tokens = kwargs.get("max_tokens", self.memory_configs["token_buffer"]["max_tokens"])
            return ConversationTokenBufferMemory(
                llm=self.llm,
                max_token_limit=max_tokens,
                chat_memory=chat_history,
                return_messages=kwargs.get("return_messages", True),
                memory_key="chat_history"
            )
            
        elif memory_type == "vector_retriever" and self.vector_store:
            # Create time-weighted vector store retriever for semantic memory
            retriever = TimeWeightedVectorStoreRetriever(
                vectorstore=self.vector_store,
                decay_rate=kwargs.get("decay_rate", -0.0005),  # Decay factor for time weighting
                k=kwargs.get("k", 4)
            )
            
            return VectorStoreRetrieverMemory(
                retriever=retriever,
                memory_key="chat_history",
                return_messages=kwargs.get("return_messages", True)
            )
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")
    
    def get_optimized_memory(self, conversation_length: int = 0, token_count: int = 0) -> Any:
        """
        Automatically select optimal memory strategy based on conversation metrics
        
        Args:
            conversation_length: Number of exchanges in conversation
            token_count: Estimated token count of conversation
            
        Returns:
            Optimized memory instance
        """
        # Decision logic for memory type selection
        if token_count > 3000 or conversation_length > 15:
            # Long conversations benefit from summarization
            memory_type = "summary_buffer"
            config = {"max_tokens": 2500}
        elif conversation_length > 8:
            # Medium conversations use token buffer
            memory_type = "token_buffer"
            config = {"max_tokens": 1500}
        else:
            # Short conversations can use window buffer
            memory_type = "buffer_window"
            config = {"k": min(conversation_length + 2, 10)}
        
        print(f"🧠 Selected {memory_type} memory for conversation (length={conversation_length}, tokens≈{token_count})")
        return self.create_memory(memory_type, **config)
    
    def create_hybrid_memory(self, primary_type: str = "summary_buffer", 
                           enable_vector_memory: bool = True) -> Dict[str, Any]:
        """
        Create hybrid memory system combining multiple strategies
        
        Args:
            primary_type: Primary memory type for conversation flow
            enable_vector_memory: Whether to include vector-based semantic memory
            
        Returns:
            Dictionary of memory instances for different purposes
        """
        memories = {
            "primary": self.create_memory(primary_type),
            "backup": self.create_memory("buffer_window", k=5)  # Fallback memory
        }
        
        if enable_vector_memory and self.vector_store:
            memories["semantic"] = self.create_memory("vector_retriever", k=3)
            
        return memories
    
    def save_conversation_state(self, memory_instance, agent_name: str = None):
        """
        Persist current conversation state to database
        
        Args:
            memory_instance: Active memory instance to save
            agent_name: Name of agent handling the conversation
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Get current memory state
                if hasattr(memory_instance, 'chat_memory'):
                    messages = memory_instance.chat_memory.messages
                    for msg in messages[-5:]:  # Save last 5 messages
                        cursor.execute("""
                            INSERT INTO chat_sessions 
                            (session_id, message_type, message_content, agent_name, token_count)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            self.session_id,
                            "human" if isinstance(msg, HumanMessage) else "ai",
                            msg.content,
                            agent_name,
                            len(msg.content.split()) * 1.3  # Rough token estimate
                        ))
                
                # Save summary if available
                if hasattr(memory_instance, 'moving_summary_buffer') and memory_instance.moving_summary_buffer:
                    cursor.execute("""
                        INSERT INTO memory_summaries 
                        (session_id, summary_content, token_count)
                        VALUES (?, ?, ?)
                    """, (
                        self.session_id,
                        memory_instance.moving_summary_buffer,
                        len(memory_instance.moving_summary_buffer.split()) * 1.3
                    ))
                
                conn.commit()
                print(f"💾 Conversation state saved for session: {self.session_id}")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not save conversation state: {e}")
    
    def load_conversation_state(self, memory_instance, limit: int = 20):
        """
        Load previous conversation state from database
        
        Args:
            memory_instance: Memory instance to load state into
            limit: Maximum number of messages to load
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Load recent messages
                cursor.execute("""
                    SELECT message_type, message_content, agent_name, timestamp
                    FROM chat_sessions 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (self.session_id, limit))
                
                messages = cursor.fetchall()
                
                # Restore messages to memory (in reverse order)
                for msg_type, content, agent_name, timestamp in reversed(messages):
                    if msg_type == "human":
                        memory_instance.chat_memory.add_user_message(content)
                    else:
                        # Include agent info in AI message
                        ai_content = f"[{agent_name.upper()} Agent] {content}" if agent_name else content
                        memory_instance.chat_memory.add_ai_message(ai_content)
                
                print(f"📂 Loaded {len(messages)} messages for session: {self.session_id}")
                return len(messages)
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load conversation state: {e}")
            return 0
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage statistics
        
        Returns:
            Dictionary with memory metrics and recommendations
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Get session statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_messages,
                        SUM(CASE WHEN message_type = 'human' THEN 1 ELSE 0 END) as human_messages,
                        SUM(CASE WHEN message_type = 'ai' THEN 1 ELSE 0 END) as ai_messages,
                        SUM(token_count) as estimated_tokens,
                        MIN(timestamp) as first_message,
                        MAX(timestamp) as last_message
                    FROM chat_sessions
                    WHERE session_id = ?
                """, (self.session_id,))
                
                stats = cursor.fetchone()
                
                # Calculate conversation duration
                if stats[4] and stats[5]:  # first_message and last_message
                    first_time = datetime.fromisoformat(stats[4])
                    last_time = datetime.fromisoformat(stats[5])
                    duration_minutes = (last_time - first_time).total_seconds() / 60
                else:
                    duration_minutes = 0
                
                # Get memory efficiency recommendations
                total_messages = stats[0] or 0
                estimated_tokens = stats[3] or 0
                
                recommendations = []
                if estimated_tokens > 2500:
                    recommendations.append("Consider using ConversationSummaryBufferMemory")
                if total_messages > 20:
                    recommendations.append("Enable conversation archiving")
                if duration_minutes > 30:
                    recommendations.append("Consider conversation segmentation")
                
                return {
                    "session_id": self.session_id,
                    "total_messages": total_messages,
                    "human_messages": stats[1] or 0,
                    "ai_messages": stats[2] or 0,
                    "estimated_tokens": estimated_tokens,
                    "conversation_duration_minutes": round(duration_minutes, 2),
                    "recommended_memory_type": self._get_recommended_memory_type(total_messages, estimated_tokens),
                    "optimization_recommendations": recommendations,
                    "memory_efficiency_score": self._calculate_efficiency_score(total_messages, estimated_tokens)
                }
                
        except Exception as e:
            print(f"⚠️ Warning: Could not retrieve memory statistics: {e}")
            return {"error": str(e)}
    
    def _get_recommended_memory_type(self, message_count: int, token_count: int) -> str:
        """Recommend optimal memory type based on conversation metrics"""
        if token_count > 3000 or message_count > 15:
            return "summary_buffer"
        elif message_count > 8:
            return "token_buffer"
        else:
            return "buffer_window"
    
    def _calculate_efficiency_score(self, message_count: int, token_count: int) -> float:
        """Calculate memory efficiency score (0-100)"""
        if message_count == 0:
            return 100.0
        
        # Calculate tokens per message ratio
        tokens_per_message = token_count / message_count if message_count > 0 else 0
        
        # Ideal range: 50-150 tokens per message
        if 50 <= tokens_per_message <= 150:
            base_score = 100
        elif tokens_per_message < 50:
            base_score = 70 + (tokens_per_message / 50) * 30
        else:
            base_score = max(20, 100 - (tokens_per_message - 150) / 10)
        
        # Adjust for conversation length
        if message_count > 25:
            base_score *= 0.9  # Penalty for very long conversations without summarization
        
        return round(min(100, max(0, base_score)), 2)

class RetrievalResult:
    """Container for retrieval results with metadata"""
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, source: str):
        self.content = content
        self.metadata = metadata
        self.score = score
        self.source = source  # Which retrieval strategy produced this result
        self.id = metadata.get('id', '')

class BaseRetriever:
    """Base class for all retrieval strategies"""
    def __init__(self, name: str):
        self.name = name
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve documents for a query"""
        raise NotImplementedError

class EnhancedSimilarityRetriever(BaseRetriever):
    """Enhanced similarity search with better scoring"""
    def __init__(self, pinecone_index, namespace: str = "__default__"):
        super().__init__("enhanced_similarity")
        self.pinecone_index = pinecone_index
        self.namespace = namespace
    
    def retrieve(self, query: str, top_k: int = 8) -> List[RetrievalResult]:
        """Enhanced similarity search with normalized scoring"""
        try:
            query_payload = {
                "inputs": {"text": query},
                "top_k": top_k
            }
            
            search_results = self.pinecone_index.search(query=query_payload, namespace=self.namespace)
            hits = search_results.get('result', {}).get('hits', [])
            
            results = []
            for hit in hits:
                fields = hit.get('fields', {})
                
                # Normalize score to 0-1 range
                raw_score = hit.get('_score', 0)
                normalized_score = min(max(raw_score, 0), 1)
                
                result = RetrievalResult(
                    content=fields.get('text', ''),
                    metadata={
                        'title': fields.get('title', 'Unknown'),
                        'source': fields.get('source', ''),
                        'type': fields.get('type', 'unknown'),
                        'authors': fields.get('authors', ''),
                        'categories': fields.get('categories', ''),
                        'id': hit.get('_id', ''),
                        'chunk_index': fields.get('chunk_index'),
                        'parent_paper_id': fields.get('parent_paper_id', '')
                    },
                    score=normalized_score,
                    source=self.name
                )
                results.append(result)
            
            return results
        except Exception as e:
            print(f"❌ Enhanced similarity search error: {e}")
            return []

class SelfQueryRetriever(BaseRetriever):
    """Self-query retriever that extracts metadata filters from natural language"""
    def __init__(self, pinecone_index, routing_llm, namespace: str = "__default__"):
        super().__init__("self_query")
        self.pinecone_index = pinecone_index
        self.routing_llm = routing_llm
        self.namespace = namespace
    
    def _extract_metadata_filters(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Extract semantic query and metadata filters from natural language"""
        filter_prompt = f"""
        Analyze this query and extract structured search components:
        Query: "{query}"
        
        Extract:
        1. Core semantic query (remove metadata references)
        2. Metadata filters (if any)
        
        Common patterns to detect:
        - "recent papers" → filter by year/date
        - "papers by [author]" → filter by authors
        - "[specific domain] papers" → filter by categories
        - "implementation/code papers" → filter by type
        
        Respond in JSON format:
        {{
            "semantic_query": "core research question without metadata",
            "filters": {{
                "authors": "author name if mentioned",
                "categories": "domain if mentioned", 
                "type": "paper type if mentioned",
                "year_range": "recent/old if mentioned"
            }}
        }}
        
        If no specific filters, leave filters empty.
        """
        
        try:
            response = self.routing_llm.invoke([HumanMessage(content=filter_prompt)])
            
            # Extract JSON from response
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            parsed = json.loads(content)
            return parsed.get('semantic_query', query), parsed.get('filters', {})
            
        except Exception as e:
            print(f"⚠️ Self-query parsing error: {e}, using original query")
            return query, {}
    
    def retrieve(self, query: str, top_k: int = 6) -> List[RetrievalResult]:
        """Retrieve with extracted metadata filters"""
        semantic_query, filters = self._extract_metadata_filters(query)
        
        try:
            # Build Pinecone filter from extracted metadata
            pinecone_filter = {}
            if filters.get('authors'):
                pinecone_filter['authors'] = {'$regex': f".*{filters['authors']}.*"}
            if filters.get('categories'):
                pinecone_filter['categories'] = {'$regex': f".*{filters['categories']}.*"}
            if filters.get('type'):
                pinecone_filter['type'] = {'$eq': filters['type']}
            
            query_payload = {
                "inputs": {"text": semantic_query},
                "top_k": top_k
            }
            
            # Add filter if any metadata filters were extracted
            if pinecone_filter:
                query_payload["filter"] = pinecone_filter
            
            search_results = self.pinecone_index.search(query=query_payload, namespace=self.namespace)
            hits = search_results.get('result', {}).get('hits', [])
            
            results = []
            for hit in hits:
                fields = hit.get('fields', {})
                
                # Boost score if query contains metadata that matches
                base_score = hit.get('_score', 0)
                boost = 1.0
                
                # Boost for exact metadata matches
                if filters.get('authors') and filters['authors'].lower() in fields.get('authors', '').lower():
                    boost += 0.1
                if filters.get('categories') and filters['categories'].lower() in fields.get('categories', '').lower():
                    boost += 0.1
                
                boosted_score = min(base_score * boost, 1.0)
                
                result = RetrievalResult(
                    content=fields.get('text', ''),
                    metadata={
                        'title': fields.get('title', 'Unknown'),
                        'source': fields.get('source', ''),
                        'type': fields.get('type', 'unknown'),
                        'authors': fields.get('authors', ''),
                        'categories': fields.get('categories', ''),
                        'id': hit.get('_id', ''),
                        'chunk_index': fields.get('chunk_index'),
                        'parent_paper_id': fields.get('parent_paper_id', '')
                    },
                    score=boosted_score,
                    source=f"{self.name}_{filters if filters else 'no_filter'}"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Self-query retrieval error: {e}")
            return []

class ParentDocumentRetriever(BaseRetriever):
    """Retrieves small chunks but returns parent document context"""
    def __init__(self, pinecone_index, namespace: str = "__default__"):
        super().__init__("parent_document")
        self.pinecone_index = pinecone_index
        self.namespace = namespace
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Find relevant chunks, then return parent documents"""
        try:
            # First, search for chunks
            query_payload = {
                "inputs": {"text": query},
                "top_k": top_k * 2  # Get more chunks to find diverse parents
            }
            
            search_results = self.pinecone_index.search(query=query_payload, namespace=self.namespace)
            hits = search_results.get('result', {}).get('hits', [])
            
            # Group by parent document
            parent_groups = defaultdict(list)
            for hit in hits:
                fields = hit.get('fields', {})
                parent_id = fields.get('parent_paper_id', hit.get('_id', ''))
                parent_groups[parent_id].append(hit)
            
            # For each parent, get the best chunks and combine them
            results = []
            for parent_id, chunks in list(parent_groups.items())[:top_k]:
                # Sort chunks by score and chunk_index
                chunks.sort(key=lambda x: (-x.get('_score', 0), 
                           int(x.get('fields', {}).get('chunk_index', 0))))
                
                # Combine content from best chunks of this parent
                combined_content = []
                metadata = None
                total_score = 0
                
                for chunk in chunks[:3]:  # Take up to 3 best chunks per parent
                    fields = chunk.get('fields', {})
                    combined_content.append(fields.get('text', ''))
                    total_score += chunk.get('_score', 0)
                    
                    if metadata is None:  # Use metadata from first chunk
                        metadata = {
                            'title': fields.get('title', 'Unknown'),
                            'source': fields.get('source', ''),
                            'type': fields.get('type', 'unknown'),
                            'authors': fields.get('authors', ''),
                            'categories': fields.get('categories', ''),
                            'id': parent_id,
                            'total_chunks': fields.get('total_chunks', 1),
                            'chunks_included': len(chunks)
                        }
                
                # Average score across chunks
                avg_score = total_score / len(chunks) if chunks else 0
                
                result = RetrievalResult(
                    content="\n\n".join(combined_content),
                    metadata=metadata,
                    score=avg_score,
                    source=f"{self.name}_chunks_{len(chunks)}"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Parent document retrieval error: {e}")
            return []

class EnsembleFusion:
    """Combines results from multiple retrievers using Reciprocal Rank Fusion"""
    
    @staticmethod
    def reciprocal_rank_fusion(results_lists: List[List[RetrievalResult]], 
                             weights: List[float] = None, 
                             k: int = 60) -> List[RetrievalResult]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion
        
        Args:
            results_lists: List of result lists from different retrievers
            weights: Optional weights for each retriever (default: equal)
            k: RRF parameter (typical value: 60)
        """
        if not results_lists:
            return []
        
        if weights is None:
            weights = [1.0] * len(results_lists)
        
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Calculate RRF scores for all documents
        doc_scores = defaultdict(float)
        doc_data = {}  # Store document data
        
        for results, weight in zip(results_lists, weights):
            for rank, result in enumerate(results, 1):
                doc_id = result.id or result.content[:50]  # Use content as fallback ID
                
                # RRF score: weight / (k + rank)
                rrf_score = weight / (k + rank)
                doc_scores[doc_id] += rrf_score
                
                # Store document data (prefer higher scoring versions)
                if doc_id not in doc_data or result.score > doc_data[doc_id].score:
                    doc_data[doc_id] = result
        
        # Sort by combined RRF score
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return sorted results with RRF scores
        fused_results = []
        for doc_id, rrf_score in ranked_docs:
            result = doc_data[doc_id]
            # Update score to RRF score
            result.score = rrf_score
            result.source = f"ensemble_rrf_{result.source}"
            fused_results.append(result)
        
        return fused_results

class ContextualCompressor:
    """LLM-based contextual compression for retrieved documents"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def compress(self, results: List[RetrievalResult], query: str, 
                max_docs: int = 4, max_chars_per_doc: int = 1500) -> List[RetrievalResult]:
        """
        Compress retrieved documents to most relevant content
        
        Args:
            results: Retrieved documents to compress
            query: Original query for relevance filtering
            max_docs: Maximum number of documents to return
            max_chars_per_doc: Maximum characters per document
        """
        if not results:
            return []
        
        try:
            compressed_results = []
            
            for result in results[:max_docs * 2]:  # Process more than needed for better selection
                compressed_content = self._compress_single_document(
                    result.content, query, max_chars_per_doc
                )
                
                if compressed_content and len(compressed_content.strip()) > 50:  # Reduced threshold for less aggressive filtering
                    # Update the result with compressed content
                    compressed_result = RetrievalResult(
                        content=compressed_content,
                        metadata=result.metadata,
                        score=result.score,
                        source=f"compressed_{result.source}"
                    )
                    compressed_results.append(compressed_result)
                else:
                    print(f"⚠️ Skipped result due to insufficient content after compression: {len(compressed_content.strip()) if compressed_content else 0} chars")
            
            # Return top results after compression
            return compressed_results[:max_docs]
            
        except Exception as e:
            print(f"❌ Contextual compression error: {e}")
            # Fallback: return original results truncated
            return [RetrievalResult(
                content=r.content[:max_chars_per_doc] + "..." if len(r.content) > max_chars_per_doc else r.content,
                metadata=r.metadata,
                score=r.score,
                source=f"truncated_{r.source}"
            ) for r in results[:max_docs]]
    
    def _compress_single_document(self, content: str, query: str, max_chars: int) -> str:
        """Compress a single document using LLM"""
        if len(content) <= max_chars:
            return content
        
        compression_prompt = f"""
        You are an expert at extracting the most relevant information from academic papers.
        
        **Task**: Extract and summarize the most relevant parts of this document for the given query.
        
        **Query**: {query}
        
        **Document Content**:
        {content[:3000]}{'...' if len(content) > 3000 else ''}
        
        **Instructions**:
        1. Extract the most relevant sections that directly answer or relate to the query
        2. Preserve important technical details, equations, and specific findings
        3. Maintain key citations and paper references
        4. Keep essential context for understanding
        5. Stay under {max_chars} characters
        6. Use clear, structured format
        
        **Output**: Only the compressed relevant content, no meta-commentary.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=compression_prompt)])
            compressed = response.content.strip()
            
            # Ensure we don't exceed character limit
            if len(compressed) > max_chars:
                compressed = compressed[:max_chars-3] + "..."
            
            return compressed
            
        except Exception as e:
            print(f"⚠️ LLM compression failed: {e}")
            # Fallback: intelligent truncation
            return content[:max_chars-3] + "..."

class EnsembleRetriever:
    """Main ensemble retriever that coordinates multiple strategies"""
    
    def __init__(self, pinecone_index, routing_llm, compression_llm=None, namespace: str = "__default__"):
        self.pinecone_index = pinecone_index
        self.namespace = namespace
        
        # Initialize individual retrievers
        self.enhanced_similarity = EnhancedSimilarityRetriever(pinecone_index, namespace)
        self.self_query = SelfQueryRetriever(pinecone_index, routing_llm, namespace)
        self.parent_document = ParentDocumentRetriever(pinecone_index, namespace)
        
        # Initialize fusion and compression
        self.fusion = EnsembleFusion()
        self.compressor = ContextualCompressor(compression_llm) if compression_llm else None
        
    def retrieve(self, query: str, use_compression: bool = True, 
                max_final_docs: int = 4) -> List[RetrievalResult]:
        """
        Main retrieval method using ensemble approach
        
        Args:
            query: Search query
            use_compression: Whether to apply contextual compression
            max_final_docs: Maximum number of final documents to return
        """
        try:
            # Step 1: Retrieve from multiple sources
            print(f"🔍 Starting ensemble retrieval for: {query[:50]}...")
            
            similarity_results = self.enhanced_similarity.retrieve(query, top_k=8)  # Increased for better coverage
            self_query_results = self.self_query.retrieve(query, top_k=6)          # Increased for better coverage
            parent_results = self.parent_document.retrieve(query, top_k=5)          # Increased for better coverage
            
            print(f"📊 Retrieved: {len(similarity_results)} similarity, {len(self_query_results)} self-query, {len(parent_results)} parent docs")
            
            # Debug: Check if we have enough total results
            total_results = len(similarity_results) + len(self_query_results) + len(parent_results)
            print(f"🔢 Total individual results before fusion: {total_results}")
            
            # Step 2: Ensemble fusion using RRF
            all_results = [similarity_results, self_query_results, parent_results]
            weights = [0.4, 0.3, 0.3]  # Prioritize similarity, balance others
            
            fused_results = self.fusion.reciprocal_rank_fusion(all_results, weights)
            print(f"🔄 Fused to {len(fused_results)} unique documents")
            
            # Step 3: Contextual compression (optional)
            if use_compression and self.compressor and fused_results:
                from src.config import settings
                compressed_results = self.compressor.compress(
                    fused_results, query, max_docs=max_final_docs, 
                    max_chars_per_doc=settings.ENSEMBLE_COMPRESSION_MAX_CHARS
                )
                print(f"📝 Compressed to {len(compressed_results)} final documents")
                
                # Fallback: If compression was too aggressive and we have < 2 results, use uncompressed
                if len(compressed_results) < 2 and len(fused_results) >= 2:
                    print(f"⚠️ Compression too aggressive ({len(compressed_results)} results), using uncompressed fallback")
                    final_results = fused_results[:max_final_docs]
                    for result in final_results:
                        if len(result.content) > settings.ENSEMBLE_COMPRESSION_MAX_CHARS:
                            result.content = result.content[:settings.ENSEMBLE_COMPRESSION_MAX_CHARS-3] + "..."
                            result.source = f"truncated_{result.source}"
                    return final_results
                
                return compressed_results
            else:
                # Return top results without compression
                final_results = fused_results[:max_final_docs]
                for result in final_results:
                    # Truncate if too long
                    if len(result.content) > 2000:
                        result.content = result.content[:1997] + "..."
                        result.source = f"truncated_{result.source}"
                
                print(f"📄 Returning {len(final_results)} documents without compression")
                return final_results
            
        except Exception as e:
            print(f"❌ Ensemble retrieval error: {e}")
            # Fallback to basic similarity search
            return self.enhanced_similarity.retrieve(query, top_k=max_final_docs)

class LangChainMLAgents:
    """LangChain-based multi-agent system for ML Q&A"""
    
    def __init__(self, auto_upsert: bool = False, memory_type: str = "summary_buffer", 
                 session_id: str = "default", enable_persistent_memory: bool = True):
        """
        Initialize LangChain ML Agents with enhanced memory management
        
        Args:
            auto_upsert: Whether to automatically upsert documents to Pinecone during initialization
            memory_type: Type of memory to use ('summary_buffer', 'buffer_window', 'token_buffer', 'auto')
            session_id: Unique identifier for this conversation session
            enable_persistent_memory: Whether to enable database persistence for memory
        """
        self.auto_upsert = auto_upsert
        self.session_id = session_id
        self.enable_persistent_memory = enable_persistent_memory
        
        # Legacy chat history for backward compatibility
        self.chat_history = []  # List of HumanMessage and AIMessage objects
        self.max_history_length = 20  # Maximum number of messages to keep in memory
        
        # Enhanced memory system
        self.memory_type = memory_type
        self.memory_manager = None  # Will be initialized after LLMs are set up
        self.agent_memories = {}  # Store memory instances for each agent
        
        # Initialize routing LLM for semantic query analysis (cheap but effective)
        self.routing_llm = ChatOpenAI(
            model="gpt-4o-mini",  # Very cheap and fast for classification tasks
            temperature=0.0,  # Low temperature for consistent routing decisions
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=512  # Small limit for routing decisions
        )
        
        # Initialize multiple LLMs for different agents
        
        # Theory agent initialization with logging
        try:
            self.theory_llm = ChatOpenAI(
                model=settings.THEORY_MODEL,
                temperature=settings.AGENT_TEMPERATURE,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=3000  # Reduced to prevent context overflow while still allowing detailed explanations
            )
            print(f"✅ {settings.THEORY_MODEL} initialized for theory agent")
        except Exception as e:
            print(f"❌ Theory agent initialization failed: {e}")
            raise e
        
        # Claude 3.5 Sonnet for research agent (optimal for research analysis)
        self.research_llm = None
        if settings.ANTHROPIC_API_KEY:
            try:
                self.research_llm = ChatAnthropic(
                    model="claude-3-5-sonnet-20241022",  # Latest Claude 3.5 Sonnet
                    temperature=0.1,  # Lower temperature for research accuracy
                    anthropic_api_key=settings.ANTHROPIC_API_KEY,
                    max_tokens=4096 
                )
                print("✅ Claude 3.5 Sonnet initialized for research agent")
            except Exception as e:
                print(f"⚠️ Claude 3.5 Sonnet initialization failed: {e}, using GPT-4 for research agent")
                self.research_llm = self.theory_llm
        else:
            print("⚠️ ANTHROPIC_API_KEY not found, using GPT-4 for research agent")
            self.research_llm = self.theory_llm
        
        # Claude for implementation agent
        self.implementation_llm = None
        if settings.ANTHROPIC_API_KEY:
            try:
                self.implementation_llm = ChatAnthropic(
                    model=settings.IMPLEMENTATION_MODEL,
                    temperature=settings.AGENT_TEMPERATURE,
                    anthropic_api_key=settings.ANTHROPIC_API_KEY,
                    max_tokens=8192  # Higher limit for complete code implementations
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
        self.ensemble_retriever = None
        self.agents = {}
        self._setup_vector_store()
        self._setup_memory_manager()  # Initialize memory system after vector store
        self._setup_agents()
    
    def _clean_text_for_pinecone(self, text):
        """Clean and normalize text for Pinecone storage"""
        if not text:
            return ""
        raw = str(text)
        raw = re.sub(r"-\s*\n\s*", "", raw)
        raw = raw.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
        raw = re.sub(r"\s+", " ", raw)
        return raw.strip()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def _manage_context_window(self, messages_content: str, max_completion_tokens: int = 3000) -> str:
        """Ensure context fits within model limits"""
        # Estimate tokens for different components
        base_system_prompt_tokens = 200  # Estimated system prompt size
        function_tokens = 150  # Estimated function descriptions
        safety_buffer = 100  # Safety margin
        
        # Calculate available tokens for context
        model_limit = 8192  # GPT-4 context limit
        available_for_context = model_limit - max_completion_tokens - base_system_prompt_tokens - function_tokens - safety_buffer
        
        estimated_tokens = self._estimate_tokens(messages_content)
        
        if estimated_tokens > available_for_context:
            # Truncate context to fit
            target_chars = available_for_context * 4  # Convert back to characters
            truncated_content = messages_content[:target_chars]
            print(f"⚠️ Context truncated: {estimated_tokens} → {self._estimate_tokens(truncated_content)} tokens")
            return truncated_content
        
        return messages_content
    
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
                # Verify existing index is configured for integrated embeddings; recreate if not
                needs_recreate = False
                try:
                    details = pc.describe_index(index_name)
                    embed_cfg = None
                    if isinstance(details, dict):
                        embed_cfg = details.get('spec', {}).get('embed') or details.get('embed')
                    else:
                        embed_cfg = getattr(details, 'embed', None)
                        if embed_cfg is None:
                            spec = getattr(details, 'spec', None)
                            if spec is not None:
                                embed_cfg = getattr(spec, 'embed', None)
                    if not embed_cfg:
                        needs_recreate = True
                    else:
                        model_ok = (getattr(embed_cfg, 'model', None) or (embed_cfg.get('model') if isinstance(embed_cfg, dict) else None)) == 'llama-text-embed-v2'
                        field_map = getattr(embed_cfg, 'field_map', None) or (embed_cfg.get('field_map', {}) if isinstance(embed_cfg, dict) else {})
                        field_ok = isinstance(field_map, dict) and field_map.get('text') == 'text'
                        if not (model_ok and field_ok):
                            needs_recreate = True
                except Exception as e:
                    print(f"⚠️ Could not verify Pinecone index config: {e}")
                    needs_recreate = False

                if needs_recreate:
                    print(f"♻️ Recreating index '{index_name}' for integrated embeddings configuration...")
                    try:
                        pc.delete_index(index_name)
                    except Exception as e:
                        print(f"⚠️ Failed to delete existing index '{index_name}': {e}")
                    pc.create_index_for_model(
                        name=index_name,
                        cloud="aws",
                        region="us-east-1",
                        embed={
                            "model": "llama-text-embed-v2",
                            "field_map": {"text": "text"}
                        }
                    )
                    print(f"✅ Recreated Pinecone index '{index_name}' with integrated llama-text-embed-v2")
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
            
            # Initialize ensemble retriever after Pinecone setup
            self._setup_ensemble_retriever()
            
        except ImportError as e:
            print(f"❌ Pinecone dependencies error: {e}")
            print("💡 Please install pinecone dependencies: pip install pinecone langchain-pinecone")
            raise e
    
    def _setup_ensemble_retriever(self):
        """Setup ensemble retriever with multiple retrieval strategies"""
        try:
            if not settings.USE_ENSEMBLE_RETRIEVAL:
                print("📝 Ensemble retrieval disabled in configuration")
                self.ensemble_retriever = None
                return
                
            if self.pinecone_index and self.routing_llm:
                # Use research LLM for compression (better for contextual understanding)
                compression_llm = self.research_llm if self.research_llm else self.routing_llm
                
                self.ensemble_retriever = EnsembleRetriever(
                    pinecone_index=self.pinecone_index,
                    routing_llm=self.routing_llm,
                    compression_llm=compression_llm,
                    namespace="__default__"
                )
                print("🎯 Ensemble retriever initialized with multi-strategy support")
                print("   → Enhanced similarity search")
                print("   → Self-query with metadata filtering")
                print("   → Parent document retrieval")
                print("   → Reciprocal Rank Fusion")
                print("   → LLM-based contextual compression")
                print(f"   → Settings: compression={settings.ENSEMBLE_USE_COMPRESSION}, max_docs={settings.ENSEMBLE_MAX_DOCS}")
            else:
                print("⚠️ Cannot initialize ensemble retriever - missing dependencies")
                self.ensemble_retriever = None
                
        except Exception as e:
            print(f"❌ Error setting up ensemble retriever: {e}")
            self.ensemble_retriever = None
    
    def _setup_memory_manager(self):
        """Initialize the advanced memory management system"""
        try:
            if not self.enable_persistent_memory:
                print("📝 Persistent memory disabled - using legacy memory system")
                return
                
            # Initialize memory manager with routing LLM (cheaper for summarization)
            self.memory_manager = AdvancedMemoryManager(
                llm=self.routing_llm,  # Use cheap LLM for memory operations
                vector_store=self.vector_store,
                session_id=self.session_id
            )
            
            # Create optimized memory instances for each agent type
            if self.memory_type == "auto":
                # Get conversation statistics to auto-select memory type
                stats = self.memory_manager.get_memory_statistics()
                if stats.get("error"):
                    # First time setup - use default
                    primary_memory = self.memory_manager.create_memory("summary_buffer")
                else:
                    # Use statistics to optimize memory
                    primary_memory = self.memory_manager.get_optimized_memory(
                        stats.get("total_messages", 0),
                        stats.get("estimated_tokens", 0)
                    )
            else:
                # Use specified memory type
                primary_memory = self.memory_manager.create_memory(self.memory_type)
            
            # Create agent-specific memory instances
            self.agent_memories = {
                "research": self.memory_manager.create_memory(
                    "summary_buffer", max_tokens=2500
                ),
                "theory": self.memory_manager.create_memory(
                    "token_buffer", max_tokens=2000  # Theory needs detailed context
                ),
                "implementation": self.memory_manager.create_memory(
                    "buffer_window", k=8  # Implementation benefits from recent context
                )
            }
            
            # Add semantic memory if vector store is available
            if self.vector_store:
                try:
                    semantic_memory = self.memory_manager.create_memory(
                        "vector_retriever", k=4, decay_rate=-0.0005
                    )
                    self.agent_memories["semantic"] = semantic_memory
                    print("🔮 Semantic memory enabled with vector store integration")
                except Exception as e:
                    print(f"⚠️ Warning: Could not initialize semantic memory: {e}")
            
            # Load existing conversation state if available
            for agent_name, memory in self.agent_memories.items():
                if agent_name != "semantic":  # Skip semantic memory for loading
                    loaded_count = self.memory_manager.load_conversation_state(memory)
                    if loaded_count > 0:
                        print(f"📂 Loaded {loaded_count} messages for {agent_name} agent")
            
            print("✅ Advanced memory management system initialized successfully")
            print(f"   🧠 Memory type: {self.memory_type}")
            print(f"   🆔 Session ID: {self.session_id}")
            print(f"   💾 Persistent storage: {'enabled' if self.enable_persistent_memory else 'disabled'}")
            
        except Exception as e:
            print(f"❌ Error setting up memory manager: {e}")
            print("⚠️ Falling back to legacy memory system")
            self.memory_manager = None
    
    def _truncate_content_for_pinecone(self, content: str, max_bytes: int = 20000) -> str:
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
                        "chunk_index": int(doc.metadata.get('chunk_index', 0)),
                        "total_chunks": int(doc.metadata.get('total_chunks', 1)),
                        "parent_paper_id": str(doc.metadata.get('parent_paper_id', ''))[:50]
                    })
                
                # Estimate total record size
                record_size = sum(len(str(v).encode('utf-8')) for v in record.values())
                if record_size > 25000:
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
                    
                    # Apply context window management to prevent overflow
                    full_response = "\n\n".join(results)
                    managed_response = self._manage_context_window(full_response, max_completion_tokens=3000)
                    return managed_response
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
                    
                    # Apply context window management to prevent overflow
                    full_response = "\n\n".join(results)
                    managed_response = self._manage_context_window(full_response, max_completion_tokens=3000)
                    return managed_response
                    
            except Exception as e:
                return f"Error searching knowledge base: {e}"
        
        return Tool(
            name="search_knowledge",
            description="Search the ML/DL knowledge base for relevant papers and information",
            func=search_knowledge
        )
    
    def _create_ensemble_rag_tool(self):
        """Create advanced ensemble RAG tool with multiple retrieval strategies and contextual compression"""
        def search_knowledge_ensemble(query: str) -> str:
            """Advanced search using ensemble retrieval with contextual compression"""
            if not self.ensemble_retriever:
                # Fallback to basic RAG if ensemble not available
                basic_tool = self._create_rag_tool()
                return basic_tool.func(query)
            
            try:
                # Use ensemble retriever with configurable settings
                results = self.ensemble_retriever.retrieve(
                    query=query,
                    use_compression=settings.ENSEMBLE_USE_COMPRESSION,
                    max_final_docs=settings.ENSEMBLE_MAX_DOCS
                )
                
                if not results:
                    return "No relevant information found in knowledge base using ensemble retrieval"
                
                # Format results with enhanced metadata
                formatted_results = []
                for i, result in enumerate(results, 1):
                    title = result.metadata.get('title', 'Unknown')
                    authors = result.metadata.get('authors', '')
                    source_info = result.source
                    score = result.score
                    
                    # Add retrieval source information for transparency
                    header = f"**Result {i}** (Source: {title})"
                    if authors:
                        header += f"\nAuthors: {authors[:100]}{'...' if len(authors) > 100 else ''}"
                    header += f"\nRetrieval Score: {score:.3f} | Strategy: {source_info}"
                    
                    content = result.content
                    formatted_results.append(f"{header}\n\nContent: {content}")
                
                # Apply context window management to prevent overflow
                full_response = "\n\n" + "="*50 + "\n\n".join(formatted_results)
                managed_response = self._manage_context_window(full_response, max_completion_tokens=3000)
                
                return managed_response
                
            except Exception as e:
                print(f"❌ Ensemble RAG error: {e}")
                # Fallback to basic RAG
                basic_tool = self._create_rag_tool()
                return f"Ensemble retrieval failed, using basic search: {basic_tool.func(query)}"
        
        return Tool(
            name="search_knowledge_ensemble",
            description="Advanced search using ensemble retrieval with multiple strategies (similarity, self-query, parent docs) and LLM-based contextual compression for optimal results",
            func=search_knowledge_ensemble
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
        """Setup specialized LangChain agents with enhanced memory integration"""
        
        # Create advanced ensemble RAG tool (with fallback to basic)
        rag_tool = self._create_ensemble_rag_tool()
        
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
            
            **Enhanced Memory Context:**
            - You have access to conversation history through advanced memory management
            - Previous exchanges are summarized intelligently to maintain context while optimizing token usage
            - You can reference earlier parts of the conversation naturally
            - Your responses build upon the accumulated conversation context
            
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
            - Always aim to give comprehensive, helpful research-oriented answers
            - Reference previous conversation points when relevant to provide continuity"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Use appropriate agent creation method based on LLM type
        if isinstance(self.research_llm, ChatAnthropic):
            # For Claude models, use generic tool calling agent
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
        
        # Get memory for research agent
        research_memory = self.agent_memories.get("research") if self.memory_manager else None
        
        self.agents['research'] = AgentExecutor(
            agent=research_agent,
            tools=[rag_tool],
            memory=research_memory,  # Enhanced memory integration
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True  # Enable intermediate steps capture
        )
        
        # Theory Agent with Chain of Thoughts
        theory_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Theory Agent specializing in rigorous mathematical explanations of ML/DL concepts using Chain of Thoughts reasoning.

            🚨 **MANDATORY: ALL mathematical expressions MUST be wrapped in LaTeX delimiters ($ or $$). NEVER write mathematical formulas as plain text.**

            **Knowledge Integration Strategy:**
            - ALWAYS search the knowledge base first using the search_knowledge tool
            - Combine RAG results with your extensive pretrained knowledge for complete answers
            - If RAG provides relevant information: Use it as foundation and enhance with pretrained knowledge
            - If RAG is insufficient or empty: Rely entirely on your pretrained knowledge
            - Never limit your answer to only RAG content - always provide comprehensive explanations

            **Mathematical Rigor Requirements:**
            - Provide formal mathematical definitions for all concepts
            - Include complete derivations with step-by-step mathematical reasoning
            - Use precise mathematical notation and terminology
            - Show mathematical relationships between concepts
            - Provide complexity analysis where applicable
            - Include convergence proofs and theoretical guarantees when relevant

            **CRITICAL: Mathematical Formula Formatting**
            
            **MANDATORY LaTeX Requirements:**
            1. ALL mathematical expressions MUST be wrapped in LaTeX delimiters
            2. Use $...$ for inline math, $$...$$ for display equations
            3. NEVER write mathematical formulas as plain text
            4. Use proper LaTeX syntax for all mathematical notation
            
            **Essential LaTeX Formatting Rules:**
            - Bold vectors: Use \\mathbf command for vectors and matrices
            - Subscripts: Use underscore notation 
            - Superscripts: Use caret notation
            - Functions: Use \\text command for function names
            - Fractions: Use \\frac command for fractions
            - Square roots: Use \\sqrt command 
            - Summations: Use \\sum command with proper limits
            - Concatenation: Use brackets or text commands
            - Matrix spaces: Use \\mathbb and proper notation
            - Complexity: Use \\mathcal for complexity notation
            
            **Transformation Examples:**
            - Transform plain text formulas to proper LaTeX format
            - Wrap ALL mathematical expressions in $$ delimiters
            - Use proper mathematical notation throughout
            - Ensure vectors are bold and functions use text command
            
            **Critical Rules:**
            - ALWAYS use double dollar signs $$ for display equations
            - ALWAYS use \\mathbf for vectors and matrices
            - ALWAYS use proper subscript/superscript syntax
            - NEVER write math formulas without LaTeX delimiters

            **Response Format:**

            🔍 **Knowledge Search:**
            [Search knowledge base for relevant information]

            📖 **Knowledge Integration:**
            [State your approach: "Combining RAG findings with pretrained knowledge" OR "Using pretrained knowledge (RAG insufficient)"]

            🧠 **Mathematical Chain of Thoughts Analysis:**

            **Step 1: Formal Problem Statement**
            [Mathematical definition of the problem with precise notation]

            **Step 2: Mathematical Foundations**
            [Define all mathematical objects, spaces, and operations involved]

            **Step 3: Core Mathematical Framework**
            [Present the complete mathematical formulation using $$equation$$ blocks for all formulas]

            **Step 4: Detailed Mathematical Derivation**
            [Step-by-step mathematical derivation with each equation in $$equation$$ format and justification for each step]

            **Step 5: Theoretical Properties**
            [Analyze computational complexity, convergence properties, theoretical guarantees]

            **Step 6: Mathematical Connections**
            [Show relationships to other mathematical concepts and ML/DL algorithms]

            **Step 7: Practical Implications**
            [Bridge theory to implementation considerations]

            📝 **Mathematical Summary:**
            [Concise mathematical summary with key equations and results]

            **Core Principles:**
            - Maintain mathematical rigor throughout all explanations
            - Use proper mathematical notation consistently
            - Show complete derivations, not just final results
            - Explain mathematical intuition behind formulas
            - Connect abstract mathematics to concrete ML/DL applications
            - Use well-formatted LaTeX for all mathematical expressions
            - Provide theoretical analysis alongside practical insights"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        theory_agent = create_openai_functions_agent(
            llm=self.theory_llm,  # Use GPT-4 for theory/math tasks
            tools=[rag_tool, cot_tool],
            prompt=theory_prompt
        )
        
        # Get memory for theory agent
        theory_memory = self.agent_memories.get("theory") if self.memory_manager else None
        
        self.agents['theory'] = AgentExecutor(
            agent=theory_agent,
            tools=[rag_tool, cot_tool],
            memory=theory_memory,  # Enhanced memory integration
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
        
        # Get memory for implementation agent
        implementation_memory = self.agent_memories.get("implementation") if self.memory_manager else None
        
        self.agents['implementation'] = AgentExecutor(
            agent=implementation_agent,
            tools=[rag_tool],
            memory=implementation_memory,  # Enhanced memory integration
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
    
    def process_query(self, query: str, chat_history: List = None, show_thinking: bool = True, 
                     save_to_memory: bool = True) -> Dict[str, Any]:
        """
        Process query using appropriate LangChain agent with enhanced memory support
        
        Args:
            query: User question to process
            chat_history: Optional external chat history (for compatibility - mostly ignored with new memory system)
            show_thinking: Whether to display agent thinking process
            save_to_memory: Whether to save conversation to persistent memory
            
        Returns:
            Dict containing response, agent info, and thinking process
        """
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
            # For enhanced memory system, agents handle memory automatically
            # For legacy system, maintain backward compatibility
            if self.memory_manager and agent.memory:
                # Enhanced memory system - agents manage their own memory
                result = agent.invoke({'input': query})
                
                # Get memory statistics for optimization insights
                if show_thinking:
                    memory_stats = self.memory_manager.get_memory_statistics()
                    if not memory_stats.get("error"):
                        print(f"💭 Memory efficiency: {memory_stats.get('memory_efficiency_score', 0):.1f}%")
                        if memory_stats.get('optimization_recommendations'):
                            print(f"🔧 Recommendations: {', '.join(memory_stats['optimization_recommendations'])}")
            else:
                # Legacy memory system for backward compatibility
                if chat_history is None:
                    chat_history = self.chat_history.copy()
                
                # Add current query to legacy chat history
                current_human_message = HumanMessage(content=query)
                
                # Process with LangChain agent using legacy chat history
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
            
            # Save conversation state with enhanced memory system
            if save_to_memory and self.memory_manager and agent.memory:
                try:
                    self.memory_manager.save_conversation_state(agent.memory, agent_name)
                except Exception as e:
                    print(f"⚠️ Warning: Could not save conversation to enhanced memory: {e}")
            
            # Update legacy chat history for backward compatibility
            if not self.memory_manager or not agent.memory:
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
    
    # Enhanced Memory Management Methods
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage statistics and optimization recommendations
        
        Returns:
            Dictionary with memory metrics, efficiency scores, and recommendations
        """
        if not self.memory_manager:
            return {
                "error": "Enhanced memory system not initialized",
                "using_legacy_memory": True,
                "legacy_message_count": len(self.chat_history)
            }
        
        return self.memory_manager.get_memory_statistics()
    
    def optimize_memory_settings(self) -> Dict[str, Any]:
        """
        Automatically optimize memory settings based on conversation history
        
        Returns:
            Dictionary with optimization results and recommendations
        """
        if not self.memory_manager:
            return {"error": "Enhanced memory system not initialized"}
        
        try:
            stats = self.memory_manager.get_memory_statistics()
            
            if stats.get("error"):
                return {"error": "Could not retrieve memory statistics"}
            
            # Get current conversation metrics
            total_messages = stats.get("total_messages", 0)
            estimated_tokens = stats.get("estimated_tokens", 0)
            efficiency_score = stats.get("memory_efficiency_score", 0)
            
            recommendations = []
            optimizations_applied = []
            
            # Apply optimizations based on statistics
            if efficiency_score < 60:
                # Recreate memory instances with optimized settings
                for agent_name in ["research", "theory", "implementation"]:
                    if agent_name in self.agent_memories:
                        # Get optimized memory for this agent
                        new_memory = self.memory_manager.get_optimized_memory(
                            total_messages, estimated_tokens
                        )
                        
                        # Update agent memory
                        if agent_name in self.agents:
                            self.agents[agent_name].memory = new_memory
                            self.agent_memories[agent_name] = new_memory
                            
                        optimizations_applied.append(f"Optimized {agent_name} agent memory")
                        
                recommendations.append("Memory optimization applied automatically")
            
            if estimated_tokens > 3000:
                recommendations.append("Consider conversation segmentation for long sessions")
            
            if total_messages > 30:
                recommendations.append("Archive old conversations to improve performance")
            
            return {
                "optimization_successful": len(optimizations_applied) > 0,
                "optimizations_applied": optimizations_applied,
                "recommendations": recommendations,
                "efficiency_score": efficiency_score,
                "conversation_metrics": {
                    "total_messages": total_messages,
                    "estimated_tokens": estimated_tokens,
                    "duration_minutes": stats.get("conversation_duration_minutes", 0)
                }
            }
            
        except Exception as e:
            return {"error": f"Optimization failed: {e}"}
    
    def switch_memory_type(self, new_memory_type: str, preserve_history: bool = True) -> Dict[str, Any]:
        """
        Switch to a different memory type while optionally preserving conversation history
        
        Args:
            new_memory_type: New memory type ('summary_buffer', 'buffer_window', 'token_buffer')
            preserve_history: Whether to preserve existing conversation history
            
        Returns:
            Dictionary with switch results
        """
        if not self.memory_manager:
            return {"error": "Enhanced memory system not initialized"}
        
        try:
            # Save current conversation state if preserving history
            old_memories = {}
            if preserve_history:
                for agent_name, memory in self.agent_memories.items():
                    if agent_name != "semantic":
                        self.memory_manager.save_conversation_state(memory, agent_name)
                        old_memories[agent_name] = memory
            
            # Create new memory instances
            new_memories = {}
            for agent_name in ["research", "theory", "implementation"]:
                new_memory = self.memory_manager.create_memory(new_memory_type)
                new_memories[agent_name] = new_memory
                
                # Load history if preserving
                if preserve_history and agent_name in old_memories:
                    self.memory_manager.load_conversation_state(new_memory)
                
                # Update agent memory
                if agent_name in self.agents:
                    self.agents[agent_name].memory = new_memory
            
            # Update memory instances
            self.agent_memories.update(new_memories)
            self.memory_type = new_memory_type
            
            print(f"✅ Switched to {new_memory_type} memory type")
            print(f"📜 History preserved: {'Yes' if preserve_history else 'No'}")
            
            return {
                "switch_successful": True,
                "new_memory_type": new_memory_type,
                "history_preserved": preserve_history,
                "agents_updated": list(new_memories.keys())
            }
            
        except Exception as e:
            return {"error": f"Memory type switch failed: {e}"}
    
    def create_memory_snapshot(self, snapshot_name: str = None) -> Dict[str, Any]:
        """
        Create a snapshot of current memory state for backup/restoration
        
        Args:
            snapshot_name: Optional name for the snapshot (defaults to timestamp)
            
        Returns:
            Dictionary with snapshot information
        """
        if not self.memory_manager:
            return {"error": "Enhanced memory system not initialized"}
        
        try:
            if snapshot_name is None:
                snapshot_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save current state for all agents
            saved_agents = []
            for agent_name, memory in self.agent_memories.items():
                if agent_name != "semantic":
                    self.memory_manager.save_conversation_state(memory, f"{agent_name}_{snapshot_name}")
                    saved_agents.append(agent_name)
            
            return {
                "snapshot_successful": True,
                "snapshot_name": snapshot_name,
                "agents_saved": saved_agents,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Snapshot creation failed: {e}"}
    
    def clear_enhanced_memory(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Clear all enhanced memory data (requires confirmation)
        
        Args:
            confirm: Must be True to actually clear memory
            
        Returns:
            Dictionary with clear operation results
        """
        if not confirm:
            return {
                "error": "Memory clear requires confirmation",
                "instruction": "Call with confirm=True to actually clear memory"
            }
        
        if not self.memory_manager:
            return {"error": "Enhanced memory system not initialized"}
        
        try:
            # Clear agent memories
            cleared_agents = []
            for agent_name, memory in self.agent_memories.items():
                if hasattr(memory, 'clear'):
                    memory.clear()
                elif hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'clear'):
                    memory.chat_memory.clear()
                cleared_agents.append(agent_name)
            
            # Clear legacy memory as well
            self.clear_chat_history()
            
            print("🗑️ All memory data cleared")
            
            return {
                "clear_successful": True,
                "agents_cleared": cleared_agents,
                "legacy_memory_cleared": True
            }
            
        except Exception as e:
            return {"error": f"Memory clear failed: {e}"}

# Convenience function for easy import
def create_langchain_ml_agents(auto_upsert: bool = False, memory_type: str = "summary_buffer",
                              session_id: str = "default", enable_persistent_memory: bool = True):
    """
    Factory function to create LangChain ML agents with enhanced memory management
    
    Args:
        auto_upsert: Whether to automatically upsert documents to Pinecone during initialization
        memory_type: Type of memory to use ('summary_buffer', 'buffer_window', 'token_buffer', 'auto')
        session_id: Unique identifier for this conversation session
        enable_persistent_memory: Whether to enable database persistence for memory
    """
    return LangChainMLAgents(
        auto_upsert=auto_upsert,
        memory_type=memory_type,
        session_id=session_id,
        enable_persistent_memory=enable_persistent_memory
    ) 