"""
Vector store management using Pinecone
"""
import os
import json
from typing import List, Dict, Any, Optional
import pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

from ..config import settings

class VectorStore:
    """Manage vector storage and retrieval using Pinecone"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index"""
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            
            # Create index if it doesn't exist
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if settings.PINECONE_INDEX_NAME not in existing_indexes:
                pc.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=self.embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-west-2'
                    )
                )
                print(f"Created Pinecone index: {settings.PINECONE_INDEX_NAME}")
            
            self.index = pc.Index(settings.PINECONE_INDEX_NAME)
            print(f"Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")
            
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for texts"""
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Add documents to vector store"""
        if not documents:
            print("No documents to add")
            return
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare texts for embedding
        texts = []
        for doc in documents:
            # Combine title and content for better embeddings
            text = f"{doc.get('title', '')}\n{doc.get('content', '')}"
            texts.append(text)
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vector = {
                'id': doc['id'],
                'values': embedding.tolist(),
                'metadata': {
                    'title': doc.get('title', ''),
                    'content': doc.get('content', ''),
                    'source': doc.get('source', ''),
                    'authors': doc.get('authors', []),
                    'categories': doc.get('categories', []),
                    'type': doc.get('type', 'unknown')
                }
            }
            vectors.append(vector)
        
        # Upload in batches
        for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading vectors"):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                print(f"Error uploading batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"Successfully added {len(documents)} documents to vector store")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            documents = []
            for match in results['matches']:
                doc = {
                    'id': match['id'],
                    'score': match['score'],
                    'title': match['metadata'].get('title', ''),
                    'content': match['metadata'].get('content', ''),
                    'source': match['metadata'].get('source', ''),
                    'authors': match['metadata'].get('authors', []),
                    'categories': match['metadata'].get('categories', []),
                    'type': match['metadata'].get('type', 'unknown')
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}
    
    def clear_index(self):
        """Clear all vectors from index"""
        try:
            self.index.delete(delete_all=True)
            print("Cleared all vectors from index")
        except Exception as e:
            print(f"Error clearing index: {e}")

class RAGPipeline:
    """RAG pipeline combining vector store and generation"""
    
    def __init__(self):
        self.vector_store = VectorStore()
    
    def build_knowledge_base(self, knowledge_file: str = None):
        """Build knowledge base from processed papers"""
        if knowledge_file is None:
            knowledge_file = settings.KNOWLEDGE_BASE_FILE
        
        if not os.path.exists(knowledge_file):
            print(f"Knowledge base file not found: {knowledge_file}")
            return
        
        # Load knowledge chunks
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"Loaded {len(chunks)} knowledge chunks")
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        # Print stats
        stats = self.vector_store.get_index_stats()
        print(f"Vector store stats: {stats}")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant context for query"""
        return self.vector_store.search(query, top_k=top_k)
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context"""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_part = f"Document {i}:\n"
            context_part += f"Title: {doc['title']}\n"
            context_part += f"Content: {doc['content'][:500]}...\n"
            context_part += f"Source: {doc['source']}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)

if __name__ == "__main__":
    # Test the RAG pipeline
    pipeline = RAGPipeline()
    
    # Build knowledge base
    pipeline.build_knowledge_base()
    
    # Test retrieval
    query = "What are convolutional neural networks?"
    context_docs = pipeline.retrieve_context(query)
    context = pipeline.format_context(context_docs)
    
    print(f"Query: {query}")
    print(f"Retrieved Context:\n{context}") 