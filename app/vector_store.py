"""
Vector Store Module - Handles document embeddings and similarity search using Pinecone
"""
import os
import time
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from app.config import settings


class VectorStore:
    """Manages vector database operations with Pinecone"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        
        # Get embedding dimension based on model
        self.dimension = self._get_embedding_dimension()
        
        # Create index if it doesn't exist
        self._ensure_index_exists()
        
        # Get reference to the index
        self.index = self.pc.Index(settings.pinecone_index_name)
    
    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension based on the model"""
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimension_map.get(settings.embedding_model, 1536)
    
    def _ensure_index_exists(self):
        """Create the Pinecone index if it doesn't exist"""
        try:
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
        except Exception:
            # Fallback for different Pinecone SDK versions
            existing_indexes = self.pc.list_indexes()
            index_names = list(existing_indexes.names()) if hasattr(existing_indexes, 'names') else [str(idx) for idx in existing_indexes]
        
        if settings.pinecone_index_name not in index_names:
            self.pc.create_index(
                name=settings.pinecone_index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(settings.pinecone_index_name).status['ready']:
                time.sleep(1)
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for the given text using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        response = self.openai_client.embeddings.create(
            model=settings.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts in batch
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # OpenAI has a limit, so batch in chunks of 100
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.openai_client.embeddings.create(
                model=settings.embedding_model,
                input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])
        
        return all_embeddings
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of document dicts with 'id', 'text', and 'metadata'
            
        Returns:
            Status information about the operation
        """
        if not documents:
            return {"status": "error", "message": "No documents to add"}
        
        # Extract texts and create embeddings in batch
        texts = [doc["text"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        
        # Create embeddings
        embeddings = self.create_embeddings_batch(texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, doc in enumerate(documents):
            # Pinecone metadata must be flat (no nested objects)
            metadata = {
                "text": doc["text"][:1000],  # Store truncated text in metadata
                "source": doc["metadata"].get("source", "unknown"),
                "chunk_index": doc["metadata"].get("chunk_index", 0),
                "total_chunks": doc["metadata"].get("total_chunks", 1),
            }
            vectors.append({
                "id": ids[i],
                "values": embeddings[i],
                "metadata": metadata
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        return {
            "status": "success",
            "documents_added": len(documents),
            "index_name": settings.pinecone_index_name
        }
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector store
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        # Create query embedding
        query_embedding = self.create_embedding(query)
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "text": match.metadata.get("text", ""),
                "metadata": {
                    "source": match.metadata.get("source", "unknown"),
                    "chunk_index": match.metadata.get("chunk_index", 0),
                    "total_chunks": match.metadata.get("total_chunks", 1),
                },
                "score": match.score,
                "id": match.id
            })
        
        return formatted_results
    
    def delete_by_source(self, source_filename: str) -> Dict[str, Any]:
        """
        Delete all documents from a specific source file
        
        Args:
            source_filename: The filename to delete documents for
            
        Returns:
            Status information
        """
        try:
            # Pinecone requires querying first to get IDs, then deleting
            # Use a dummy query to find all vectors with this source
            dummy_embedding = [0.0] * self.dimension
            
            # Query for vectors with this source
            results = self.index.query(
                vector=dummy_embedding,
                top_k=10000,
                include_metadata=True,
                filter={"source": {"$eq": source_filename}}
            )
            
            if results.matches:
                ids_to_delete = [match.id for match in results.matches]
                
                # Delete in batches
                batch_size = 1000
                for i in range(0, len(ids_to_delete), batch_size):
                    batch = ids_to_delete[i:i + batch_size]
                    self.index.delete(ids=batch)
                
                return {
                    "status": "success",
                    "deleted_count": len(ids_to_delete),
                    "source": source_filename
                }
            else:
                return {
                    "status": "success",
                    "deleted_count": 0,
                    "message": "No documents found with this source"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_all_sources(self) -> List[str]:
        """
        Get a list of all unique source filenames in the index
        
        Returns:
            List of source filenames
        """
        try:
            # Query with a dummy vector to get all vectors
            dummy_embedding = [0.0] * self.dimension
            
            results = self.index.query(
                vector=dummy_embedding,
                top_k=10000,
                include_metadata=True
            )
            
            sources = set()
            for match in results.matches:
                source = match.metadata.get("source")
                if source:
                    sources.add(source)
            
            return list(sources)
        except Exception as e:
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index
        
        Returns:
            Index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            sources = self.get_all_sources()
            
            return {
                "total_documents": stats.total_vector_count,
                "total_sources": len(sources),
                "sources": sources,
                "collection_name": settings.pinecone_index_name
            }
        except Exception as e:
            return {
                "total_documents": 0,
                "total_sources": 0,
                "sources": [],
                "collection_name": settings.pinecone_index_name,
                "error": str(e)
            }
    
    def clear_collection(self) -> Dict[str, Any]:
        """
        Clear all documents from the index
        
        Returns:
            Status information
        """
        try:
            # Delete all vectors in the index
            self.index.delete(delete_all=True)
            
            return {
                "status": "success",
                "message": "Index cleared successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


# Create global vector store instance
vector_store = VectorStore()
