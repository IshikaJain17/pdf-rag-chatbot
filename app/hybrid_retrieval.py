"""
Hybrid Retrieval System - Inspired by RAG-Anything
Combines vector similarity search with keyword-based BM25 search
"""
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from app.config import settings


class BM25Retriever:
    """
    BM25 keyword-based retriever for hybrid search
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0
        self.term_frequencies: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: freq}
        self.doc_frequencies: Dict[str, int] = {}  # term -> num docs containing term
        self.vocab: set = set()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        # Remove very short tokens
        return [t for t in tokens if len(t) > 2]
    
    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None):
        """Add a document to the BM25 index"""
        tokens = self._tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.documents[doc_id] = {
            "text": text,
            "metadata": metadata or {},
            "tokens": tokens
        }
        
        # Update term frequencies
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            self.vocab.add(token)
            if token not in self.term_frequencies:
                self.term_frequencies[token] = {}
            self.term_frequencies[token][doc_id] = count
            
            if doc_id not in self.doc_frequencies.get(token, {}):
                self.doc_frequencies[token] = self.doc_frequencies.get(token, 0) + 1
        
        # Update average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add multiple documents"""
        for doc in documents:
            self.add_document(
                doc_id=doc["id"],
                text=doc["text"],
                metadata=doc.get("metadata", {})
            )
    
    def _bm25_score(self, query_tokens: List[str], doc_id: str) -> float:
        """Calculate BM25 score for a document"""
        score = 0.0
        doc_length = self.doc_lengths[doc_id]
        n_docs = len(self.documents)
        
        for token in query_tokens:
            if token not in self.term_frequencies:
                continue
            
            # Document frequency
            df = self.doc_frequencies.get(token, 0)
            if df == 0:
                continue
            
            # Term frequency in document
            tf = self.term_frequencies[token].get(doc_id, 0)
            if tf == 0:
                continue
            
            # IDF component
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            
            # TF component with length normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            )
            
            score += idf * tf_norm
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents using BM25"""
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Calculate scores for all documents
        scores = []
        for doc_id in self.documents:
            score = self._bm25_score(query_tokens, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        results = []
        for doc_id, score in scores[:top_k]:
            doc = self.documents[doc_id]
            results.append({
                "id": doc_id,
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": score,
                "retrieval_type": "bm25"
            })
        
        return results
    
    def clear(self):
        """Clear all documents"""
        self.documents.clear()
        self.doc_lengths.clear()
        self.avg_doc_length = 0
        self.term_frequencies.clear()
        self.doc_frequencies.clear()
        self.vocab.clear()


class HybridRetriever:
    """
    Hybrid retrieval combining vector search and BM25
    Inspired by RAG-Anything's modality-aware retrieval
    """
    
    def __init__(self, vector_store, bm25_weight: float = 0.3, vector_weight: float = 0.7):
        self.vector_store = vector_store
        self.bm25_retriever = BM25Retriever()
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to both retrievers"""
        # Add to BM25
        self.bm25_retriever.add_documents(documents)
        # Vector store handles its own addition
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize scores to 0-1 range"""
        if not results:
            return results
        
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for r in results:
                r["normalized_score"] = 1.0
        else:
            for r in results:
                r["normalized_score"] = (r["score"] - min_score) / (max_score - min_score)
        
        return results
    
    def _reciprocal_rank_fusion(
        self, 
        vector_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion
        RRF score = sum(1 / (k + rank))
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_data: Dict[str, Dict[str, Any]] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            rrf_scores[doc_id] += self.vector_weight * (1 / (k + rank + 1))
            if doc_id not in doc_data:
                doc_data[doc_id] = result
                doc_data[doc_id]["vector_score"] = result.get("score", 0)
                doc_data[doc_id]["vector_rank"] = rank + 1
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            rrf_scores[doc_id] += self.bm25_weight * (1 / (k + rank + 1))
            if doc_id not in doc_data:
                doc_data[doc_id] = result
            doc_data[doc_id]["bm25_score"] = result.get("score", 0)
            doc_data[doc_id]["bm25_rank"] = rank + 1
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results = []
        for doc_id in sorted_ids:
            doc = doc_data[doc_id]
            doc["score"] = rrf_scores[doc_id]
            doc["retrieval_type"] = "hybrid"
            results.append(doc)
        
        return results
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        mode: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Search using hybrid retrieval
        
        Args:
            query: Search query
            top_k: Number of results to return
            mode: "hybrid", "vector", "bm25", or "naive"
            
        Returns:
            List of search results
        """
        if mode == "vector":
            return self.vector_store.search(query, top_k=top_k)
        
        if mode == "bm25":
            return self.bm25_retriever.search(query, top_k=top_k)
        
        if mode == "naive":
            # Simple concatenation without ranking
            vector_results = self.vector_store.search(query, top_k=top_k // 2)
            bm25_results = self.bm25_retriever.search(query, top_k=top_k // 2)
            return vector_results + bm25_results
        
        # Hybrid mode: use RRF fusion
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
        
        combined = self._reciprocal_rank_fusion(vector_results, bm25_results)
        return combined[:top_k]
    
    def search_multimodal(
        self, 
        query: str, 
        content_types: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search with content type filtering (for multimodal content)
        
        Args:
            query: Search query
            content_types: Filter by content types (e.g., ["table", "text", "equation"])
            top_k: Number of results
            
        Returns:
            Filtered search results
        """
        results = self.search(query, top_k=top_k * 2, mode="hybrid")
        
        if not content_types:
            return results[:top_k]
        
        # Filter by content type
        filtered = []
        for result in results:
            content_type = result.get("metadata", {}).get("content_type", "text")
            if content_type in content_types:
                filtered.append(result)
        
        return filtered[:top_k]
    
    def clear(self):
        """Clear BM25 index (vector store handles its own clearing)"""
        self.bm25_retriever.clear()


class QueryExpander:
    """
    Expand queries for better retrieval coverage
    """
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate query variations for better recall
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": """Generate 3 alternative phrasings of the user's question that might help find relevant information. 
Keep the same meaning but use different words/structure.
Return only the 3 alternatives, one per line, without numbering."""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            alternatives = response.choices[0].message.content.strip().split('\n')
            return [query] + [alt.strip() for alt in alternatives if alt.strip()]
            
        except Exception:
            return [query]
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract key search terms from query"""
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract 3-5 key search terms from the query. Return only the terms, comma-separated."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=50,
                temperature=0.2
            )
            
            keywords = response.choices[0].message.content.strip().split(',')
            return [kw.strip() for kw in keywords if kw.strip()]
            
        except Exception:
            # Simple fallback
            words = re.findall(r'\b[a-z]+\b', query.lower())
            stopwords = {'what', 'how', 'why', 'when', 'where', 'which', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for'}
            return [w for w in words if w not in stopwords and len(w) > 2][:5]


# Factory function to create hybrid retriever
def create_hybrid_retriever(vector_store, bm25_weight: float = 0.3, vector_weight: float = 0.7):
    """Create a hybrid retriever instance"""
    return HybridRetriever(vector_store, bm25_weight, vector_weight)
