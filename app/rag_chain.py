"""
RAG Chain Module - Handles question answering with retrieval augmented generation
"""
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import settings
from app.vector_store import vector_store


class RAGChain:
    """Manages RAG-based question answering"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.vector_store = vector_store
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context for the LLM
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found in the uploaded documents."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get("metadata", {}).get("source", "Unknown")
            text = doc.get("text", "")
            score = doc.get("score", 0)
            context_parts.append(f"[Source: {source} | Relevance: {score:.2f}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def create_system_prompt(self) -> str:
        """
        Create the system prompt for the chatbot
        
        Returns:
            System prompt string
        """
        return """You are a helpful AI assistant that answers questions based on the provided document context.

Your guidelines:
1. Answer questions ONLY based on the information provided in the context below
2. If the context doesn't contain relevant information to answer the question, clearly state that the information is not available in the documents
3. When answering, cite which source document the information came from when possible
4. Be accurate, concise, and helpful
5. If asked about topics not covered in the documents, politely explain that you can only answer questions about the uploaded documents
6. Provide detailed explanations when the context supports it
7. If the question is ambiguous, ask for clarification

Remember: You are an assistant for the uploaded PDF documents. Your knowledge is limited to what's in those documents."""
    
    def create_user_prompt(self, question: str, context: str) -> str:
        """
        Create the user prompt with context and question
        
        Args:
            question: User's question
            context: Retrieved context from documents
            
        Returns:
            Formatted user prompt
        """
        return f"""Based on the following context from the uploaded documents, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive answer based on the context above. If the information isn't available in the context, say so clearly."""
    
    def get_answer(
        self, 
        question: str, 
        top_k: int = 5,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Get an answer to a question using RAG
        
        Args:
            question: The user's question
            top_k: Number of relevant documents to retrieve
            chat_history: Optional list of previous messages for context
            
        Returns:
            Dictionary containing the answer and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(question, top_k=top_k)
        
        # Format context
        context = self.format_context(retrieved_docs)
        
        # Build messages
        messages = [
            {"role": "system", "content": self.create_system_prompt()}
        ]
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history[-6:]:  # Keep last 6 messages for context
                messages.append(msg)
        
        # Add current question with context
        messages.append({
            "role": "user",
            "content": self.create_user_prompt(question, context)
        })
        
        # Get response from OpenAI
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature
            )
            
            answer = response.choices[0].message.content
            
            # Extract sources from retrieved documents
            sources = []
            for doc in retrieved_docs:
                source = doc.get("metadata", {}).get("source", "Unknown")
                if source not in sources:
                    sources.append(source)
            
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_documents": len(retrieved_docs),
                "model": settings.chat_model,
                "tokens_used": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "error": True
            }
    
    def get_streaming_answer(
        self,
        question: str,
        top_k: int = 5,
        chat_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Get a streaming answer to a question using RAG
        
        Args:
            question: The user's question
            top_k: Number of relevant documents to retrieve
            chat_history: Optional list of previous messages for context
            
        Yields:
            Chunks of the response as they're generated
        """
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(question, top_k=top_k)
        
        # Format context
        context = self.format_context(retrieved_docs)
        
        # Build messages
        messages = [
            {"role": "system", "content": self.create_system_prompt()}
        ]
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append(msg)
        
        # Add current question with context
        messages.append({
            "role": "user",
            "content": self.create_user_prompt(question, context)
        })
        
        # Get streaming response from OpenAI
        try:
            stream = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error generating response: {str(e)}"


# Create global RAG chain instance
rag_chain = RAGChain()
