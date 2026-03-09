"""
Advanced RAG Chain Module - Inspired by RAG-Anything
Enhanced retrieval, multi-stage reasoning, and intelligent response generation
"""
from typing import List, Dict, Any, Optional, Generator
from openai import OpenAI
from app.config import settings
from app.vector_store import vector_store
from app.hybrid_retrieval import create_hybrid_retriever, QueryExpander
from app.query_enhancer import query_enhancer, contextual_handler
from app.knowledge_graph import knowledge_graph


class AdvancedRAGChain:
    """
    Advanced RAG-based question answering with:
    - Hybrid retrieval (vector + BM25)
    - Query enhancement and decomposition
    - Knowledge graph integration
    - Multi-stage reasoning
    - Context-aware responses
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.vector_store = vector_store
        self.hybrid_retriever = create_hybrid_retriever(vector_store)
        self.query_expander = QueryExpander(self.openai_client)
        self.knowledge_graph = knowledge_graph
    
    def format_context(self, documents: List[Dict[str, Any]], include_metadata: bool = True) -> str:
        """
        Format retrieved documents into rich context for the LLM
        
        Args:
            documents: List of retrieved documents
            include_metadata: Whether to include detailed metadata
            
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
            content_type = doc.get("metadata", {}).get("content_type", "text")
            page = doc.get("metadata", {}).get("page_number", "")
            retrieval_type = doc.get("retrieval_type", "vector")
            
            if include_metadata:
                header = f"[#{i} | Source: {source}"
                if page:
                    header += f" | Page: {page}"
                header += f" | Type: {content_type} | Score: {score:.2f} | Via: {retrieval_type}]"
            else:
                header = f"[Source: {source}]"
            
            context_parts.append(f"{header}\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _get_entity_context(self, query: str) -> str:
        """Get relevant entity information from knowledge graph"""
        entities = query_enhancer.extract_entities_from_query(query)
        
        entity_contexts = []
        for entity in entities[:3]:  # Limit to top 3 entities
            context = self.knowledge_graph.get_entity_context(entity.get("name", ""))
            if context:
                entity_contexts.append(context)
        
        if entity_contexts:
            return "\n\n[Knowledge Graph Context]\n" + "\n---\n".join(entity_contexts)
        return ""
    
    def create_system_prompt(self, mode: str = "standard") -> str:
        """
        Create adaptive system prompt based on query type
        
        Args:
            mode: Response mode - "standard", "analytical", "comprehensive", "concise"
            
        Returns:
            System prompt string
        """
        base_prompt = """You are an advanced AI document assistant powered by RAG-Anything inspired technology.

Your capabilities:
1. Answer questions based ONLY on the provided document context
2. Analyze tables, equations, and structured content
3. Synthesize information from multiple sources
4. Identify relationships between concepts
5. Provide accurate citations to source documents

Guidelines:"""
        
        if mode == "analytical":
            return base_prompt + """
- Provide deep analysis with evidence from documents
- Compare and contrast different viewpoints if present
- Identify patterns and trends in the data
- Draw logical conclusions supported by context
- Use structured formatting (bullets, numbered lists) for clarity"""
        
        elif mode == "comprehensive":
            return base_prompt + """
- Give thorough, detailed responses
- Cover all relevant aspects found in documents
- Include specific quotes and references
- Explain complex concepts clearly
- Provide background context when helpful"""
        
        elif mode == "concise":
            return base_prompt + """
- Be brief and direct
- Focus on the most relevant information
- Use bullet points for multiple items
- Avoid unnecessary elaboration
- Still cite sources when possible"""
        
        else:  # standard
            return base_prompt + """
- Be accurate, helpful, and well-organized
- Cite sources when providing specific information
- If information isn't available, clearly state so
- Ask for clarification if the question is ambiguous
- Balance detail with readability"""
    
    def create_user_prompt(self, question: str, context: str, entity_context: str = "") -> str:
        """Create enhanced user prompt with context"""
        prompt = f"""Based on the following context from the uploaded documents, please answer the question.

DOCUMENT CONTEXT:
{context}"""
        
        if entity_context:
            prompt += f"\n\n{entity_context}"
        
        prompt += f"""

QUESTION: {question}

Provide a well-structured answer based on the context. Cite specific sources when possible."""
        
        return prompt
    
    def get_answer(
        self, 
        question: str, 
        top_k: int = 5,
        chat_history: Optional[List[Dict[str, str]]] = None,
        retrieval_mode: str = "hybrid",
        response_mode: str = "standard",
        use_query_enhancement: bool = True
    ) -> Dict[str, Any]:
        """
        Get an enhanced answer using advanced RAG
        
        Args:
            question: The user's question
            top_k: Number of documents to retrieve
            chat_history: Previous conversation messages
            retrieval_mode: "hybrid", "vector", "bm25"
            response_mode: "standard", "analytical", "comprehensive", "concise"
            use_query_enhancement: Whether to enhance the query
            
        Returns:
            Dictionary with answer and rich metadata
        """
        # Resolve coreferences if chat history exists
        resolved_query = question
        if chat_history:
            resolved_query = contextual_handler.resolve_coreferences(question, chat_history)
        
        # Enhance query if enabled
        enhanced_info = {}
        search_queries = [resolved_query]
        
        if use_query_enhancement:
            enhanced_info = query_enhancer.enhance_query(resolved_query)
            search_queries = enhanced_info.get("expansions", [resolved_query])[:3]
            
            # Adjust response mode based on query complexity
            if enhanced_info.get("classification", {}).get("complexity") == "complex":
                response_mode = "analytical"
        
        # Retrieve documents using multiple queries
        all_docs = []
        seen_ids = set()
        
        for query in search_queries:
            docs = self.hybrid_retriever.search(
                query, 
                top_k=top_k,
                mode=retrieval_mode
            )
            for doc in docs:
                if doc["id"] not in seen_ids:
                    seen_ids.add(doc["id"])
                    all_docs.append(doc)
        
        # Re-rank by score and take top_k
        all_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
        retrieved_docs = all_docs[:top_k]
        
        # Format context
        context = self.format_context(retrieved_docs)
        
        # Get entity context from knowledge graph
        entity_context = self._get_entity_context(question)
        
        # Build messages
        messages = [
            {"role": "system", "content": self.create_system_prompt(response_mode)}
        ]
        
        # Add chat history
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append(msg)
        
        # Add current question with context
        messages.append({
            "role": "user",
            "content": self.create_user_prompt(resolved_query, context, entity_context)
        })
        
        # Generate response
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature
            )
            
            answer = response.choices[0].message.content
            
            # Extract unique sources
            sources = list(set(
                doc.get("metadata", {}).get("source", "Unknown")
                for doc in retrieved_docs
            ))
            
            # Generate follow-up questions
            follow_ups = []
            if use_query_enhancement:
                follow_ups = query_enhancer.generate_follow_up_questions(
                    question, answer, context[:1000]
                )
            
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_documents": len(retrieved_docs),
                "model": settings.chat_model,
                "tokens_used": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "retrieval_mode": retrieval_mode,
                "response_mode": response_mode,
                "query_enhanced": use_query_enhancement,
                "query_classification": enhanced_info.get("classification", {}),
                "follow_up_questions": follow_ups,
                "entities_found": enhanced_info.get("entities", [])
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "error": True,
                "retrieved_documents": 0,
                "model": settings.chat_model
            }
    
    def get_streaming_answer(
        self,
        question: str,
        top_k: int = 5,
        chat_history: Optional[List[Dict[str, str]]] = None,
        retrieval_mode: str = "hybrid"
    ) -> Generator[str, None, None]:
        """
        Get a streaming answer using advanced RAG
        
        Yields:
            Chunks of the response as they're generated
        """
        # Resolve query
        resolved_query = question
        if chat_history:
            resolved_query = contextual_handler.resolve_coreferences(question, chat_history)
        
        # Retrieve documents
        retrieved_docs = self.hybrid_retriever.search(
            resolved_query,
            top_k=top_k,
            mode=retrieval_mode
        )
        
        # Format context
        context = self.format_context(retrieved_docs)
        entity_context = self._get_entity_context(question)
        
        # Build messages
        messages = [
            {"role": "system", "content": self.create_system_prompt()}
        ]
        
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append(msg)
        
        messages.append({
            "role": "user",
            "content": self.create_user_prompt(resolved_query, context, entity_context)
        })
        
        # Stream response
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
    
    def answer_with_decomposition(
        self,
        question: str,
        top_k: int = 5,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Answer complex questions by decomposing them
        
        Uses query decomposition for multi-step reasoning
        """
        # Decompose the question
        sub_questions = query_enhancer.decompose_complex_query(question)
        
        if len(sub_questions) <= 1:
            # Simple question, use normal flow
            return self.get_answer(question, top_k, chat_history)
        
        # Answer each sub-question
        sub_answers = []
        for sq in sub_questions:
            sub_result = self.get_answer(
                sq["question"],
                top_k=top_k // 2 + 1,
                use_query_enhancement=False,
                response_mode="concise"
            )
            sub_answers.append({
                "question": sq["question"],
                "answer": sub_result["answer"],
                "sources": sub_result["sources"]
            })
        
        # Synthesize final answer
        synthesis_prompt = f"""Based on the following sub-questions and their answers, provide a comprehensive answer to the original question.

Original Question: {question}

Sub-questions and Answers:
"""
        for i, sa in enumerate(sub_answers, 1):
            synthesis_prompt += f"\n{i}. Q: {sa['question']}\n   A: {sa['answer']}\n"
        
        synthesis_prompt += "\nProvide a unified, coherent answer that synthesizes all the above information."
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": self.create_system_prompt("comprehensive")},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=settings.max_tokens,
                temperature=settings.temperature
            )
            
            # Combine all sources
            all_sources = set()
            for sa in sub_answers:
                all_sources.update(sa["sources"])
            
            return {
                "answer": response.choices[0].message.content,
                "sources": list(all_sources),
                "sub_questions": sub_answers,
                "reasoning_type": "decomposition",
                "model": settings.chat_model,
                "tokens_used": {
                    "total": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            return {
                "answer": f"Error synthesizing answer: {str(e)}",
                "sources": [],
                "error": True
            }


# Legacy compatibility - keep old class name working
class RAGChain(AdvancedRAGChain):
    """Alias for backward compatibility"""
    pass


# Create global RAG chain instance
rag_chain = AdvancedRAGChain()
