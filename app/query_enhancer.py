"""
Query Enhancement Module - Inspired by RAG-Anything
Smart query processing, rewriting, and decomposition
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from app.config import settings


class QueryEnhancer:
    """
    Enhance user queries for better RAG retrieval and answering
    Implements query understanding, expansion, and decomposition
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify the type and intent of a query
        
        Returns:
            Query classification with type, intent, and complexity
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze the user query and classify it. Return a JSON object with:
- query_type: One of ["factual", "analytical", "comparative", "procedural", "definitional", "exploratory"]
- intent: Brief description of what the user wants
- complexity: One of ["simple", "moderate", "complex"]
- requires_context: Boolean - does this need document context?
- key_concepts: List of main concepts/topics in the query
- temporal_aspect: Does query involve time/dates? ["none", "past", "present", "future", "comparative"]"""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=300,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "query_type": "factual",
                "intent": query,
                "complexity": "simple",
                "requires_context": True,
                "key_concepts": self._extract_keywords_simple(query),
                "temporal_aspect": "none"
            }
    
    def _extract_keywords_simple(self, query: str) -> List[str]:
        """Simple keyword extraction fallback"""
        stopwords = {'what', 'how', 'why', 'when', 'where', 'which', 'who',
                     'is', 'are', 'was', 'were', 'the', 'a', 'an', 
                     'in', 'on', 'at', 'to', 'for', 'of', 'with',
                     'can', 'could', 'would', 'should', 'do', 'does', 'did',
                     'this', 'that', 'these', 'those', 'it', 'its'}
        words = re.findall(r'\b[a-z]+\b', query.lower())
        return [w for w in words if w not in stopwords and len(w) > 2][:5]
    
    def expand_query(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate query expansions for better recall
        
        Args:
            query: Original user query
            num_variations: Number of alternative queries to generate
            
        Returns:
            List of query variations including original
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": f"""Generate {num_variations} alternative phrasings of the user's question.
Each alternative should:
1. Keep the same meaning/intent
2. Use different words or sentence structure
3. Potentially highlight different aspects of the question

Return only the alternatives, one per line, without numbering or bullets."""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            alternatives = [
                alt.strip() 
                for alt in response.choices[0].message.content.strip().split('\n') 
                if alt.strip()
            ]
            return [query] + alternatives[:num_variations]
            
        except Exception:
            return [query]
    
    def decompose_complex_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Decompose complex queries into simpler sub-questions
        
        Args:
            query: Complex user query
            
        Returns:
            List of sub-questions with dependencies
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze the query and break it down into simpler sub-questions if needed.

Return a JSON object with:
{
    "is_complex": boolean,
    "sub_questions": [
        {
            "question": "sub-question text",
            "order": 1,
            "depends_on": [] or [1, 2] (indices of prerequisite questions),
            "type": "lookup" | "reasoning" | "comparison"
        }
    ],
    "synthesis_needed": boolean (whether answers need to be combined)
}

If the query is simple, return is_complex: false with the original query as the only sub_question."""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=500,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("sub_questions", [{"question": query, "order": 1, "depends_on": [], "type": "lookup"}])
            
        except Exception:
            return [{"question": query, "order": 1, "depends_on": [], "type": "lookup"}]
    
    def rewrite_for_retrieval(self, query: str) -> str:
        """
        Rewrite query to optimize for vector search retrieval
        
        Args:
            query: Original query
            
        Returns:
            Optimized query for retrieval
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": """Rewrite the user's question as a statement or description that would match relevant document content.
                        
For example:
- "What causes climate change?" → "Climate change is caused by greenhouse gas emissions..."
- "How do I bake a cake?" → "Steps to bake a cake include mixing ingredients..."

Keep key terms and concepts. Return only the rewritten text."""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception:
            return query
    
    def extract_entities_from_query(self, query: str) -> List[Dict[str, str]]:
        """Extract named entities from the query for knowledge graph lookup"""
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": """Extract named entities from the query. Return a JSON object with:
{
    "entities": [
        {"name": "entity name", "type": "PERSON|ORG|LOCATION|CONCEPT|PRODUCT|DATE"}
    ]
}
Only include clear, specific entities. Return empty array if none found."""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=200,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("entities", [])
            
        except Exception:
            return []
    
    def generate_follow_up_questions(self, query: str, answer: str, context: str) -> List[str]:
        """
        Generate relevant follow-up questions based on the Q&A
        
        Args:
            query: Original query
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            List of suggested follow-up questions
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": """Based on the question, answer, and context, suggest 3 relevant follow-up questions.
These should:
1. Dive deeper into the topic
2. Ask about related aspects not covered
3. Clarify or expand on the answer

Return only the questions, one per line, without numbering."""
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\nAnswer: {answer[:500]}\n\nContext excerpt: {context[:500]}"
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            questions = [
                q.strip() for q in response.choices[0].message.content.strip().split('\n')
                if q.strip() and '?' in q
            ]
            return questions[:3]
            
        except Exception:
            return []
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """
        Full query enhancement pipeline
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query information including classification,
            expansions, and retrieval-optimized versions
        """
        # Classify the query
        classification = self.classify_query(query)
        
        # Expand query for better recall
        expansions = self.expand_query(query, num_variations=2)
        
        # Rewrite for retrieval
        retrieval_query = self.rewrite_for_retrieval(query)
        
        # Extract entities for knowledge graph
        entities = self.extract_entities_from_query(query)
        
        # Decompose if complex
        is_complex = classification.get("complexity") == "complex"
        sub_questions = self.decompose_complex_query(query) if is_complex else []
        
        return {
            "original_query": query,
            "classification": classification,
            "expansions": expansions,
            "retrieval_query": retrieval_query,
            "entities": entities,
            "sub_questions": sub_questions,
            "search_keywords": classification.get("key_concepts", [])
        }


class ContextualQueryHandler:
    """
    Handle queries with conversational context
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
    
    def resolve_coreferences(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Resolve pronouns and references using chat history
        
        Example: "What did he say about it?" → "What did John say about the proposal?"
        """
        if not chat_history:
            return query
        
        # Check if query has pronouns that need resolution
        pronouns = ['it', 'this', 'that', 'these', 'those', 'he', 'she', 'they', 'them', 'his', 'her', 'their']
        query_lower = query.lower()
        
        if not any(f' {p} ' in f' {query_lower} ' or query_lower.startswith(f'{p} ') for p in pronouns):
            return query
        
        try:
            # Format recent history
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content'][:200]}"
                for msg in chat_history[-4:]
            ])
            
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": """Resolve any pronouns or vague references in the query using the conversation history.
Replace "it", "this", "that", "he", "she", etc. with their specific referents.
If already clear, return the query unchanged.
Return only the resolved query."""
                    },
                    {
                        "role": "user",
                        "content": f"Conversation:\n{history_text}\n\nQuery to resolve: {query}"
                    }
                ],
                max_tokens=150,
                temperature=0.2
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception:
            return query
    
    def combine_with_context(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """Combine current query with relevant context from history"""
        if not chat_history:
            return query
        
        # Get last few exchanges
        recent = chat_history[-4:] if len(chat_history) > 4 else chat_history
        
        # Build context
        context_parts = []
        for msg in recent:
            if msg["role"] == "assistant":
                # Extract key info from previous answers
                context_parts.append(f"Previous context: {msg['content'][:200]}...")
        
        if context_parts:
            return f"{query}\n\n[Conversation context: {' '.join(context_parts[:2])}]"
        
        return query


# Singleton instances
query_enhancer = QueryEnhancer()
contextual_handler = ContextualQueryHandler()
