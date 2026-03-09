"""
Knowledge Graph Module - Inspired by RAG-Anything
Entity extraction, relationship mapping, and graph-based retrieval
"""
import re
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from openai import OpenAI
from app.config import settings


class KnowledgeGraph:
    """
    Build and query knowledge graphs from documents
    Implements entity extraction and relationship mapping
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids
        self.document_entities: Dict[str, Set[str]] = defaultdict(set)  # doc_id -> entity_ids
    
    def extract_entities_llm(self, text: str, doc_id: str = "unknown") -> List[Dict[str, Any]]:
        """
        Extract entities from text using LLM
        
        Args:
            text: Document text to analyze
            doc_id: Document identifier
            
        Returns:
            List of extracted entities
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an entity extraction expert. Extract key entities from the text.
                        
For each entity, provide:
- name: The entity name
- type: One of [PERSON, ORGANIZATION, LOCATION, DATE, CONCEPT, PRODUCT, EVENT, METRIC, TECHNOLOGY, DOCUMENT]
- description: Brief description (1-2 sentences)
- importance: Score from 1-10

Return a JSON array of entities. Be thorough but only include meaningful entities."""
                    },
                    {
                        "role": "user",
                        "content": f"Extract entities from this text:\n\n{text[:4000]}"  # Limit text length
                    }
                ],
                max_tokens=1500,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            entities = result.get("entities", result) if isinstance(result, dict) else result
            
            if isinstance(entities, list):
                for entity in entities:
                    entity["source_doc"] = doc_id
                return entities
            return []
            
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return self._extract_entities_rules(text, doc_id)
    
    def _extract_entities_rules(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Rule-based entity extraction fallback
        """
        entities = []
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{4}\b'
        ]
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "name": match.group(),
                    "type": "DATE",
                    "description": "Date reference",
                    "importance": 5,
                    "source_doc": doc_id
                })
        
        # Extract percentages and metrics
        metric_pattern = r'\b\d+(?:\.\d+)?%|\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?\b'
        for match in re.finditer(metric_pattern, text, re.IGNORECASE):
            entities.append({
                "name": match.group(),
                "type": "METRIC",
                "description": "Numerical metric or percentage",
                "importance": 7,
                "source_doc": doc_id
            })
        
        # Extract capitalized phrases (potential names/organizations)
        cap_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        for match in re.finditer(cap_pattern, text):
            entities.append({
                "name": match.group(),
                "type": "CONCEPT",
                "description": "Named entity",
                "importance": 6,
                "source_doc": doc_id
            })
        
        # Deduplicate by name
        seen = set()
        unique_entities = []
        for e in entities:
            if e["name"] not in seen:
                seen.add(e["name"])
                unique_entities.append(e)
        
        return unique_entities[:30]  # Limit to top 30
    
    def extract_relationships_llm(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using LLM
        """
        if len(entities) < 2:
            return []
        
        entity_names = [e["name"] for e in entities[:20]]  # Limit for context
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a relationship extraction expert. Given entities and text, identify relationships between entities.

For each relationship, provide:
- source: Source entity name
- target: Target entity name
- relation_type: Type of relationship (e.g., "belongs_to", "created_by", "part_of", "related_to", "causes", "located_in", "works_for")
- description: Brief description of the relationship
- strength: Confidence score 1-10

Return a JSON object with a "relationships" array."""
                    },
                    {
                        "role": "user",
                        "content": f"Entities: {entity_names}\n\nText:\n{text[:3000]}"
                    }
                ],
                max_tokens=1000,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("relationships", [])
            
        except Exception as e:
            print(f"Relationship extraction error: {e}")
            return []
    
    def add_document(self, doc_id: str, text: str, extract_relations: bool = True) -> Dict[str, Any]:
        """
        Add a document to the knowledge graph
        
        Args:
            doc_id: Document identifier
            text: Document text
            extract_relations: Whether to extract relationships
            
        Returns:
            Extraction results
        """
        # Extract entities
        entities = self.extract_entities_llm(text, doc_id)
        
        # Store entities
        for entity in entities:
            entity_id = f"{entity['type']}:{entity['name']}"
            if entity_id not in self.entities:
                self.entities[entity_id] = entity
            self.entity_index[entity["type"]].add(entity_id)
            self.document_entities[doc_id].add(entity_id)
        
        # Extract relationships
        if extract_relations and len(entities) >= 2:
            relationships = self.extract_relationships_llm(text, entities)
            for rel in relationships:
                rel["doc_id"] = doc_id
                self.relationships.append(rel)
        
        return {
            "doc_id": doc_id,
            "entities_extracted": len(entities),
            "relationships_extracted": len(self.relationships) if extract_relations else 0,
            "entities": entities
        }
    
    def get_related_entities(self, entity_name: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Get entities related to a given entity through graph traversal
        """
        related = []
        visited = set()
        to_visit = [(entity_name, 0)]
        
        while to_visit:
            current, depth = to_visit.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            
            for rel in self.relationships:
                if rel["source"] == current and rel["target"] not in visited:
                    related.append({
                        "entity": rel["target"],
                        "relation": rel["relation_type"],
                        "depth": depth + 1
                    })
                    to_visit.append((rel["target"], depth + 1))
                elif rel["target"] == current and rel["source"] not in visited:
                    related.append({
                        "entity": rel["source"],
                        "relation": f"inverse_{rel['relation_type']}",
                        "depth": depth + 1
                    })
                    to_visit.append((rel["source"], depth + 1))
        
        return related
    
    def search_entities(self, query: str, entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for entities matching a query
        """
        results = []
        query_lower = query.lower()
        
        for entity_id, entity in self.entities.items():
            if entity_types and entity["type"] not in entity_types:
                continue
            
            # Match by name or description
            if (query_lower in entity["name"].lower() or 
                query_lower in entity.get("description", "").lower()):
                results.append({
                    **entity,
                    "entity_id": entity_id,
                    "relevance": 1.0 if query_lower == entity["name"].lower() else 0.7
                })
        
        return sorted(results, key=lambda x: (-x["relevance"], -x.get("importance", 0)))
    
    def get_entity_context(self, entity_name: str) -> str:
        """
        Get rich context about an entity including its relationships
        """
        context_parts = []
        
        # Find the entity
        matching_entities = [e for e in self.entities.values() if e["name"].lower() == entity_name.lower()]
        
        if matching_entities:
            entity = matching_entities[0]
            context_parts.append(f"Entity: {entity['name']} (Type: {entity['type']})")
            context_parts.append(f"Description: {entity.get('description', 'N/A')}")
            
            # Get relationships
            related = self.get_related_entities(entity_name, max_depth=1)
            if related:
                context_parts.append("\nRelated entities:")
                for rel in related[:10]:
                    context_parts.append(f"  - {rel['entity']} ({rel['relation']})")
        
        return "\n".join(context_parts)
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        type_counts = defaultdict(int)
        for entity in self.entities.values():
            type_counts[entity["type"]] += 1
        
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "total_documents": len(self.document_entities),
            "entities_by_type": dict(type_counts),
            "relationship_types": list(set(r["relation_type"] for r in self.relationships))
        }
    
    def export_graph(self) -> Dict[str, Any]:
        """Export the knowledge graph for visualization"""
        nodes = []
        for entity_id, entity in self.entities.items():
            nodes.append({
                "id": entity_id,
                "label": entity["name"],
                "type": entity["type"],
                "importance": entity.get("importance", 5)
            })
        
        edges = []
        for rel in self.relationships:
            source_id = None
            target_id = None
            for eid, e in self.entities.items():
                if e["name"] == rel["source"]:
                    source_id = eid
                if e["name"] == rel["target"]:
                    target_id = eid
            
            if source_id and target_id:
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "label": rel["relation_type"],
                    "strength": rel.get("strength", 5)
                })
        
        return {"nodes": nodes, "edges": edges}
    
    def clear(self):
        """Clear the knowledge graph"""
        self.entities.clear()
        self.relationships.clear()
        self.entity_index.clear()
        self.document_entities.clear()


# Singleton instance
knowledge_graph = KnowledgeGraph()
