"""
Main FastAPI Application - Advanced PDF RAG Chatbot
Powered by RAG-Anything Inspired Architecture
"""
import os
import shutil
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.pdf_processor import pdf_processor
from app.vector_store import vector_store
from app.rag_chain import rag_chain
from app.knowledge_graph import knowledge_graph
from app.multimodal_processor import multimodal_processor
from app.hybrid_retrieval import create_hybrid_retriever

# Create FastAPI app
app = FastAPI(
    title="Advanced PDF RAG Chatbot",
    description="A smart RAG-based chatbot inspired by RAG-Anything with multimodal content understanding",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files
STATIC_DIR = "./static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize hybrid retriever
hybrid_retriever = create_hybrid_retriever(vector_store)


# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = None
    top_k: int = 5
    retrieval_mode: str = "hybrid"  # "hybrid", "vector", "bm25"
    response_mode: str = "standard"  # "standard", "analytical", "comprehensive", "concise"
    use_query_enhancement: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved_documents: int
    model: str
    tokens_used: dict
    follow_up_questions: Optional[List[str]] = None
    query_classification: Optional[dict] = None
    retrieval_mode: Optional[str] = None


class UploadResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int
    message: str
    document_metadata: Optional[dict] = None
    entities_extracted: Optional[int] = None


class StatsResponse(BaseModel):
    total_documents: int
    total_sources: int
    sources: List[str]
    collection_name: str
    knowledge_graph_stats: Optional[dict] = None


class EntityResponse(BaseModel):
    entities: List[dict]
    relationships: int
    documents: int


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Welcome to PDF RAG Chatbot</h1><p>Please ensure static/index.html exists.</p>")


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), extract_entities: bool = True):
    """
    Upload a PDF file and process it with advanced multimodal extraction
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())[:8]
    safe_filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process PDF with enhanced extraction
        chunks, doc_metadata = pdf_processor.process_pdf(file_path, file.filename)
        
        # Add to vector store
        result = vector_store.add_documents(chunks)
        
        # Add to hybrid retriever's BM25 index
        hybrid_retriever.add_documents(chunks)
        
        # Extract entities for knowledge graph
        entities_count = 0
        if extract_entities:
            # Get full text for entity extraction
            text, _ = pdf_processor.extract_text_from_pdf(file_path)
            kg_result = knowledge_graph.add_document(file_id, text[:10000])  # Limit text for efficiency
            entities_count = kg_result.get("entities_extracted", 0)
        
        return UploadResponse(
            status="success",
            filename=file.filename,
            chunks_created=len(chunks),
            message=f"Successfully processed and indexed {file.filename} with advanced multimodal extraction",
            document_metadata={
                "total_pages": doc_metadata.get("total_pages", 0),
                "total_words": doc_metadata.get("total_words", 0),
                "tables_found": len(doc_metadata.get("pages_with_tables", [])),
                "sections": len(doc_metadata.get("sections", []))
            },
            entities_extracted=entities_count
        )
        
    except ValueError as e:
        # Clean up file if processing failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Clean up file if processing failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a question and get an intelligent answer using advanced RAG
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Convert chat history to proper format
    chat_history = None
    if request.chat_history:
        chat_history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
    
    # Get answer using advanced RAG
    result = rag_chain.get_answer(
        question=request.question,
        top_k=request.top_k,
        chat_history=chat_history,
        retrieval_mode=request.retrieval_mode,
        response_mode=request.response_mode,
        use_query_enhancement=request.use_query_enhancement
    )
    
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["answer"])
    
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieved_documents=result["retrieved_documents"],
        model=result["model"],
        tokens_used=result["tokens_used"],
        follow_up_questions=result.get("follow_up_questions", []),
        query_classification=result.get("query_classification"),
        retrieval_mode=result.get("retrieval_mode")
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Send a question and get a streaming answer with advanced RAG
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Convert chat history to proper format
    chat_history = None
    if request.chat_history:
        chat_history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
    
    def generate():
        for chunk in rag_chain.get_streaming_answer(
            question=request.question,
            top_k=request.top_k,
            chat_history=chat_history,
            retrieval_mode=request.retrieval_mode
        ):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/api/chat/decompose")
async def chat_with_decomposition(request: ChatRequest):
    """
    Answer complex questions using decomposition strategy
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    chat_history = None
    if request.chat_history:
        chat_history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
    
    result = rag_chain.answer_with_decomposition(
        question=request.question,
        top_k=request.top_k,
        chat_history=chat_history
    )
    
    return result


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get comprehensive statistics about the document collection
    """
    stats = vector_store.get_collection_stats()
    kg_stats = knowledge_graph.get_graph_statistics()
    
    if "error" in stats:
        raise HTTPException(status_code=500, detail=stats["error"])
    
    return StatsResponse(
        total_documents=stats["total_documents"],
        total_sources=stats["total_sources"],
        sources=stats["sources"],
        collection_name=stats["collection_name"],
        knowledge_graph_stats=kg_stats
    )


@app.get("/api/entities")
async def get_entities(entity_type: Optional[str] = None, limit: int = 50):
    """
    Get entities from the knowledge graph
    """
    entities = list(knowledge_graph.entities.values())
    
    if entity_type:
        entities = [e for e in entities if e.get("type") == entity_type]
    
    return {
        "entities": entities[:limit],
        "total": len(entities),
        "types": list(knowledge_graph.entity_index.keys())
    }


@app.get("/api/entities/{entity_name}/context")
async def get_entity_context(entity_name: str):
    """
    Get rich context about a specific entity
    """
    context = knowledge_graph.get_entity_context(entity_name)
    related = knowledge_graph.get_related_entities(entity_name, max_depth=2)
    
    return {
        "entity": entity_name,
        "context": context,
        "related_entities": related
    }


@app.get("/api/knowledge-graph")
async def get_knowledge_graph():
    """
    Export knowledge graph for visualization
    """
    return knowledge_graph.export_graph()


@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Analyze a PDF document without adding to the knowledge base
    Returns detailed content analysis
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_id = str(uuid.uuid4())[:8]
    safe_filename = f"temp_{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get full analysis
        text, doc_metadata = pdf_processor.extract_text_from_pdf(file_path)
        content_analysis = multimodal_processor.analyze_content_structure(text)
        
        # Clean up temp file
        os.remove(file_path)
        
        return {
            "filename": file.filename,
            "document_metadata": doc_metadata,
            "content_analysis": {
                "tables_found": len(content_analysis.get("tables", [])),
                "equations_found": len(content_analysis.get("equations", [])),
                "lists_found": len(content_analysis.get("lists", [])),
                "sections_found": len(content_analysis.get("sections", [])),
                "statistics": content_analysis.get("statistics", {})
            },
            "sections": content_analysis.get("sections", [])[:20],
            "tables_preview": [
                {"content": t["content"][:500], "format": t["format"]} 
                for t in content_analysis.get("tables", [])[:5]
            ]
        }
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error analyzing PDF: {str(e)}")


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete all chunks from a specific document
    """
    result = vector_store.delete_by_source(filename)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result


@app.delete("/api/documents")
async def clear_all_documents():
    """
    Clear all documents from the vector store and knowledge graph
    """
    result = vector_store.clear_collection()
    hybrid_retriever.clear()
    knowledge_graph.clear()
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    result["knowledge_graph_cleared"] = True
    return result


@app.get("/api/health")
async def health_check():
    """
    Comprehensive health check endpoint
    """
    kg_stats = knowledge_graph.get_graph_statistics()
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": {
            "hybrid_retrieval": True,
            "knowledge_graph": True,
            "query_enhancement": True,
            "multimodal_processing": True,
            "streaming_responses": True
        },
        "openai_configured": bool(settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here"),
        "pinecone_configured": bool(settings.pinecone_api_key and settings.pinecone_api_key != "your_pinecone_api_key_here"),
        "pinecone_index": settings.pinecone_index_name,
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "knowledge_graph": {
            "entities": kg_stats.get("total_entities", 0),
            "relationships": kg_stats.get("total_relationships", 0)
        }
    }


@app.get("/api/retrieval-modes")
async def get_retrieval_modes():
    """
    Get available retrieval modes and their descriptions
    """
    return {
        "modes": {
            "hybrid": {
                "name": "Hybrid (Recommended)",
                "description": "Combines vector similarity with keyword matching using Reciprocal Rank Fusion"
            },
            "vector": {
                "name": "Vector Only",
                "description": "Pure semantic similarity search using embeddings"
            },
            "bm25": {
                "name": "BM25 (Keyword)",
                "description": "Traditional keyword-based search with TF-IDF weighting"
            }
        },
        "response_modes": {
            "standard": "Balanced responses",
            "analytical": "Deep analysis with evidence",
            "comprehensive": "Thorough, detailed coverage",
            "concise": "Brief, direct answers"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
