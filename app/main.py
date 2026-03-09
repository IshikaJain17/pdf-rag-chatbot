"""
Main FastAPI Application - PDF RAG Chatbot
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

# Create FastAPI app
app = FastAPI(
    title="PDF RAG Chatbot",
    description="A RAG-based chatbot that answers questions from uploaded PDF documents",
    version="1.0.0"
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


# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = None
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved_documents: int
    model: str
    tokens_used: dict


class UploadResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int
    message: str


class StatsResponse(BaseModel):
    total_documents: int
    total_sources: int
    sources: List[str]
    collection_name: str


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
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and process it into the vector store
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
        
        # Process PDF
        chunks = pdf_processor.process_pdf(file_path, file.filename)
        
        # Add to vector store
        result = vector_store.add_documents(chunks)
        
        return UploadResponse(
            status="success",
            filename=file.filename,
            chunks_created=len(chunks),
            message=f"Successfully processed and indexed {file.filename}"
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
    Send a question and get an answer based on uploaded documents
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Convert chat history to proper format
    chat_history = None
    if request.chat_history:
        chat_history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
    
    # Get answer using RAG
    result = rag_chain.get_answer(
        question=request.question,
        top_k=request.top_k,
        chat_history=chat_history
    )
    
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["answer"])
    
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieved_documents=result["retrieved_documents"],
        model=result["model"],
        tokens_used=result["tokens_used"]
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Send a question and get a streaming answer based on uploaded documents
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
            chat_history=chat_history
        ):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get statistics about the current document collection
    """
    stats = vector_store.get_collection_stats()
    
    if "error" in stats:
        raise HTTPException(status_code=500, detail=stats["error"])
    
    return StatsResponse(
        total_documents=stats["total_documents"],
        total_sources=stats["total_sources"],
        sources=stats["sources"],
        collection_name=stats["collection_name"]
    )


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
    Clear all documents from the vector store
    """
    result = vector_store.clear_collection()
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "openai_configured": bool(settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here"),
        "pinecone_configured": bool(settings.pinecone_api_key and settings.pinecone_api_key != "your_pinecone_api_key_here"),
        "pinecone_index": settings.pinecone_index_name,
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
