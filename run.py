"""
Run script for PDF RAG Chatbot
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("=" * 50)
    print("PDF RAG Chatbot Assistant")
    print("=" * 50)
    print(f"Starting server at http://{settings.host}:{settings.port}")
    print(f"Using OpenAI model: {settings.chat_model}")
    print(f"Embedding model: {settings.embedding_model}")
    print("=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
