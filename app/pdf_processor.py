"""
PDF Processing Module - Handles PDF upload, text extraction, and chunking
"""
import os
import uuid
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings


class PDFProcessor:
    """Handles PDF document processing"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract all text content from a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            reader = PdfReader(file_path)
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into smaller chunks with metadata
        
        Args:
            text: The text to split
            metadata: Additional metadata to attach to each chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if metadata is None:
            metadata = {}
        
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            documents.append(doc)
        
        return documents
    
    def process_pdf(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Complete PDF processing pipeline: extract text and create chunks
        
        Args:
            file_path: Path to the PDF file
            filename: Original filename for metadata
            
        Returns:
            List of document chunks with metadata
        """
        # Extract text
        text = self.extract_text_from_pdf(file_path)
        
        if not text.strip():
            raise ValueError("No text content could be extracted from the PDF")
        
        # Create metadata
        metadata = {
            "source": filename,
            "file_path": file_path,
        }
        
        # Chunk the text
        chunks = self.chunk_text(text, metadata)
        
        return chunks


# Create global processor instance
pdf_processor = PDFProcessor()
