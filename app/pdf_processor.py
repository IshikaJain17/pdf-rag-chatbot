"""
PDF Processing Module - Enhanced with Multimodal Support
Inspired by RAG-Anything's document parsing approach
"""
import os
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings


class PDFProcessor:
    """
    Enhanced PDF document processing with multimodal content extraction
    Inspired by RAG-Anything's multi-stage parsing pipeline
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Semantic chunking for better context preservation
        self.semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size * 2,
            chunk_overlap=settings.chunk_overlap * 2,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", " "]
        )
    
    def extract_page_metadata(self, page, page_num: int) -> Dict[str, Any]:
        """Extract metadata from a PDF page"""
        metadata = {
            "page_number": page_num + 1,
            "has_images": False,
            "has_tables": False,
            "word_count": 0
        }
        
        text = page.extract_text() or ""
        metadata["word_count"] = len(text.split())
        
        # Check for potential tables (simple heuristic)
        if '|' in text or '\t\t' in text or re.search(r'\s{3,}[\d\w]', text):
            metadata["has_tables"] = True
        
        # Check for images (if page has xobjects)
        try:
            if '/XObject' in page.get('/Resources', {}):
                metadata["has_images"] = True
        except:
            pass
        
        return metadata
    
    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract all text content from a PDF file with enhanced metadata
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted text, document metadata)
        """
        try:
            reader = PdfReader(file_path)
            text = ""
            doc_metadata = {
                "total_pages": len(reader.pages),
                "pages_with_tables": [],
                "pages_with_images": [],
                "sections": [],
                "total_words": 0
            }
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                page_meta = self.extract_page_metadata(page, page_num)
                
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
                    doc_metadata["total_words"] += page_meta["word_count"]
                
                if page_meta["has_tables"]:
                    doc_metadata["pages_with_tables"].append(page_num + 1)
                if page_meta["has_images"]:
                    doc_metadata["pages_with_images"].append(page_num + 1)
            
            # Extract document outline/sections
            doc_metadata["sections"] = self._extract_sections(text)
            
            return text, doc_metadata
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract document sections and headings"""
        sections = []
        
        # Common heading patterns
        patterns = [
            (r'^(#{1,6})\s+(.+)$', 'markdown'),
            (r'^(\d+\.?\d*\.?)\s+([A-Z][^\n]+)$', 'numbered'),
            (r'^(Chapter|Section|Part)\s+(\d+|[IVX]+)[:\.]?\s*(.*)$', 'formal'),
            (r'^([A-Z][A-Z\s]{2,})$', 'caps'),
        ]
        
        for pattern, format_type in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                if format_type == 'markdown':
                    level = len(match.group(1))
                    title = match.group(2)
                elif format_type == 'numbered':
                    level = match.group(1).count('.') + 1
                    title = match.group(2)
                elif format_type == 'formal':
                    level = 1
                    title = f"{match.group(1)} {match.group(2)}"
                else:
                    level = 1
                    title = match.group(1)
                
                sections.append({
                    "level": level,
                    "title": title.strip(),
                    "position": match.start()
                })
        
        return sorted(sections, key=lambda x: x["position"])
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None, use_semantic: bool = False) -> List[Dict[str, Any]]:
        """
        Split text into smaller chunks with enhanced metadata
        
        Args:
            text: The text to split
            metadata: Additional metadata to attach to each chunk
            use_semantic: Use semantic chunking for better context
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if metadata is None:
            metadata = {}
        
        splitter = self.semantic_splitter if use_semantic else self.text_splitter
        chunks = splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            # Detect content type
            content_type = self._detect_content_type(chunk)
            
            # Extract page number from chunk if present
            page_match = re.search(r'--- Page (\d+) ---', chunk)
            page_num = int(page_match.group(1)) if page_match else None
            
            doc = {
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_type": content_type,
                    "page_number": page_num,
                    "char_count": len(chunk),
                    "word_count": len(chunk.split())
                }
            }
            documents.append(doc)
        
        return documents
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content in a chunk"""
        # Table detection
        if '|' in text and text.count('|') > 4:
            return "table"
        if re.search(r'\t.*\t.*\t', text):
            return "table"
        
        # Equation detection
        if re.search(r'\$[^$]+\$|\$\$[^$]+\$\$', text):
            return "equation"
        if re.search(r'[=∑∫∏√±×÷].*[a-zA-Z]', text):
            return "equation"
        
        # List detection
        if re.search(r'^[\s]*[•\-\*\d+\.]\s', text, re.MULTILINE):
            return "list"
        
        # Code detection
        if re.search(r'```|def |class |function |import |from .* import', text):
            return "code"
        
        return "text"
    
    def extract_tables_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract table-like structures from text"""
        tables = []
        
        # Markdown table pattern
        markdown_pattern = r'(\|[^\n]+\|(?:\n\|[^\n]+\|)+)'
        for match in re.finditer(markdown_pattern, text):
            table_text = match.group(1)
            tables.append({
                "type": "markdown",
                "content": table_text,
                "position": match.start(),
                "parsed": self._parse_table(table_text)
            })
        
        return tables
    
    def _parse_table(self, table_text: str) -> Dict[str, Any]:
        """Parse table text into structured data"""
        lines = [l.strip() for l in table_text.split('\n') if l.strip()]
        if len(lines) < 2:
            return {"headers": [], "rows": []}
        
        # Parse headers
        headers = [c.strip() for c in lines[0].split('|') if c.strip()]
        
        # Parse rows (skip separator)
        rows = []
        for line in lines[2:]:
            if '---' not in line:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if cells:
                    rows.append(cells)
        
        return {"headers": headers, "rows": rows}
    
    def process_pdf(self, file_path: str, filename: str, use_semantic_chunking: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Complete PDF processing pipeline with enhanced multimodal extraction
        
        Args:
            file_path: Path to the PDF file
            filename: Original filename for metadata
            use_semantic_chunking: Use semantic chunking for better context
            
        Returns:
            Tuple of (document chunks, document metadata)
        """
        # Extract text with metadata
        text, doc_metadata = self.extract_text_from_pdf(file_path)
        
        if not text.strip():
            raise ValueError("No text content could be extracted from the PDF")
        
        # Extract tables
        tables = self.extract_tables_from_text(text)
        doc_metadata["extracted_tables"] = len(tables)
        
        # Create base metadata
        metadata = {
            "source": filename,
            "file_path": file_path,
            "total_pages": doc_metadata["total_pages"],
            "total_words": doc_metadata["total_words"],
        }
        
        # Chunk the text
        chunks = self.chunk_text(text, metadata, use_semantic=use_semantic_chunking)
        
        # Add table chunks as separate documents with enhanced metadata
        for table in tables:
            table_chunk = {
                "id": str(uuid.uuid4()),
                "text": f"[TABLE]\n{table['content']}\n[/TABLE]",
                "metadata": {
                    **metadata,
                    "content_type": "table",
                    "chunk_index": len(chunks),
                    "total_chunks": len(chunks) + len(tables)
                }
            }
            chunks.append(table_chunk)
        
        return chunks, doc_metadata
    
    def process_pdf_advanced(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Advanced PDF processing with full content analysis
        
        Returns comprehensive document analysis including:
        - Text chunks
        - Document structure
        - Extracted entities (placeholder for knowledge graph)
        - Content statistics
        """
        chunks, doc_metadata = self.process_pdf(file_path, filename, use_semantic_chunking=True)
        
        # Get full text for analysis
        text, _ = self.extract_text_from_pdf(file_path)
        
        return {
            "chunks": chunks,
            "metadata": doc_metadata,
            "full_text": text,
            "content_analysis": {
                "total_chunks": len(chunks),
                "content_types": self._count_content_types(chunks),
                "sections": doc_metadata.get("sections", [])
            }
        }
    
    def _count_content_types(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count chunks by content type"""
        counts = {}
        for chunk in chunks:
            ct = chunk.get("metadata", {}).get("content_type", "text")
            counts[ct] = counts.get(ct, 0) + 1
        return counts


# Create global processor instance
pdf_processor = PDFProcessor()
