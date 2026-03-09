"""
Multimodal Content Processor - Inspired by RAG-Anything
Handles table extraction, image analysis, and structured content processing
"""
import re
import base64
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from app.config import settings


class MultimodalProcessor:
    """Process multimodal content from PDF documents"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
    
    def extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract table-like structures from text
        
        Args:
            text: Document text
            
        Returns:
            List of detected tables with metadata
        """
        tables = []
        
        # Pattern for markdown-style tables
        markdown_table_pattern = r'(\|[^\n]+\|(?:\n\|[^\n]+\|)+)'
        
        # Pattern for space/tab separated tables
        aligned_table_pattern = r'(?:^[^\n]*(?:\t|  +)[^\n]*$\n?){3,}'
        
        # Find markdown tables
        for match in re.finditer(markdown_table_pattern, text, re.MULTILINE):
            table_text = match.group(1)
            tables.append({
                "type": "table",
                "format": "markdown",
                "content": table_text,
                "start_pos": match.start(),
                "end_pos": match.end(),
                "parsed_data": self._parse_markdown_table(table_text)
            })
        
        return tables
    
    def _parse_markdown_table(self, table_text: str) -> Dict[str, Any]:
        """Parse markdown table into structured data"""
        lines = [line.strip() for line in table_text.strip().split('\n') if line.strip()]
        
        if len(lines) < 2:
            return {"headers": [], "rows": []}
        
        # Parse headers
        header_line = lines[0]
        headers = [cell.strip() for cell in header_line.split('|') if cell.strip()]
        
        # Skip separator line and parse rows
        rows = []
        for line in lines[2:]:  # Skip header and separator
            if '---' not in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    rows.append(cells)
        
        return {"headers": headers, "rows": rows}
    
    def extract_equations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mathematical equations and formulas
        
        Args:
            text: Document text
            
        Returns:
            List of detected equations
        """
        equations = []
        
        # LaTeX inline math
        inline_math = re.finditer(r'\$([^$]+)\$', text)
        for match in inline_math:
            equations.append({
                "type": "equation",
                "format": "latex_inline",
                "content": match.group(1),
                "start_pos": match.start(),
                "end_pos": match.end()
            })
        
        # LaTeX display math
        display_math = re.finditer(r'\$\$([^$]+)\$\$', text)
        for match in display_math:
            equations.append({
                "type": "equation",
                "format": "latex_display",
                "content": match.group(1),
                "start_pos": match.start(),
                "end_pos": match.end()
            })
        
        # Common equation patterns (x = ..., f(x) = ...)
        equation_patterns = [
            r'([a-zA-Z]\s*=\s*[^,\n]+)',  # Simple assignment
            r'(∑[^\n]+)',  # Summation
            r'(∫[^\n]+)',  # Integral
            r'([a-zA-Z]\([^)]+\)\s*=\s*[^\n]+)',  # Function definition
        ]
        
        for pattern in equation_patterns:
            for match in re.finditer(pattern, text):
                equations.append({
                    "type": "equation",
                    "format": "text",
                    "content": match.group(1),
                    "start_pos": match.start(),
                    "end_pos": match.end()
                })
        
        return equations
    
    def extract_lists(self, text: str) -> List[Dict[str, Any]]:
        """Extract bullet points and numbered lists"""
        lists = []
        
        # Bullet points
        bullet_pattern = r'((?:^[\s]*[•\-\*]\s*[^\n]+\n?)+)'
        for match in re.finditer(bullet_pattern, text, re.MULTILINE):
            items = [item.strip()[1:].strip() for item in match.group(1).strip().split('\n') if item.strip()]
            lists.append({
                "type": "list",
                "format": "bullet",
                "items": items,
                "start_pos": match.start(),
                "end_pos": match.end()
            })
        
        # Numbered lists
        numbered_pattern = r'((?:^[\s]*\d+[\.\)]\s*[^\n]+\n?)+)'
        for match in re.finditer(numbered_pattern, text, re.MULTILINE):
            items = re.findall(r'\d+[\.\)]\s*([^\n]+)', match.group(1))
            lists.append({
                "type": "list",
                "format": "numbered",
                "items": items,
                "start_pos": match.start(),
                "end_pos": match.end()
            })
        
        return lists
    
    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract document sections and headings"""
        sections = []
        
        # Common heading patterns
        patterns = [
            (r'^(#{1,6})\s+(.+)$', 'markdown'),  # Markdown headings
            (r'^([A-Z][A-Z\s]+)$', 'caps'),  # ALL CAPS headings
            (r'^(\d+\.?\d*\.?)\s+([A-Z][^\n]+)$', 'numbered'),  # Numbered sections
            (r'^(Chapter|Section|Part)\s+(\d+|[IVX]+)[:\.]?\s*(.*)$', 'formal'),  # Formal sections
        ]
        
        for pattern, format_type in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                if format_type == 'markdown':
                    level = len(match.group(1))
                    title = match.group(2)
                elif format_type == 'caps':
                    level = 1
                    title = match.group(1)
                elif format_type == 'numbered':
                    level = match.group(1).count('.') + 1
                    title = match.group(2)
                else:
                    level = 1
                    title = f"{match.group(1)} {match.group(2)}: {match.group(3) if len(match.groups()) > 2 else ''}"
                
                sections.append({
                    "type": "section",
                    "level": level,
                    "title": title.strip(),
                    "start_pos": match.start(),
                    "end_pos": match.end()
                })
        
        return sections
    
    def analyze_content_structure(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of document structure
        
        Args:
            text: Full document text
            
        Returns:
            Structured analysis of document content
        """
        return {
            "tables": self.extract_tables(text),
            "equations": self.extract_equations(text),
            "lists": self.extract_lists(text),
            "sections": self.extract_sections(text),
            "statistics": {
                "total_characters": len(text),
                "total_words": len(text.split()),
                "total_paragraphs": len([p for p in text.split('\n\n') if p.strip()]),
                "total_lines": len(text.split('\n'))
            }
        }
    
    def generate_table_summary(self, table: Dict[str, Any]) -> str:
        """Generate a natural language summary of a table using LLM"""
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst. Summarize the following table concisely, highlighting key data points and patterns."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this table:\n{table['content']}"
                    }
                ],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Table with {len(table.get('parsed_data', {}).get('rows', []))} rows"
    
    def generate_equation_explanation(self, equation: Dict[str, Any]) -> str:
        """Generate an explanation of a mathematical equation"""
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a mathematics expert. Explain the following equation in simple terms."
                    },
                    {
                        "role": "user",
                        "content": f"Explain this equation: {equation['content']}"
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Mathematical equation: {equation['content']}"
    
    def enrich_content(self, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich extracted content with AI-generated summaries
        
        Args:
            content_analysis: Output from analyze_content_structure
            
        Returns:
            Enriched content with summaries
        """
        enriched = content_analysis.copy()
        
        # Enrich tables (limit to first 5 to manage API costs)
        for i, table in enumerate(enriched.get("tables", [])[:5]):
            table["summary"] = self.generate_table_summary(table)
        
        # Enrich equations (limit to first 5)
        for i, equation in enumerate(enriched.get("equations", [])[:5]):
            equation["explanation"] = self.generate_equation_explanation(equation)
        
        return enriched


# Singleton instance
multimodal_processor = MultimodalProcessor()
