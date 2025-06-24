"""
Content extraction from papers for knowledge base creation
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional
from io import BytesIO
import PyPDF2
from tqdm import tqdm
import time

from ..config import settings

class ContentExtractor:
    """Extract and process content from research papers"""
    
    def __init__(self):
        self.max_content_length = 10000  # Limit content length
        self.min_content_length = 100    # Minimum useful content
    
    def download_pdf(self, pdf_url: str, arxiv_id: str) -> Optional[str]:
        """Download PDF from arXiv"""
        try:
            # Create papers directory
            os.makedirs(settings.PAPERS_DIR, exist_ok=True)
            
            filepath = os.path.join(settings.PAPERS_DIR, f"{arxiv_id}.pdf")
            
            # Skip if already downloaded
            if os.path.exists(filepath):
                return filepath
            
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return filepath
            
        except Exception as e:
            print(f"Error downloading {arxiv_id}: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text content from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from first 10 pages (introduction, methodology)
                max_pages = min(10, len(pdf_reader.pages))
                
                for page_num in range(max_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                # Clean and truncate text
                text = self._clean_text(text)
                
                if len(text) < self.min_content_length:
                    return None
                
                return text[:self.max_content_length]
                
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace('', '')  # Remove null characters
        text = text.replace('\x0c', ' ')  # Remove form feed
        
        # Remove URLs and email addresses
        import re
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def create_knowledge_chunks(self, paper_data: Dict[str, Any], content: str) -> List[Dict[str, Any]]:
        """Create knowledge chunks from paper content"""
        chunks = []
        
        # Chunk 1: Title + Abstract
        abstract_chunk = {
            'id': f"{paper_data['arxiv_id']}_abstract",
            'title': paper_data['title'],
            'content': f"Title: {paper_data['title']}\n\nAbstract: {paper_data['abstract']}",
            'source': paper_data['url'],
            'authors': paper_data['authors'],
            'categories': paper_data['categories'],
            'type': 'abstract'
        }
        chunks.append(abstract_chunk)
        
        # Chunk 2: Main content (if available)
        if content:
            content_chunk = {
                'id': f"{paper_data['arxiv_id']}_content",
                'title': paper_data['title'],
                'content': content,
                'source': paper_data['url'],
                'authors': paper_data['authors'],
                'categories': paper_data['categories'],
                'type': 'content'
            }
            chunks.append(content_chunk)
        
        return chunks
    
    def process_papers(self, papers: List[Dict[str, Any]], max_papers: int = None) -> List[Dict[str, Any]]:
        """Process papers and extract content"""
        if max_papers:
            papers = papers[:max_papers]
        
        knowledge_chunks = []
        
        print(f"Processing {len(papers)} papers...")
        
        for paper in tqdm(papers, desc="Processing papers"):
            try:
                # Always create abstract chunk
                chunks = self.create_knowledge_chunks(paper, None)
                knowledge_chunks.extend(chunks)
                
                # Try to download and extract full content (optional)
                if paper.get('pdf_url'):
                    pdf_path = self.download_pdf(paper['pdf_url'], paper['arxiv_id'])
                    
                    if pdf_path:
                        content = self.extract_text_from_pdf(pdf_path)
                        if content:
                            content_chunks = self.create_knowledge_chunks(paper, content)
                            # Only add content chunk (abstract already added)
                            knowledge_chunks.extend(content_chunks[1:])
                
                # Be respectful to servers
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing paper {paper.get('arxiv_id', 'unknown')}: {e}")
                continue
        
        return knowledge_chunks
    
    def save_knowledge_base(self, chunks: List[Dict[str, Any]], filename: str = None):
        """Save knowledge base to JSON"""
        if filename is None:
            filename = settings.KNOWLEDGE_BASE_FILE
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunks)} knowledge chunks to {filename}")

if __name__ == "__main__":
    from .paper_collector import PaperCollector
    
    # Load papers metadata
    collector = PaperCollector()
    papers = collector.load_papers_metadata()
    
    if not papers:
        print("No papers found. Run paper_collector.py first.")
        exit(1)
    
    # Extract content
    extractor = ContentExtractor()
    chunks = extractor.process_papers(papers, max_papers=20)  # Start with 20 for testing
    extractor.save_knowledge_base(chunks)
    
    print(f"Successfully processed {len(chunks)} knowledge chunks") 