"""
Paper collection from arXiv for Deep Learning topics
"""
import arxiv
import os
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta
from tqdm import tqdm
import time

from ..config import settings

class PaperCollector:
    """Collect Deep Learning papers from arXiv"""
    
    def __init__(self):
        self.client = arxiv.Client()
        self.dl_keywords = [
            "deep learning", "neural networks", "convolutional neural networks",
            "recurrent neural networks", "transformer", "attention mechanism",
            "computer vision", "natural language processing", "reinforcement learning",
            "generative adversarial networks", "variational autoencoder",
            "graph neural networks", "self-supervised learning", "few-shot learning"
        ]
    
    def search_papers(self, max_papers: int = 100) -> List[Dict[str, Any]]:
        """Search for Deep Learning papers on arXiv"""
        papers = []
        
        print(f"Collecting {max_papers} Deep Learning papers from arXiv...")
        
        # Build search query
        query_parts = []
        for keyword in self.dl_keywords[:5]:  # Use top 5 keywords to avoid overly complex query
            query_parts.append(f'ti:"{keyword}" OR abs:"{keyword}"')
        
        query = f"({' OR '.join(query_parts)}) AND cat:cs.LG"
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_papers,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            for paper in tqdm(self.client.results(search), desc="Fetching papers", total=max_papers):
                paper_data = {
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'abstract': paper.summary,
                    'url': paper.entry_id,
                    'pdf_url': paper.pdf_url,
                    'published': paper.published.isoformat(),
                    'categories': paper.categories,
                    'arxiv_id': paper.entry_id.split('/')[-1]
                }
                papers.append(paper_data)
                
                # Add small delay to be respectful to arXiv
                time.sleep(0.1)
                
                if len(papers) >= max_papers:
                    break
                    
        except Exception as e:
            print(f"Error collecting papers: {e}")
            
        return papers
    
    def save_papers_metadata(self, papers: List[Dict[str, Any]], filename: str = None):
        """Save papers metadata to JSON file"""
        if filename is None:
            filename = os.path.join(settings.DATA_DIR, "papers_metadata.json")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(papers)} papers metadata to {filename}")
    
    def load_papers_metadata(self, filename: str = None) -> List[Dict[str, Any]]:
        """Load papers metadata from JSON file"""
        if filename is None:
            filename = os.path.join(settings.DATA_DIR, "papers_metadata.json")
        
        if not os.path.exists(filename):
            return []
        
        with open(filename, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        return papers
    
    def collect_and_save(self, max_papers: int = None) -> List[Dict[str, Any]]:
        """Collect papers and save metadata"""
        if max_papers is None:
            max_papers = settings.MAX_PAPERS
        
        papers = self.search_papers(max_papers)
        self.save_papers_metadata(papers)
        
        return papers

if __name__ == "__main__":
    collector = PaperCollector()
    papers = collector.collect_and_save()
    print(f"Successfully collected {len(papers)} papers") 