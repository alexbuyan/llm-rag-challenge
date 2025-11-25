import os
import arxiv
from typing import List

def download_arxiv_papers(topics: List[str], max_results: int = 5, data_dir: str = "data/raw"):
    """
    Downloads papers from ArXiv based on the provided topics.
    
    Args:
        topics: List of search topics.
        max_results: Max papers per topic.
        data_dir: Directory to save PDFs.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    client = arxiv.Client()
    
    for topic in topics:
        print(f"Searching for: {topic}")
        search = arxiv.Search(
            query = topic,
            max_results = max_results,
            sort_by = arxiv.SortCriterion.Relevance
        )
        
        results = list(client.results(search))
        
        for paper in results:
            # Sanitize filename
            safe_title = "".join(x for x in paper.title if x.isalnum() or x in " -_").strip()
            filename = f"{safe_title}.pdf"
            filepath = os.path.join(data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Downloading: {paper.title}")
                paper.download_pdf(dirpath=data_dir, filename=filename)
            else:
                print(f"Skipping (already exists): {paper.title}")

if __name__ == "__main__":
    topics = ["Deep Learning", "LLM", "AI Agents", "Reinforcement Learning"]
    download_arxiv_papers(topics)

