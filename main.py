import argparse
import sys
from src.processing.gather import download_arxiv_papers
from src.processing.index import build_index
from src.processing.rag import RAGService

def main():
    parser = argparse.ArgumentParser(description="AI Agent RAG Pipeline")
    
    # Define arguments
    parser.add_argument("--gather", action="store_true", help="Gather data from ArXiv")
    parser.add_argument("--index", action="store_true", help="Build/Update the index")
    parser.add_argument("--query", type=str, help="Query the RAG system")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # 1. Gather Data
    if args.gather:
        print("Starting data gathering...")
        topics = ["Deep Learning", "Large Language Models", "AI Agents", "Reinforcement Learning", "NLP"]
        download_arxiv_papers(topics, max_results=3)
        print("Data gathering complete.")

    # 2. Index Data
    if args.index:
        print("Starting indexing...")
        build_index()
        print("Indexing complete.")

    # 3. Query RAG
    if args.query:
        print(f"Querying: {args.query}")
        rag = RAGService()
        response = rag.query(args.query)
        print("\n=== Response ===\n")
        print(response)
        print("\n================\n")

if __name__ == "__main__":
    main()
