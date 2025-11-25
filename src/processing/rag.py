import os

# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.processing.index import load_existing_index

class RAGService:
    def __init__(self, persist_dir: str = "data/processed"):
        self.persist_dir = persist_dir
        self.index = None
        self.retriever = None

    def initialize(self):
        """Loads the index and sets up the retriever."""
        self.index = load_existing_index(self.persist_dir)
        if self.index:
            # Configure retriever to fetch top 3 similar chunks
            self.retriever = self.index.as_retriever(similarity_top_k=3)
            return True
        return False

    def query(self, question: str) -> str:
        """
        Retrieves relevant chunks for the question.
        
        Args:
            question: The user's question.
            
        Returns:
            The formatted string of retrieved chunks.
        """
        if not self.retriever:
            success = self.initialize()
            if not success:
                return "Error: knowledge base not initialized or found. Please run indexing first."
        
        nodes = self.retriever.retrieve(question)
        
        # Format the output
        response_parts = [f"Found {len(nodes)} relevant chunks:\n"]
        for i, node in enumerate(nodes, 1):
            score = f"{node.score:.4f}" if node.score else "N/A"
            content = node.node.get_content().strip()
            # truncate content if too long for display
            display_content = content[:500] + "..." if len(content) > 500 else content
            
            response_parts.append(f"--- Chunk {i} (Score: {score}) ---")
            response_parts.append(display_content)
            response_parts.append(f"[Source: {node.node.metadata.get('file_name', 'unknown')}]\n")
            
        return "\n".join(response_parts)

# Simple standalone test
if __name__ == "__main__":
    service = RAGService()
    service.initialize()
    print(service.query("What are the latest trends in LLMs?"))

