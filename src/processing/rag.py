import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, Any, List, Optional
from rank_bm25 import BM25Okapi
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining vector search with BM25 keyword search."""
    
    def __init__(self, index, config: Dict):
        self.index = index
        self.config = config
        self.similarity_top_k = config.get("similarity_top_k", 5)
        self.bm25_weight = config.get("bm25_weight", 0.3)
        self.vector_weight = config.get("vector_weight", 0.7)
        self.bm25_index: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.node_map: Dict[int, Any] = {}
        
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents in the vector store."""
        try:
            # Get all nodes from the index's docstore
            docstore = self.index.docstore
            all_nodes = list(docstore.docs.values())
            
            if not all_nodes:
                logger.warning("No documents found for BM25 indexing")
                return
            
            # Tokenize documents for BM25
            self.corpus = []
            self.node_map = {}
            
            for i, node in enumerate(all_nodes):
                text = node.get_content().lower()
                tokens = text.split()
                self.corpus.append(tokens)
                self.node_map[i] = node
            
            self.bm25_index = BM25Okapi(self.corpus)
            logger.info(f"BM25 index built with {len(self.corpus)} documents")
            
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            self.bm25_index = None
    
    def _get_bm25_scores(self, query: str) -> Dict[int, float]:
        """Get BM25 scores for all documents."""
        if not self.bm25_index:
            return {}
        
        tokens = query.lower().split()
        scores = self.bm25_index.get_scores(tokens)
        
        # Normalize scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1
        return {i: score / max_score for i, score in enumerate(scores)}
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining vector similarity and BM25 scores.
        Returns ranked list of chunks with combined scores.
        """
        try:
            # Get vector search results
            vector_retriever = self.index.as_retriever(
                similarity_top_k=self.similarity_top_k * 2  # Get more for reranking
            )
            vector_nodes = vector_retriever.retrieve(query)
            
            # Get BM25 scores
            bm25_scores = self._get_bm25_scores(query)
            
            # Combine scores
            combined_results = []
            seen_contents = set()
            
            for node in vector_nodes:
                content = node.node.get_content().strip()
                content_hash = hash(content[:200])
                
                if content_hash in seen_contents:
                    continue
                seen_contents.add(content_hash)
                
                vector_score = node.score or 0.0
                
                # Find matching BM25 score
                bm25_score = 0.0
                node_content_lower = content.lower()
                for idx, tokens in enumerate(self.corpus):
                    if ' '.join(tokens)[:200] == node_content_lower[:200]:
                        bm25_score = bm25_scores.get(idx, 0.0)
                        break
                
                # Calculate combined score
                combined_score = (
                    self.vector_weight * vector_score + 
                    self.bm25_weight * bm25_score
                )
                
                combined_results.append({
                    "node": node,
                    "vector_score": vector_score,
                    "bm25_score": bm25_score,
                    "combined_score": combined_score,
                    "content": content
                })
            
            # Sort by combined score and return top-k
            combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
            return combined_results[:self.similarity_top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            # Fallback to vector-only retrieval
            return self._fallback_vector_retrieval(query)
    
    def _fallback_vector_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Fallback to simple vector retrieval if hybrid fails."""
        try:
            retriever = self.index.as_retriever(
                similarity_top_k=self.similarity_top_k
            )
            nodes = retriever.retrieve(query)
            
            return [{
                "node": node,
                "vector_score": node.score or 0.0,
                "bm25_score": 0.0,
                "combined_score": node.score or 0.0,
                "content": node.node.get_content().strip()
            } for node in nodes]
            
        except Exception as e:
            logger.error(f"Error in fallback retrieval: {e}")
            return []


class InterviewRAGService:
    def __init__(self, index, config: Dict):
        self.index = index
        self.config = config
        self.query_engine = None
        self.hybrid_retriever = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize query engine and hybrid retriever."""
        try:
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=self.config.get("similarity_top_k", 4),
                response_mode="tree_summarize",
                temperature=0.1,
                verbose=True
            )
            
            # Initialize hybrid retriever
            self.hybrid_retriever = HybridRetriever(self.index, self.config)
            
            logger.info("Query engine and hybrid retriever initialized")
        except Exception as e:
            logger.error(f"Error initializing components: {e}")

    def get_study_context(self, query: str, use_hybrid: bool = True) -> Dict[str, Any]:
        """
        Get relevant context for studying.
        
        Args:
            query: The search query
            use_hybrid: Whether to use hybrid retrieval (default: True)
        """
        if not self.query_engine:
            return {"error": "Query engine not initialized"}

        try:
            # Use hybrid retrieval if available and requested
            if use_hybrid and self.hybrid_retriever:
                results = self.hybrid_retriever.retrieve(query)
            else:
                # Fallback to simple vector retrieval
                retriever = self.index.as_retriever(
                    similarity_top_k=self.config.get("similarity_top_k", 4)
                )
                nodes = retriever.retrieve(query)
                results = [{
                    "node": node,
                    "vector_score": node.score or 0.0,
                    "bm25_score": 0.0,
                    "combined_score": node.score or 0.0,
                    "content": node.node.get_content().strip()
                } for node in nodes]

            # Structure the context
            context = {
                "query": query,
                "total_chunks": len(results),
                "chunks": [],
                "topics": set(),
                "sources": set(),
                "retrieval_method": "hybrid" if use_hybrid else "vector"
            }

            for i, result in enumerate(results, 1):
                node = result["node"]
                content = result["content"]
                metadata = node.node.metadata

                chunk_info = {
                    "id": i,
                    "score": float(result["combined_score"]),
                    "vector_score": float(result["vector_score"]),
                    "bm25_score": float(result["bm25_score"]),
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "full_content": content,
                    "metadata": {
                        "title": metadata.get('title', 'Unknown'),
                        "source": metadata.get('source', 'Unknown'),
                        "url": metadata.get('url', ''),
                        "topic": metadata.get('topic', ''),
                        "file_name": metadata.get('file_name', ''),
                        "page_label": metadata.get('page_label', '')
                    }
                }

                context["chunks"].append(chunk_info)
                if metadata.get('topic'):
                    context["topics"].add(metadata['topic'])
                if metadata.get('source'):
                    context["sources"].add(metadata['source'])

            context["topics"] = list(context["topics"])
            context["sources"] = list(context["sources"])

            return context

        except Exception as e:
            logger.error(f"Error getting study context: {e}")
            return {"error": str(e)}

    def generate_study_guide(self, query: str, context: Dict[str, Any]) -> str:
        """Generate a structured study guide."""
        if "error" in context:
            return f"Error: {context['error']}"

        guide_parts = []

        # Header
        guide_parts.append("=" * 80)
        guide_parts.append(f"INTERVIEW PREPARATION GUIDE: {query.upper()}")
        guide_parts.append("=" * 80)

        # Overview
        guide_parts.append("\n📋 OVERVIEW")
        guide_parts.append("-" * 40)
        guide_parts.append(f"• Found {context['total_chunks']} relevant information chunks")
        guide_parts.append(f"• Sources: {', '.join(context['sources'])}")
        guide_parts.append(f"• Topics covered: {', '.join(context['topics'])}")
        guide_parts.append(f"• Retrieval method: {context.get('retrieval_method', 'vector')}")

        # Key questions
        guide_parts.append("\n❓ KEY QUESTIONS TO PREPARE")
        guide_parts.append("-" * 40)

        key_questions = self._extract_key_questions(context)
        for i, question in enumerate(key_questions, 1):
            guide_parts.append(f"{i}. {question}")

        # Key concepts
        guide_parts.append("\n💡 KEY CONCEPTS TO UNDERSTAND")
        guide_parts.append("-" * 40)

        concepts = self._extract_key_concepts(context)
        for concept in concepts:
            guide_parts.append(f"• {concept}")

        # Relevant materials
        guide_parts.append("\n📚 RELEVANT STUDY MATERIALS")
        guide_parts.append("-" * 40)

        for chunk in context["chunks"]:
            guide_parts.append(f"\n[Source: {chunk['metadata']['source']}]")
            guide_parts.append(f"Title: {chunk['metadata']['title']}")
            if chunk['metadata']['url']:
                guide_parts.append(f"URL: {chunk['metadata']['url']}")
            guide_parts.append(f"Combined score: {chunk['score']:.3f} (vector: {chunk['vector_score']:.3f}, bm25: {chunk['bm25_score']:.3f})")
            guide_parts.append(f"Content:\n{chunk['content']}\n")

        # Practical tips
        guide_parts.append("\n🎯 PRACTICAL TIPS")
        guide_parts.append("-" * 40)
        guide_parts.append("1. Practice explaining each concept out loud")
        guide_parts.append("2. Create flashcards for key terms")
        guide_parts.append("3. Solve related coding problems")
        guide_parts.append("4. Prepare real-world examples")
        guide_parts.append("5. Review system design trade-offs")

        # Next steps
        guide_parts.append("\n🚀 NEXT STEPS")
        guide_parts.append("-" * 40)
        guide_parts.append("1. Review the materials above")
        guide_parts.append("2. Practice with mock interviews")
        guide_parts.append("3. Update your knowledge gaps")
        guide_parts.append("4. Prepare your success stories")

        return "\n".join(guide_parts)

    def _extract_key_questions(self, context: Dict) -> List[str]:
        """Extract key questions from context."""
        questions = []

        base_questions = [
            f"What is {context['query']} and how does it work?",
            f"Explain the main components of {context['query']}",
            f"What are the advantages and disadvantages of {context['query']}?",
            f"How would you implement {context['query']} in a real system?",
            f"What are common use cases for {context['query']}?"
        ]

        for chunk in context["chunks"]:
            content = chunk["full_content"].lower()

            if '?' in content:
                sentences = content.split('.')
                for sentence in sentences:
                    if '?' in sentence and len(sentence.split()) > 5:
                        question = sentence.strip()
                        if question not in questions and len(questions) < 10:
                            questions.append(question[:200])

        if not questions:
            questions = base_questions[:5]

        return questions[:7]

    def _extract_key_concepts(self, context: Dict) -> List[str]:
        """Extract key concepts from context."""
        concepts = set()

        base_concepts = [
            "Time and Space Complexity",
            "System Architecture",
            "Design Patterns",
            "Best Practices",
            "Trade-offs and Optimization",
            "Scalability Considerations",
            "Security Implications"
        ]

        for chunk in context["chunks"]:
            metadata = chunk["metadata"]
            if metadata.get('topic'):
                concepts.add(metadata['topic'].title())

        for concept in base_concepts:
            concepts.add(concept)

        return list(concepts)[:10]


# Backward compatibility class
class RAGService:
    def __init__(self, persist_dir: str = "data/processed"):
        self.persist_dir = persist_dir
        self.rag_service = None

    def initialize(self):
        from src.processing.index import load_existing_index
        index = load_existing_index(self.persist_dir)
        if not index:
            return False

        config = {"similarity_top_k": 4}
        self.rag_service = InterviewRAGService(index, config)
        return True

    def query(self, question: str) -> str:
        if not self.rag_service:
            if not self.initialize():
                return "Error: Failed to initialize RAG service"

        context = self.rag_service.get_study_context(question)
        return self.rag_service.generate_study_guide(question, context)
