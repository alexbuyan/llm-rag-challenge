"""
RAG Evaluation Module

Provides comprehensive evaluation metrics for the RAG pipeline including:
- Retrieval metrics (Precision@k, Recall@k, MRR)
- RAGAS metrics (Faithfulness, Answer Relevance, Context Precision/Recall)
- Human evaluation interface
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Retrieval Metrics
# =============================================================================

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Precision@k.
    
    Args:
        retrieved: List of retrieved document identifiers
        relevant: List of relevant document identifiers
        k: Number of top results to consider
    
    Returns:
        Precision@k score (0-1)
    """
    if k <= 0 or not retrieved:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant_set)
    
    return relevant_retrieved / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Recall@k.
    
    Args:
        retrieved: List of retrieved document identifiers
        relevant: List of relevant document identifiers
        k: Number of top results to consider
    
    Returns:
        Recall@k score (0-1)
    """
    if not relevant:
        return 0.0
    
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    relevant_retrieved = len(retrieved_at_k & relevant_set)
    
    return relevant_retrieved / len(relevant_set)


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved: List of retrieved document identifiers
        relevant: List of relevant document identifiers
    
    Returns:
        MRR score (0-1)
    """
    relevant_set = set(relevant)
    
    for rank, doc in enumerate(retrieved, 1):
        if doc in relevant_set:
            return 1.0 / rank
    
    return 0.0


def calculate_retrieval_metrics(
    retrieved: List[str], 
    relevant: List[str], 
    k_values: List[int] = [3, 5, 10]
) -> Dict[str, float]:
    """
    Calculate all retrieval metrics.
    
    Args:
        retrieved: List of retrieved document identifiers
        relevant: List of relevant document identifiers
        k_values: List of k values for P@k and R@k
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "mrr": mean_reciprocal_rank(retrieved, relevant)
    }
    
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
    
    return metrics


# =============================================================================
# RAGAS Metrics
# =============================================================================

class RAGASEvaluator:
    """RAGAS-based evaluation for RAG systems."""
    
    def __init__(self):
        self.ragas_available = False
        self._init_ragas()
    
    def _init_ragas(self):
        """Initialize RAGAS if available."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            from datasets import Dataset
            
            self.evaluate = evaluate
            self.faithfulness = faithfulness
            self.answer_relevancy = answer_relevancy
            self.context_precision = context_precision
            self.context_recall = context_recall
            self.Dataset = Dataset
            self.ragas_available = True
        except ImportError as e:
            print(f"RAGAS not available: {e}")
            self.ragas_available = False
    
    def evaluate_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG sample using RAGAS metrics.
        
        Args:
            question: The input question
            answer: The generated answer
            contexts: List of retrieved context strings
            ground_truth: Optional ground truth answer
        
        Returns:
            Dictionary with RAGAS metrics
        """
        if not self.ragas_available:
            return {"error": "RAGAS not available"}
        
        try:
            # Prepare dataset for RAGAS
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            
            if ground_truth:
                data["ground_truth"] = [ground_truth]
            
            dataset = self.Dataset.from_dict(data)
            
            # Select metrics based on available data
            metrics = [self.faithfulness, self.answer_relevancy]
            if ground_truth:
                metrics.extend([self.context_precision, self.context_recall])
            
            # Run evaluation
            results = self.evaluate(dataset, metrics=metrics)
            
            return {
                "faithfulness": float(results.get("faithfulness", 0)),
                "answer_relevancy": float(results.get("answer_relevancy", 0)),
                "context_precision": float(results.get("context_precision", 0)) if ground_truth else None,
                "context_recall": float(results.get("context_recall", 0)) if ground_truth else None,
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of RAG samples.
        
        Args:
            samples: List of dictionaries with question, answer, contexts, ground_truth
        
        Returns:
            Aggregated metrics and per-sample results
        """
        if not self.ragas_available:
            return {"error": "RAGAS not available"}
        
        results = []
        for sample in samples:
            result = self.evaluate_sample(
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample["contexts"],
                ground_truth=sample.get("ground_truth")
            )
            results.append(result)
        
        # Aggregate metrics
        aggregated = {}
        metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        
        for key in metric_keys:
            values = [r.get(key) for r in results if r.get(key) is not None and "error" not in r]
            if values:
                aggregated[f"avg_{key}"] = sum(values) / len(values)
        
        return {
            "aggregated": aggregated,
            "per_sample": results
        }


# =============================================================================
# Human Evaluation Interface
# =============================================================================

class HumanEvaluator:
    """Simple CLI interface for human evaluation."""
    
    def __init__(self, output_dir: str = "data/evaluation/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
    
    def rate_response(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        query_id: str = ""
    ) -> Dict[str, Any]:
        """
        Interactive rating of a single response.
        
        Args:
            query: The input query
            answer: The generated answer
            contexts: Retrieved contexts
            query_id: Optional identifier for the query
        
        Returns:
            Rating dictionary
        """
        print("\n" + "=" * 60)
        print("HUMAN EVALUATION")
        print("=" * 60)
        
        print(f"\n📝 QUERY: {query}")
        print(f"\n📚 CONTEXTS ({len(contexts)} retrieved):")
        for i, ctx in enumerate(contexts[:3], 1):
            print(f"  [{i}] {ctx[:200]}...")
        
        print(f"\n💬 ANSWER:\n{answer}")
        
        print("\n" + "-" * 40)
        print("Please rate the following (1-5 scale):")
        print("  1 = Very Poor, 2 = Poor, 3 = Acceptable, 4 = Good, 5 = Excellent")
        print("-" * 40)
        
        ratings = {}
        
        # Get ratings for each dimension
        dimensions = [
            ("relevance", "Relevance (Does the answer address the query?)"),
            ("accuracy", "Accuracy (Is the information correct?)"),
            ("completeness", "Completeness (Is the answer thorough?)"),
            ("coherence", "Coherence (Is the answer well-structured?)"),
        ]
        
        for dim_key, dim_prompt in dimensions:
            while True:
                try:
                    rating = input(f"\n{dim_prompt}: ")
                    rating = int(rating)
                    if 1 <= rating <= 5:
                        ratings[dim_key] = rating
                        break
                    else:
                        print("Please enter a number between 1 and 5")
                except ValueError:
                    print("Please enter a valid number")
                except KeyboardInterrupt:
                    print("\nSkipping remaining ratings...")
                    return {"skipped": True}
        
        # Optional comments
        comments = input("\n📝 Additional comments (optional, press Enter to skip): ").strip()
        
        result = {
            "query_id": query_id,
            "query": query,
            "ratings": ratings,
            "average_rating": sum(ratings.values()) / len(ratings),
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(result)
        return result
    
    def run_batch_evaluation(
        self,
        samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run human evaluation on a batch of samples.
        
        Args:
            samples: List of samples with query, answer, contexts
            max_samples: Maximum number of samples to evaluate
        
        Returns:
            Aggregated results
        """
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"\nStarting human evaluation of {len(samples)} samples...")
        print("Press Ctrl+C to stop early and save results.\n")
        
        try:
            for i, sample in enumerate(samples, 1):
                print(f"\n[{i}/{len(samples)}]")
                self.rate_response(
                    query=sample["query"],
                    answer=sample.get("answer", "No answer generated"),
                    contexts=sample.get("contexts", []),
                    query_id=sample.get("id", f"sample_{i}")
                )
        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted. Saving results...")
        
        return self.get_aggregated_results()
    
    def get_aggregated_results(self) -> Dict[str, Any]:
        """Get aggregated results from all evaluations."""
        if not self.results:
            return {"error": "No evaluations completed"}
        
        # Filter out skipped results
        valid_results = [r for r in self.results if not r.get("skipped")]
        
        if not valid_results:
            return {"error": "No valid evaluations"}
        
        # Aggregate by dimension
        dimensions = ["relevance", "accuracy", "completeness", "coherence"]
        aggregated = {}
        
        for dim in dimensions:
            values = [r["ratings"].get(dim) for r in valid_results if r["ratings"].get(dim)]
            if values:
                aggregated[f"avg_{dim}"] = sum(values) / len(values)
        
        # Overall average
        all_averages = [r["average_rating"] for r in valid_results]
        aggregated["overall_average"] = sum(all_averages) / len(all_averages)
        aggregated["total_evaluated"] = len(valid_results)
        
        return {
            "aggregated": aggregated,
            "per_sample": valid_results
        }
    
    def save_results(self, filename: str = "human_eval_results.json"):
        """Save evaluation results to file."""
        filepath = self.output_dir / filename
        results = self.get_aggregated_results()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


# =============================================================================
# Main Evaluator Class
# =============================================================================

class RAGEvaluator:
    """
    Main evaluation class combining all metrics.
    """
    
    def __init__(
        self,
        validation_path: str = "data/evaluation/validation_set.json",
        results_dir: str = "data/evaluation/results"
    ):
        self.validation_path = Path(validation_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.ragas_evaluator = RAGASEvaluator()
        self.human_evaluator = HumanEvaluator(str(self.results_dir))
        
        self.validation_set = self._load_validation_set()
    
    def _load_validation_set(self) -> List[Dict[str, Any]]:
        """Load validation queries from JSON."""
        if not self.validation_path.exists():
            print(f"Warning: Validation set not found at {self.validation_path}")
            return []
        
        with open(self.validation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get("queries", [])
    
    def evaluate_retrieval(
        self,
        rag_service,
        queries: Optional[List[Dict[str, Any]]] = None,
        k_values: List[int] = [3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance.
        
        Args:
            rag_service: The RAG service to evaluate
            queries: Optional list of queries (uses validation set if not provided)
            k_values: K values for P@k and R@k
        
        Returns:
            Retrieval metrics
        """
        queries = queries or self.validation_set
        if not queries:
            return {"error": "No queries to evaluate"}
        
        all_metrics = []
        per_query_results = []
        
        for query_data in queries:
            query = query_data["query"]
            expected_sources = query_data.get("expected_sources", [])
            
            # Get retrieval results
            context = rag_service.get_study_context(query)
            
            if "error" in context:
                continue
            
            # Extract retrieved source names from file_name metadata
            retrieved_sources = []
            for chunk in context.get("chunks", []):
                metadata = chunk.get("metadata", {})
                # Try file_name first (from PDF indexing), then title, then source
                file_name = metadata.get("file_name", "") or metadata.get("title", "") or metadata.get("source", "")
                
                if file_name:
                    # Try to match to expected source files
                    matched = False
                    for source in expected_sources:
                        # Normalize both for comparison (remove extension, lowercase)
                        source_norm = source.lower().replace(".pdf", "").strip()
                        file_norm = file_name.lower().replace(".pdf", "").strip()
                        
                        if source_norm in file_norm or file_norm in source_norm:
                            if source not in retrieved_sources:
                                retrieved_sources.append(source)
                            matched = True
                            break
                    
                    if not matched and file_name not in retrieved_sources:
                        retrieved_sources.append(file_name)
            
            # Calculate metrics
            metrics = calculate_retrieval_metrics(
                retrieved=retrieved_sources,
                relevant=expected_sources,
                k_values=k_values
            )
            
            all_metrics.append(metrics)
            per_query_results.append({
                "query_id": query_data.get("id"),
                "query": query,
                "expected": expected_sources,
                "retrieved": retrieved_sources[:10],
                "metrics": metrics
            })
        
        # Aggregate metrics
        aggregated = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                aggregated[f"avg_{key}"] = sum(values) / len(values)
        
        return {
            "aggregated": aggregated,
            "per_query": per_query_results,
            "total_queries": len(per_query_results)
        }
    
    def evaluate_generation(
        self,
        rag_service,
        queries: Optional[List[Dict[str, Any]]] = None,
        max_queries: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate generation quality using RAGAS.
        
        Args:
            rag_service: The RAG service to evaluate
            queries: Optional list of queries
            max_queries: Maximum number of queries to evaluate (RAGAS can be slow)
        
        Returns:
            RAGAS metrics
        """
        if not self.ragas_evaluator.ragas_available:
            return {"error": "RAGAS not available"}
        
        queries = (queries or self.validation_set)[:max_queries]
        if not queries:
            return {"error": "No queries to evaluate"}
        
        samples = []
        for query_data in queries:
            query = query_data["query"]
            ground_truth = query_data.get("reference_answer")
            
            # Get context and generate answer
            context = rag_service.get_study_context(query)
            if "error" in context:
                continue
            
            # Extract contexts as strings
            contexts = [chunk["full_content"] for chunk in context.get("chunks", [])]
            
            # Generate answer (use study guide as answer)
            answer = rag_service.generate_study_guide(query, context)
            
            samples.append({
                "question": query,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            })
        
        return self.ragas_evaluator.evaluate_batch(samples)
    
    def run_human_evaluation(
        self,
        rag_service,
        queries: Optional[List[Dict[str, Any]]] = None,
        max_queries: int = 5
    ) -> Dict[str, Any]:
        """
        Run interactive human evaluation.
        
        Args:
            rag_service: The RAG service to evaluate
            queries: Optional list of queries
            max_queries: Maximum number of queries to evaluate
        
        Returns:
            Human evaluation results
        """
        queries = (queries or self.validation_set)[:max_queries]
        if not queries:
            return {"error": "No queries to evaluate"}
        
        samples = []
        for query_data in queries:
            query = query_data["query"]
            
            context = rag_service.get_study_context(query)
            if "error" in context:
                continue
            
            contexts = [chunk["content"] for chunk in context.get("chunks", [])]
            answer = rag_service.generate_study_guide(query, context)
            
            samples.append({
                "id": query_data.get("id"),
                "query": query,
                "answer": answer,
                "contexts": contexts
            })
        
        results = self.human_evaluator.run_batch_evaluation(samples)
        self.human_evaluator.save_results()
        
        return results
    
    def run_full_evaluation(
        self,
        rag_service,
        include_human: bool = False,
        max_ragas_queries: int = 10
    ) -> Dict[str, Any]:
        """
        Run full evaluation suite.
        
        Args:
            rag_service: The RAG service to evaluate
            include_human: Whether to include human evaluation
            max_ragas_queries: Max queries for RAGAS (can be slow)
        
        Returns:
            Complete evaluation results
        """
        print("=" * 60)
        print("RAG EVALUATION SUITE")
        print("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_validation_queries": len(self.validation_set)
        }
        
        # Retrieval metrics
        print("\n[1/3] Evaluating retrieval...")
        results["retrieval"] = self.evaluate_retrieval(rag_service)
        print(f"  Completed: {results['retrieval'].get('total_queries', 0)} queries")
        
        # RAGAS metrics
        print("\n[2/3] Evaluating generation (RAGAS)...")
        results["ragas"] = self.evaluate_generation(
            rag_service, 
            max_queries=max_ragas_queries
        )
        if "error" in results["ragas"]:
            print(f"  Skipped: {results['ragas']['error']}")
        else:
            print(f"  Completed")
        
        # Human evaluation
        if include_human:
            print("\n[3/3] Starting human evaluation...")
            results["human"] = self.run_human_evaluation(rag_service)
        else:
            print("\n[3/3] Human evaluation skipped (use include_human=True)")
            results["human"] = {"skipped": True}
        
        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"evaluation_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResults saved to: {filepath}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        # Retrieval metrics
        if "retrieval" in results and "aggregated" in results["retrieval"]:
            print("\n📊 RETRIEVAL METRICS:")
            for key, value in results["retrieval"]["aggregated"].items():
                print(f"  {key}: {value:.4f}")
        
        # RAGAS metrics
        if "ragas" in results and "aggregated" in results.get("ragas", {}):
            print("\n📊 RAGAS METRICS:")
            for key, value in results["ragas"]["aggregated"].items():
                if value is not None:
                    print(f"  {key}: {value:.4f}")
        
        # Human metrics
        if "human" in results and "aggregated" in results.get("human", {}):
            print("\n📊 HUMAN EVALUATION:")
            for key, value in results["human"]["aggregated"].items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def run_evaluation(
    persist_dir: str = "data/processed",
    validation_path: str = "data/evaluation/validation_set.json",
    include_human: bool = False,
    max_ragas_queries: int = 10
):
    """
    Run evaluation from command line.
    
    Args:
        persist_dir: Path to persisted index
        validation_path: Path to validation set
        include_human: Whether to include human evaluation
        max_ragas_queries: Max queries for RAGAS evaluation
    """
    from src.processing.index import InterviewIndexer
    from src.processing.rag import InterviewRAGService
    
    # Initialize indexer and load index
    config = {
        "persist_dir": persist_dir,
        "use_openai_embeddings": False
    }
    indexer = InterviewIndexer(config)
    index = indexer.load_existing_index()
    
    if not index:
        print("Error: Could not load index. Run indexing first.")
        return None
    
    # Initialize RAG service
    rag_config = {"similarity_top_k": 5}
    rag_service = InterviewRAGService(index, rag_config)
    
    # Run evaluation
    evaluator = RAGEvaluator(validation_path=validation_path)
    results = evaluator.run_full_evaluation(
        rag_service,
        include_human=include_human,
        max_ragas_queries=max_ragas_queries
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Evaluation")
    parser.add_argument("--persist-dir", default="data/processed", help="Index directory")
    parser.add_argument("--validation", default="data/evaluation/validation_set.json", help="Validation set path")
    parser.add_argument("--human", action="store_true", help="Include human evaluation")
    parser.add_argument("--max-ragas", type=int, default=10, help="Max queries for RAGAS")
    
    args = parser.parse_args()
    
    run_evaluation(
        persist_dir=args.persist_dir,
        validation_path=args.validation,
        include_human=args.human,
        max_ragas_queries=args.max_ragas
    )

