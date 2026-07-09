"""
Evaluation Metrics for Information Retrieval
"""
import numpy as np
from typing import Dict, List


class IRMetrics:
    """Information Retrieval evaluation metrics."""
    
    @staticmethod
    def recall_at_k(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]], k: int) -> float:
        """
        Calculate Recall@k: fraction of relevant documents found in top-k.
        
        Args:
            results: {query_id: [doc_ids]}
            qrels: {query_id: {doc_id: relevance_score}}
            k: cutoff for top-k evaluation
        """
        recall_scores = []
        for q_id in results:
            if q_id not in qrels:
                continue
            
            # YOUR CODE HERE: Calculate recall for this query
            good_docs = [doc for doc, score in qrels[q_id].items() if score > 0]
            if len(good_docs) == 0:
                continue

            docs_in_k = results[q_id][:k]
            found = 0
            for doc in docs_in_k:
                if doc in good_docs:
                    found += 1

            recall_scores.append(found / len(good_docs))

        return np.mean(recall_scores) if recall_scores else 0.0

    @staticmethod
    def precision_at_k(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]], k: int) -> float:
        """
        Calculate Precision@k: fraction of top-k that are relevant.
        """
        precision_scores = []
        for q_id in results:
            if q_id not in qrels:
                continue
            
            # YOUR CODE HERE: Calculate precision for this query
            good_docs = [doc for doc, score in qrels[q_id].items() if score > 0]

            docs_in_k = results[q_id][:k]
            found = 0
            for doc in docs_in_k:
                if doc in good_docs:
                    found += 1

            precision_scores.append(found / k)
            
        return np.mean(precision_scores) if precision_scores else 0.0

    @staticmethod
    def mrr(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        """
        reciprocal_ranks = []
        for q_id in results:
            if q_id not in qrels:
                continue
            
            # YOUR CODE HERE: Calculate MRR for this query
            good_docs = [doc for doc, score in qrels[q_id].items() if score > 0]

            rank_found = 0
            for spot, doc in enumerate(results[q_id]):
                if doc in good_docs:
                    rank_found = spot + 1
                    break

            if rank_found > 0:
                reciprocal_ranks.append(1 / rank_found)
            else:
                reciprocal_ranks.append(0)
            
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def evaluate_retrieval(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]]) -> Dict[str, float]:
        """
        Comprehensive evaluation with standard IR metrics.
        
        Returns:
            Dictionary with metric names and values
        """
        metrics = {
            'Recall@1': IRMetrics.recall_at_k(results, qrels, 1),
            'Recall@5': IRMetrics.recall_at_k(results, qrels, 5), 
            'Recall@10': IRMetrics.recall_at_k(results, qrels, 10),
            'Precision@5': IRMetrics.precision_at_k(results, qrels, 5),
            'MRR': IRMetrics.mrr(results, qrels)
        }
        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
        """Pretty print evaluation metrics."""
        print(f"\n📊 {title}")
        print("=" * 40)
        for metric, value in metrics.items():
            print(f"{metric:12}: {value:.4f}")
        print("=" * 40)
