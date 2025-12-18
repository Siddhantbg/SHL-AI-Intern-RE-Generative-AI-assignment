"""
Metrics calculator for recommendation system evaluation.

This module provides focused implementations of various evaluation metrics
including Mean Recall@K, Precision@K, and other recommendation system metrics.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a single metric calculation."""
    metric_name: str
    value: float
    k_value: Optional[int] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MetricsCalculator:
    """
    Calculator for various recommendation system evaluation metrics.
    
    Provides implementations of standard information retrieval and recommendation
    metrics including Recall@K, Precision@K, NDCG@K, and Mean Reciprocal Rank.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.supported_metrics = [
            'recall_at_k',
            'precision_at_k', 
            'f1_at_k',
            'ndcg_at_k',
            'mrr',
            'map_at_k',
            'hit_rate_at_k'
        ]
        
        logger.debug("MetricsCalculator initialized")
    
    def calculate_recall_at_k(self, predicted_items: List[str], 
                             relevant_items: List[str], 
                             k: int) -> MetricResult:
        """
        Calculate Recall@K metric.
        
        Recall@K = |{relevant items} ∩ {top-k predicted items}| / |{relevant items}|
        
        Args:
            predicted_items: List of predicted items in ranked order
            relevant_items: List of ground truth relevant items
            k: Number of top predictions to consider
            
        Returns:
            MetricResult with Recall@K value
        """
        if not relevant_items:
            return MetricResult(
                metric_name='recall_at_k',
                value=0.0,
                k_value=k,
                metadata={'reason': 'no_relevant_items'}
            )
        
        if not predicted_items:
            return MetricResult(
                metric_name='recall_at_k',
                value=0.0,
                k_value=k,
                metadata={'reason': 'no_predictions'}
            )
        
        # Get top-k predictions
        top_k_predictions = set(predicted_items[:k])
        relevant_set = set(relevant_items)
        
        # Calculate intersection
        relevant_in_top_k = len(top_k_predictions & relevant_set)
        
        # Calculate recall
        recall = relevant_in_top_k / len(relevant_set)
        
        return MetricResult(
            metric_name='recall_at_k',
            value=recall,
            k_value=k,
            metadata={
                'relevant_in_top_k': relevant_in_top_k,
                'total_relevant': len(relevant_set),
                'total_predictions': len(predicted_items),
                'k': k
            }
        )
    
    def calculate_precision_at_k(self, predicted_items: List[str], 
                                relevant_items: List[str], 
                                k: int) -> MetricResult:
        """
        Calculate Precision@K metric.
        
        Precision@K = |{relevant items} ∩ {top-k predicted items}| / k
        
        Args:
            predicted_items: List of predicted items in ranked order
            relevant_items: List of ground truth relevant items
            k: Number of top predictions to consider
            
        Returns:
            MetricResult with Precision@K value
        """
        if not relevant_items:
            return MetricResult(
                metric_name='precision_at_k',
                value=0.0,
                k_value=k,
                metadata={'reason': 'no_relevant_items'}
            )
        
        if not predicted_items:
            return MetricResult(
                metric_name='precision_at_k',
                value=0.0,
                k_value=k,
                metadata={'reason': 'no_predictions'}
            )
        
        # Get top-k predictions
        top_k_predictions = predicted_items[:k]
        relevant_set = set(relevant_items)
        
        # Calculate intersection
        relevant_in_top_k = len(set(top_k_predictions) & relevant_set)
        
        # Calculate precision
        precision = relevant_in_top_k / len(top_k_predictions) if top_k_predictions else 0.0
        
        return MetricResult(
            metric_name='precision_at_k',
            value=precision,
            k_value=k,
            metadata={
                'relevant_in_top_k': relevant_in_top_k,
                'total_relevant': len(relevant_set),
                'k_predictions': len(top_k_predictions),
                'k': k
            }
        )
    
    def calculate_f1_at_k(self, predicted_items: List[str], 
                         relevant_items: List[str], 
                         k: int) -> MetricResult:
        """
        Calculate F1@K metric (harmonic mean of Precision@K and Recall@K).
        
        Args:
            predicted_items: List of predicted items in ranked order
            relevant_items: List of ground truth relevant items
            k: Number of top predictions to consider
            
        Returns:
            MetricResult with F1@K value
        """
        recall_result = self.calculate_recall_at_k(predicted_items, relevant_items, k)
        precision_result = self.calculate_precision_at_k(predicted_items, relevant_items, k)
        
        recall = recall_result.value
        precision = precision_result.value
        
        # Calculate F1 score
        if recall + precision == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return MetricResult(
            metric_name='f1_at_k',
            value=f1,
            k_value=k,
            metadata={
                'recall': recall,
                'precision': precision,
                'k': k
            }
        )
    
    def calculate_ndcg_at_k(self, predicted_items: List[str], 
                           relevant_items: List[str], 
                           k: int,
                           relevance_scores: Optional[List[float]] = None) -> MetricResult:
        """
        Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            predicted_items: List of predicted items in ranked order
            relevant_items: List of ground truth relevant items
            k: Number of top predictions to consider
            relevance_scores: Optional relevance scores for predicted items
            
        Returns:
            MetricResult with NDCG@K value
        """
        if not relevant_items or not predicted_items:
            return MetricResult(
                metric_name='ndcg_at_k',
                value=0.0,
                k_value=k,
                metadata={'reason': 'no_items'}
            )
        
        relevant_set = set(relevant_items)
        top_k_predictions = predicted_items[:k]
        
        # Calculate DCG@K
        dcg = 0.0
        for i, item in enumerate(top_k_predictions):
            if item in relevant_set:
                # Binary relevance (1 if relevant, 0 if not)
                relevance = 1.0
                # Use provided relevance scores if available
                if relevance_scores and i < len(relevance_scores):
                    relevance = relevance_scores[i]
                
                # DCG formula: rel_i / log2(i + 2)
                dcg += relevance / np.log2(i + 2)
        
        # Calculate IDCG@K (ideal DCG)
        # Assume all relevant items have relevance score of 1.0
        ideal_relevances = [1.0] * min(len(relevant_items), k)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return MetricResult(
            metric_name='ndcg_at_k',
            value=ndcg,
            k_value=k,
            metadata={
                'dcg': dcg,
                'idcg': idcg,
                'k': k,
                'relevant_in_top_k': len(set(top_k_predictions) & relevant_set)
            }
        )
    
    def calculate_mrr(self, predicted_items: List[str], 
                     relevant_items: List[str]) -> MetricResult:
        """
        Calculate Mean Reciprocal Rank.
        
        MRR = 1 / rank_of_first_relevant_item
        
        Args:
            predicted_items: List of predicted items in ranked order
            relevant_items: List of ground truth relevant items
            
        Returns:
            MetricResult with MRR value
        """
        if not relevant_items or not predicted_items:
            return MetricResult(
                metric_name='mrr',
                value=0.0,
                metadata={'reason': 'no_items'}
            )
        
        relevant_set = set(relevant_items)
        
        # Find rank of first relevant item (1-indexed)
        for i, item in enumerate(predicted_items):
            if item in relevant_set:
                mrr = 1.0 / (i + 1)
                return MetricResult(
                    metric_name='mrr',
                    value=mrr,
                    metadata={
                        'first_relevant_rank': i + 1,
                        'first_relevant_item': item
                    }
                )
        
        # No relevant items found
        return MetricResult(
            metric_name='mrr',
            value=0.0,
            metadata={'reason': 'no_relevant_found'}
        )
    
    def calculate_map_at_k(self, predicted_items: List[str], 
                          relevant_items: List[str], 
                          k: int) -> MetricResult:
        """
        Calculate Mean Average Precision@K.
        
        Args:
            predicted_items: List of predicted items in ranked order
            relevant_items: List of ground truth relevant items
            k: Number of top predictions to consider
            
        Returns:
            MetricResult with MAP@K value
        """
        if not relevant_items or not predicted_items:
            return MetricResult(
                metric_name='map_at_k',
                value=0.0,
                k_value=k,
                metadata={'reason': 'no_items'}
            )
        
        relevant_set = set(relevant_items)
        top_k_predictions = predicted_items[:k]
        
        # Calculate average precision
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_predictions):
            if item in relevant_set:
                relevant_count += 1
                # Precision at position i+1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        # Average precision
        if relevant_count == 0:
            map_score = 0.0
        else:
            map_score = precision_sum / len(relevant_items)
        
        return MetricResult(
            metric_name='map_at_k',
            value=map_score,
            k_value=k,
            metadata={
                'relevant_found': relevant_count,
                'total_relevant': len(relevant_items),
                'precision_sum': precision_sum,
                'k': k
            }
        )
    
    def calculate_hit_rate_at_k(self, predicted_items: List[str], 
                               relevant_items: List[str], 
                               k: int) -> MetricResult:
        """
        Calculate Hit Rate@K (whether at least one relevant item is in top-K).
        
        Args:
            predicted_items: List of predicted items in ranked order
            relevant_items: List of ground truth relevant items
            k: Number of top predictions to consider
            
        Returns:
            MetricResult with Hit Rate@K value (0.0 or 1.0)
        """
        if not relevant_items or not predicted_items:
            return MetricResult(
                metric_name='hit_rate_at_k',
                value=0.0,
                k_value=k,
                metadata={'reason': 'no_items'}
            )
        
        relevant_set = set(relevant_items)
        top_k_predictions = set(predicted_items[:k])
        
        # Check if there's any intersection
        hit = 1.0 if len(top_k_predictions & relevant_set) > 0 else 0.0
        
        return MetricResult(
            metric_name='hit_rate_at_k',
            value=hit,
            k_value=k,
            metadata={
                'hit': hit == 1.0,
                'relevant_in_top_k': len(top_k_predictions & relevant_set),
                'k': k
            }
        )
    
    def calculate_mean_recall_at_k(self, predictions_list: List[List[str]], 
                                  relevant_lists: List[List[str]], 
                                  k: int) -> MetricResult:
        """
        Calculate Mean Recall@K across multiple queries.
        
        This is the primary metric mentioned in the requirements.
        
        Args:
            predictions_list: List of prediction lists (one per query)
            relevant_lists: List of relevant item lists (one per query)
            k: K value for recall calculation
            
        Returns:
            MetricResult with Mean Recall@K value
        """
        if len(predictions_list) != len(relevant_lists):
            raise ValueError("predictions_list and relevant_lists must have same length")
        
        if not predictions_list:
            return MetricResult(
                metric_name='mean_recall_at_k',
                value=0.0,
                k_value=k,
                metadata={'reason': 'no_queries'}
            )
        
        recall_scores = []
        
        for predictions, relevant in zip(predictions_list, relevant_lists):
            recall_result = self.calculate_recall_at_k(predictions, relevant, k)
            recall_scores.append(recall_result.value)
        
        mean_recall = np.mean(recall_scores)
        
        return MetricResult(
            metric_name='mean_recall_at_k',
            value=mean_recall,
            k_value=k,
            metadata={
                'individual_scores': recall_scores,
                'num_queries': len(predictions_list),
                'std_dev': np.std(recall_scores),
                'min_score': np.min(recall_scores),
                'max_score': np.max(recall_scores),
                'k': k
            }
        )
    
    def calculate_multiple_metrics(self, predicted_items: List[str], 
                                  relevant_items: List[str], 
                                  k_values: List[int] = None,
                                  metrics: List[str] = None) -> Dict[str, MetricResult]:
        """
        Calculate multiple metrics for a single query.
        
        Args:
            predicted_items: List of predicted items in ranked order
            relevant_items: List of ground truth relevant items
            k_values: List of K values to calculate metrics for
            metrics: List of metric names to calculate
            
        Returns:
            Dictionary mapping metric names to MetricResult objects
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        if metrics is None:
            metrics = ['recall_at_k', 'precision_at_k', 'f1_at_k', 'ndcg_at_k']
        
        results = {}
        
        # Calculate metrics that depend on K
        for metric in metrics:
            if metric.endswith('_at_k'):
                for k in k_values:
                    metric_key = f"{metric}_{k}"
                    
                    if metric == 'recall_at_k':
                        results[metric_key] = self.calculate_recall_at_k(predicted_items, relevant_items, k)
                    elif metric == 'precision_at_k':
                        results[metric_key] = self.calculate_precision_at_k(predicted_items, relevant_items, k)
                    elif metric == 'f1_at_k':
                        results[metric_key] = self.calculate_f1_at_k(predicted_items, relevant_items, k)
                    elif metric == 'ndcg_at_k':
                        results[metric_key] = self.calculate_ndcg_at_k(predicted_items, relevant_items, k)
                    elif metric == 'map_at_k':
                        results[metric_key] = self.calculate_map_at_k(predicted_items, relevant_items, k)
                    elif metric == 'hit_rate_at_k':
                        results[metric_key] = self.calculate_hit_rate_at_k(predicted_items, relevant_items, k)
        
        # Calculate metrics that don't depend on K
        if 'mrr' in metrics:
            results['mrr'] = self.calculate_mrr(predicted_items, relevant_items)
        
        return results
    
    def get_supported_metrics(self) -> List[str]:
        """
        Get list of supported metric names.
        
        Returns:
            List of supported metric names
        """
        return self.supported_metrics.copy()