"""
Evaluation engine for measuring recommendation system performance.

This module provides the EvaluationEngine class that measures recommendation accuracy
using Mean Recall@K and other metrics, processes training datasets, and tracks
performance improvements over optimization iterations.
"""

import logging
import json
import csv
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import time

from ..recommendation.recommendation_engine import RecommendationEngine, RecommendationResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuery:
    """Represents a single evaluation query with ground truth labels."""
    query_id: str
    query_text: str
    ground_truth_urls: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EvaluationResult:
    """Results from evaluating a single query."""
    query_id: str
    query_text: str
    predicted_urls: List[str]
    ground_truth_urls: List[str]
    recall_at_k: Dict[int, float]  # Recall@K for different K values
    precision_at_k: Dict[int, float]  # Precision@K for different K values
    relevance_scores: List[float]
    processing_time: float
    recommendation_result: RecommendationResult


@dataclass
class EvaluationSummary:
    """Summary of evaluation results across all queries."""
    total_queries: int
    mean_recall_at_k: Dict[int, float]
    mean_precision_at_k: Dict[int, float]
    mean_processing_time: float
    evaluation_timestamp: str
    configuration: Dict[str, Any]
    individual_results: List[EvaluationResult]


class EvaluationEngine:
    """
    Engine for measuring recommendation system performance using various metrics.
    
    Supports loading training datasets, calculating Mean Recall@K, tracking
    performance over optimization iterations, and generating detailed reports.
    """
    
    def __init__(self, recommendation_engine: RecommendationEngine,
                 data_dir: str = "data/evaluation"):
        """
        Initialize the evaluation engine.
        
        Args:
            recommendation_engine: The recommendation engine to evaluate
            data_dir: Directory containing evaluation datasets
        """
        self.recommendation_engine = recommendation_engine
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation configuration
        self.k_values = [1, 3, 5, 10]  # K values for Recall@K and Precision@K
        self.max_recommendations = 10  # Maximum recommendations to request
        
        # Performance tracking
        self.evaluation_history: List[EvaluationSummary] = []
        
        logger.info(f"EvaluationEngine initialized with data_dir: {self.data_dir}")
    
    def load_training_dataset(self, file_path: Optional[str] = None) -> List[EvaluationQuery]:
        """
        Load training dataset with labeled queries.
        
        Args:
            file_path: Path to training dataset file. If None, looks for default files.
            
        Returns:
            List of EvaluationQuery objects
            
        Raises:
            FileNotFoundError: If training dataset file is not found
            ValueError: If dataset format is invalid
        """
        if file_path is None:
            # Try common file names
            possible_files = [
                self.data_dir / "training_data.json",
                self.data_dir / "training_data.csv",
                self.data_dir / "train.json",
                self.data_dir / "train.csv"
            ]
            
            file_path = None
            for possible_file in possible_files:
                if possible_file.exists():
                    file_path = possible_file
                    break
            
            if file_path is None:
                raise FileNotFoundError(
                    f"No training dataset found. Looked for: {[str(f) for f in possible_files]}"
                )
        else:
            file_path = Path(file_path)
        
        logger.info(f"Loading training dataset from: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            return self._load_json_dataset(file_path)
        elif file_path.suffix.lower() == '.csv':
            return self._load_csv_dataset(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_json_dataset(self, file_path: Path) -> List[EvaluationQuery]:
        """Load dataset from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            queries = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of query objects
                for i, item in enumerate(data):
                    query = self._parse_query_item(item, str(i))
                    queries.append(query)
            elif isinstance(data, dict):
                if 'queries' in data:
                    # Nested structure with queries key
                    for i, item in enumerate(data['queries']):
                        query = self._parse_query_item(item, str(i))
                        queries.append(query)
                else:
                    # Single query object
                    query = self._parse_query_item(data, "0")
                    queries.append(query)
            
            logger.info(f"Loaded {len(queries)} queries from JSON dataset")
            return queries
            
        except Exception as e:
            raise ValueError(f"Failed to load JSON dataset: {str(e)}")
    
    def _load_csv_dataset(self, file_path: Path) -> List[EvaluationQuery]:
        """Load dataset from CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Expected columns: query_id, query_text, ground_truth_urls
            required_columns = ['query_text', 'ground_truth_urls']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            queries = []
            
            for i, row in df.iterrows():
                query_id = str(row.get('query_id', i))
                query_text = str(row['query_text'])
                
                # Parse ground truth URLs (could be comma-separated string or list)
                ground_truth = row['ground_truth_urls']
                if isinstance(ground_truth, str):
                    # Split comma-separated URLs and clean them
                    ground_truth_urls = [url.strip() for url in ground_truth.split(',') if url.strip()]
                else:
                    ground_truth_urls = [str(ground_truth)] if ground_truth else []
                
                # Extract metadata from other columns
                metadata = {}
                for col in df.columns:
                    if col not in ['query_id', 'query_text', 'ground_truth_urls']:
                        metadata[col] = row[col]
                
                query = EvaluationQuery(
                    query_id=query_id,
                    query_text=query_text,
                    ground_truth_urls=ground_truth_urls,
                    metadata=metadata
                )
                queries.append(query)
            
            logger.info(f"Loaded {len(queries)} queries from CSV dataset")
            return queries
            
        except Exception as e:
            raise ValueError(f"Failed to load CSV dataset: {str(e)}")
    
    def _parse_query_item(self, item: Dict[str, Any], default_id: str) -> EvaluationQuery:
        """Parse a single query item from JSON data."""
        query_id = str(item.get('query_id', item.get('id', default_id)))
        query_text = str(item.get('query_text', item.get('query', item.get('text', ''))))
        
        if not query_text:
            raise ValueError(f"Missing query text for query_id: {query_id}")
        
        # Parse ground truth URLs
        ground_truth = item.get('ground_truth_urls', item.get('ground_truth', item.get('urls', [])))
        if isinstance(ground_truth, str):
            ground_truth_urls = [url.strip() for url in ground_truth.split(',') if url.strip()]
        elif isinstance(ground_truth, list):
            ground_truth_urls = [str(url) for url in ground_truth]
        else:
            ground_truth_urls = []
        
        # Extract metadata
        metadata = {k: v for k, v in item.items() 
                   if k not in ['query_id', 'id', 'query_text', 'query', 'text', 
                               'ground_truth_urls', 'ground_truth', 'urls']}
        
        return EvaluationQuery(
            query_id=query_id,
            query_text=query_text,
            ground_truth_urls=ground_truth_urls,
            metadata=metadata
        )
    
    def evaluate_single_query(self, evaluation_query: EvaluationQuery) -> EvaluationResult:
        """
        Evaluate recommendation system performance on a single query.
        
        Args:
            evaluation_query: Query with ground truth labels
            
        Returns:
            EvaluationResult with metrics and predictions
        """
        logger.debug(f"Evaluating query: {evaluation_query.query_id}")
        
        start_time = time.time()
        
        # Get recommendations from the system
        recommendation_result = self.recommendation_engine.recommend(
            query=evaluation_query.query_text,
            max_results=self.max_recommendations
        )
        
        processing_time = time.time() - start_time
        
        # Extract predicted URLs
        predicted_urls = [rec.assessment_url for rec in recommendation_result.recommendations]
        relevance_scores = [rec.relevance_score for rec in recommendation_result.recommendations]
        
        # Calculate metrics
        recall_at_k = self._calculate_recall_at_k(
            predicted_urls, evaluation_query.ground_truth_urls, self.k_values
        )
        
        precision_at_k = self._calculate_precision_at_k(
            predicted_urls, evaluation_query.ground_truth_urls, self.k_values
        )
        
        result = EvaluationResult(
            query_id=evaluation_query.query_id,
            query_text=evaluation_query.query_text,
            predicted_urls=predicted_urls,
            ground_truth_urls=evaluation_query.ground_truth_urls,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            relevance_scores=relevance_scores,
            processing_time=processing_time,
            recommendation_result=recommendation_result
        )
        
        logger.debug(f"Query {evaluation_query.query_id} - Recall@5: {recall_at_k.get(5, 0):.3f}")
        
        return result
    
    def evaluate_dataset(self, evaluation_queries: List[EvaluationQuery]) -> EvaluationSummary:
        """
        Evaluate recommendation system on a complete dataset.
        
        Args:
            evaluation_queries: List of queries with ground truth labels
            
        Returns:
            EvaluationSummary with aggregated metrics
        """
        logger.info(f"Evaluating dataset with {len(evaluation_queries)} queries")
        
        individual_results = []
        total_processing_time = 0.0
        
        for query in evaluation_queries:
            try:
                result = self.evaluate_single_query(query)
                individual_results.append(result)
                total_processing_time += result.processing_time
                
            except Exception as e:
                logger.error(f"Failed to evaluate query {query.query_id}: {str(e)}")
                # Create empty result for failed query
                empty_result = EvaluationResult(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    predicted_urls=[],
                    ground_truth_urls=query.ground_truth_urls,
                    recall_at_k={k: 0.0 for k in self.k_values},
                    precision_at_k={k: 0.0 for k in self.k_values},
                    relevance_scores=[],
                    processing_time=0.0,
                    recommendation_result=None
                )
                individual_results.append(empty_result)
        
        # Calculate mean metrics
        mean_recall_at_k = self._calculate_mean_metrics(
            individual_results, 'recall_at_k', self.k_values
        )
        
        mean_precision_at_k = self._calculate_mean_metrics(
            individual_results, 'precision_at_k', self.k_values
        )
        
        mean_processing_time = total_processing_time / len(individual_results) if individual_results else 0.0
        
        # Get system configuration
        configuration = self.recommendation_engine.get_recommendation_stats()
        
        summary = EvaluationSummary(
            total_queries=len(evaluation_queries),
            mean_recall_at_k=mean_recall_at_k,
            mean_precision_at_k=mean_precision_at_k,
            mean_processing_time=mean_processing_time,
            evaluation_timestamp=datetime.utcnow().isoformat(),
            configuration=configuration,
            individual_results=individual_results
        )
        
        # Add to evaluation history
        self.evaluation_history.append(summary)
        
        logger.info(f"Evaluation complete - Mean Recall@5: {mean_recall_at_k.get(5, 0):.3f}")
        
        return summary
    
    def _calculate_recall_at_k(self, predicted_urls: List[str], 
                              ground_truth_urls: List[str], 
                              k_values: List[int]) -> Dict[int, float]:
        """
        Calculate Recall@K for different K values.
        
        Recall@K = (Number of relevant items in top-K) / (Total number of relevant items)
        
        Args:
            predicted_urls: List of predicted URLs in ranked order
            ground_truth_urls: List of ground truth relevant URLs
            k_values: List of K values to calculate recall for
            
        Returns:
            Dictionary mapping K values to Recall@K scores
        """
        if not ground_truth_urls:
            return {k: 0.0 for k in k_values}
        
        ground_truth_set = set(ground_truth_urls)
        recall_scores = {}
        
        for k in k_values:
            # Get top-K predictions
            top_k_predictions = set(predicted_urls[:k])
            
            # Count relevant items in top-K
            relevant_in_top_k = len(top_k_predictions & ground_truth_set)
            
            # Calculate recall
            recall_at_k = relevant_in_top_k / len(ground_truth_set)
            recall_scores[k] = recall_at_k
        
        return recall_scores
    
    def _calculate_precision_at_k(self, predicted_urls: List[str], 
                                 ground_truth_urls: List[str], 
                                 k_values: List[int]) -> Dict[int, float]:
        """
        Calculate Precision@K for different K values.
        
        Precision@K = (Number of relevant items in top-K) / K
        
        Args:
            predicted_urls: List of predicted URLs in ranked order
            ground_truth_urls: List of ground truth relevant URLs
            k_values: List of K values to calculate precision for
            
        Returns:
            Dictionary mapping K values to Precision@K scores
        """
        if not ground_truth_urls:
            return {k: 0.0 for k in k_values}
        
        ground_truth_set = set(ground_truth_urls)
        precision_scores = {}
        
        for k in k_values:
            # Get top-K predictions
            top_k_predictions = predicted_urls[:k]
            
            if not top_k_predictions:
                precision_scores[k] = 0.0
                continue
            
            # Count relevant items in top-K
            relevant_in_top_k = len(set(top_k_predictions) & ground_truth_set)
            
            # Calculate precision
            precision_at_k = relevant_in_top_k / len(top_k_predictions)
            precision_scores[k] = precision_at_k
        
        return precision_scores
    
    def _calculate_mean_metrics(self, results: List[EvaluationResult], 
                               metric_name: str, k_values: List[int]) -> Dict[int, float]:
        """Calculate mean metrics across all evaluation results."""
        mean_metrics = {}
        
        for k in k_values:
            scores = []
            for result in results:
                metric_dict = getattr(result, metric_name)
                scores.append(metric_dict.get(k, 0.0))
            
            mean_metrics[k] = np.mean(scores) if scores else 0.0
        
        return mean_metrics
    
    def calculate_mean_recall_at_k(self, evaluation_results: List[EvaluationResult], 
                                  k: int = 5) -> float:
        """
        Calculate Mean Recall@K across multiple evaluation results.
        
        This is the primary metric mentioned in the requirements.
        
        Args:
            evaluation_results: List of evaluation results
            k: K value for recall calculation (default: 5)
            
        Returns:
            Mean Recall@K score
        """
        recall_scores = []
        
        for result in evaluation_results:
            recall_at_k = result.recall_at_k.get(k, 0.0)
            recall_scores.append(recall_at_k)
        
        mean_recall = np.mean(recall_scores) if recall_scores else 0.0
        
        logger.info(f"Mean Recall@{k}: {mean_recall:.4f} (across {len(recall_scores)} queries)")
        
        return mean_recall
    
    def save_evaluation_results(self, summary: EvaluationSummary, 
                               output_file: Optional[str] = None) -> str:
        """
        Save evaluation results to file.
        
        Args:
            summary: Evaluation summary to save
            output_file: Output file path. If None, generates timestamp-based name.
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"evaluation_results_{timestamp}.json"
        else:
            output_file = Path(output_file)
        
        # Convert summary to serializable format
        summary_dict = asdict(summary)
        
        # Convert dict keys to strings for JSON serialization
        summary_dict['mean_recall_at_k'] = {str(k): v for k, v in summary_dict['mean_recall_at_k'].items()}
        summary_dict['mean_precision_at_k'] = {str(k): v for k, v in summary_dict['mean_precision_at_k'].items()}
        
        results_data = {
            'summary': summary_dict,
            'individual_results': []
        }
        
        # Convert individual results (excluding recommendation_result which may not be serializable)
        for result in summary.individual_results:
            result_dict = asdict(result)
            # Convert dict keys to strings for JSON serialization
            result_dict['recall_at_k'] = {str(k): v for k, v in result_dict['recall_at_k'].items()}
            result_dict['precision_at_k'] = {str(k): v for k, v in result_dict['precision_at_k'].items()}
            # Remove non-serializable recommendation_result
            result_dict.pop('recommendation_result', None)
            # Convert numpy arrays to lists if present
            for key, value in result_dict.items():
                if isinstance(value, np.ndarray):
                    result_dict[key] = value.tolist()
            results_data['individual_results'].append(result_dict)
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {output_file}")
        
        return str(output_file)
    
    def load_evaluation_results(self, file_path: str) -> EvaluationSummary:
        """
        Load evaluation results from file.
        
        Args:
            file_path: Path to evaluation results file
            
        Returns:
            EvaluationSummary object
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct EvaluationSummary (without recommendation_result objects)
        summary_data = data['summary']
        individual_results = []
        
        for result_data in data['individual_results']:
            # Convert string keys back to integers
            recall_at_k = {int(k): v for k, v in result_data['recall_at_k'].items()}
            precision_at_k = {int(k): v for k, v in result_data['precision_at_k'].items()}
            
            result = EvaluationResult(
                query_id=result_data['query_id'],
                query_text=result_data['query_text'],
                predicted_urls=result_data['predicted_urls'],
                ground_truth_urls=result_data['ground_truth_urls'],
                recall_at_k=recall_at_k,
                precision_at_k=precision_at_k,
                relevance_scores=result_data['relevance_scores'],
                processing_time=result_data['processing_time'],
                recommendation_result=None  # Not saved/loaded
            )
            individual_results.append(result)
        
        # Convert string keys back to integers
        mean_recall_at_k = {int(k): v for k, v in summary_data['mean_recall_at_k'].items()}
        mean_precision_at_k = {int(k): v for k, v in summary_data['mean_precision_at_k'].items()}
        
        summary = EvaluationSummary(
            total_queries=summary_data['total_queries'],
            mean_recall_at_k=mean_recall_at_k,
            mean_precision_at_k=mean_precision_at_k,
            mean_processing_time=summary_data['mean_processing_time'],
            evaluation_timestamp=summary_data['evaluation_timestamp'],
            configuration=summary_data['configuration'],
            individual_results=individual_results
        )
        
        return summary
    
    def track_performance_improvement(self, baseline_summary: EvaluationSummary, 
                                    current_summary: EvaluationSummary) -> Dict[str, Any]:
        """
        Track performance improvements between two evaluation runs.
        
        Args:
            baseline_summary: Baseline evaluation results
            current_summary: Current evaluation results
            
        Returns:
            Dictionary with improvement metrics
        """
        improvements = {}
        
        # Compare Mean Recall@K
        recall_improvements = {}
        for k in self.k_values:
            baseline_recall = baseline_summary.mean_recall_at_k.get(k, 0.0)
            current_recall = current_summary.mean_recall_at_k.get(k, 0.0)
            
            improvement = current_recall - baseline_recall
            improvement_pct = (improvement / baseline_recall * 100) if baseline_recall > 0 else 0.0
            
            recall_improvements[f'recall_at_{k}'] = {
                'baseline': baseline_recall,
                'current': current_recall,
                'absolute_improvement': improvement,
                'percentage_improvement': improvement_pct
            }
        
        improvements['recall_metrics'] = recall_improvements
        
        # Compare Mean Precision@K
        precision_improvements = {}
        for k in self.k_values:
            baseline_precision = baseline_summary.mean_precision_at_k.get(k, 0.0)
            current_precision = current_summary.mean_precision_at_k.get(k, 0.0)
            
            improvement = current_precision - baseline_precision
            improvement_pct = (improvement / baseline_precision * 100) if baseline_precision > 0 else 0.0
            
            precision_improvements[f'precision_at_{k}'] = {
                'baseline': baseline_precision,
                'current': current_precision,
                'absolute_improvement': improvement,
                'percentage_improvement': improvement_pct
            }
        
        improvements['precision_metrics'] = precision_improvements
        
        # Compare processing time
        time_improvement = baseline_summary.mean_processing_time - current_summary.mean_processing_time
        time_improvement_pct = (time_improvement / baseline_summary.mean_processing_time * 100) if baseline_summary.mean_processing_time > 0 else 0.0
        
        improvements['processing_time'] = {
            'baseline': baseline_summary.mean_processing_time,
            'current': current_summary.mean_processing_time,
            'absolute_improvement': time_improvement,
            'percentage_improvement': time_improvement_pct
        }
        
        # Overall assessment
        primary_metric_improvement = recall_improvements.get('recall_at_5', {}).get('absolute_improvement', 0.0)
        improvements['overall_assessment'] = {
            'primary_metric': 'recall_at_5',
            'primary_improvement': primary_metric_improvement,
            'is_improvement': primary_metric_improvement > 0,
            'baseline_timestamp': baseline_summary.evaluation_timestamp,
            'current_timestamp': current_summary.evaluation_timestamp
        }
        
        logger.info(f"Performance tracking complete - Primary improvement: {primary_metric_improvement:.4f}")
        
        return improvements
    
    def get_evaluation_history(self) -> List[EvaluationSummary]:
        """
        Get the history of evaluation runs.
        
        Returns:
            List of EvaluationSummary objects in chronological order
        """
        return self.evaluation_history.copy()
    
    def generate_evaluation_report(self, summary: EvaluationSummary) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            summary: Evaluation summary to generate report for
            
        Returns:
            Formatted evaluation report as string
        """
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("SHL ASSESSMENT RECOMMENDATION SYSTEM - EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Basic information
        report_lines.append(f"Evaluation Timestamp: {summary.evaluation_timestamp}")
        report_lines.append(f"Total Queries Evaluated: {summary.total_queries}")
        report_lines.append(f"Mean Processing Time: {summary.mean_processing_time:.3f} seconds")
        report_lines.append("")
        
        # Recall metrics
        report_lines.append("RECALL METRICS:")
        report_lines.append("-" * 20)
        for k in sorted(summary.mean_recall_at_k.keys()):
            recall = summary.mean_recall_at_k[k]
            report_lines.append(f"Mean Recall@{k:2d}: {recall:.4f}")
        report_lines.append("")
        
        # Precision metrics
        report_lines.append("PRECISION METRICS:")
        report_lines.append("-" * 20)
        for k in sorted(summary.mean_precision_at_k.keys()):
            precision = summary.mean_precision_at_k[k]
            report_lines.append(f"Mean Precision@{k:2d}: {precision:.4f}")
        report_lines.append("")
        
        # Individual query performance
        report_lines.append("INDIVIDUAL QUERY PERFORMANCE:")
        report_lines.append("-" * 35)
        report_lines.append(f"{'Query ID':<15} {'Recall@5':<10} {'Precision@5':<12} {'Time (s)':<10}")
        report_lines.append("-" * 50)
        
        for result in summary.individual_results:
            recall_5 = result.recall_at_k.get(5, 0.0)
            precision_5 = result.precision_at_k.get(5, 0.0)
            report_lines.append(
                f"{result.query_id:<15} {recall_5:<10.4f} {precision_5:<12.4f} {result.processing_time:<10.3f}"
            )
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)