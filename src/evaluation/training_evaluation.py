"""
Training data evaluation and optimization module.

This module provides functionality to load training data, run evaluation loops,
and perform comprehensive optimization of the recommendation system.
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .evaluation_engine import EvaluationEngine, EvaluationQuery, EvaluationSummary
from .optimization_pipeline import OptimizationPipeline, OptimizationConfig
from .performance_tracker import PerformanceTracker
from ..recommendation.recommendation_engine import RecommendationEngine
from ..recommendation.query_processor import QueryProcessor
from ..processing.vector_database import VectorDatabase

logger = logging.getLogger(__name__)


class TrainingEvaluator:
    """
    Main class for training data evaluation and optimization.
    
    Coordinates the evaluation of the recommendation system using training data
    and performs comprehensive optimization to improve performance.
    """
    
    def __init__(self, 
                 recommendation_engine: RecommendationEngine,
                 training_data_path: str = "data/evaluation/sample_training_data.json",
                 results_dir: str = "data/evaluation/results"):
        """
        Initialize the training evaluator.
        
        Args:
            recommendation_engine: The recommendation engine to evaluate and optimize
            training_data_path: Path to training data file
            results_dir: Directory to save evaluation results
        """
        self.recommendation_engine = recommendation_engine
        self.training_data_path = Path(training_data_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluation components
        self.evaluation_engine = EvaluationEngine(recommendation_engine)
        self.performance_tracker = PerformanceTracker()
        self.optimization_pipeline = OptimizationPipeline(
            recommendation_engine=recommendation_engine,
            evaluation_engine=self.evaluation_engine,
            performance_tracker=self.performance_tracker
        )
        
        # Training data
        self.training_queries: Optional[List[EvaluationQuery]] = None
        
        logger.info(f"TrainingEvaluator initialized with training data: {self.training_data_path}")
    
    def load_training_data(self) -> List[EvaluationQuery]:
        """
        Load the training dataset.
        
        Returns:
            List of training queries
        """
        logger.info("Loading training dataset")
        
        if not self.training_data_path.exists():
            raise FileNotFoundError(f"Training data file not found: {self.training_data_path}")
        
        self.training_queries = self.evaluation_engine.load_training_dataset(str(self.training_data_path))
        
        logger.info(f"Loaded {len(self.training_queries)} training queries")
        
        # Log training data summary
        self._log_training_data_summary()
        
        return self.training_queries
    
    def _log_training_data_summary(self) -> None:
        """Log summary information about the training data."""
        if not self.training_queries:
            return
        
        # Analyze training data characteristics
        job_roles = {}
        difficulties = {}
        avg_ground_truth_count = 0
        
        for query in self.training_queries:
            # Count job roles
            job_role = query.metadata.get('job_role', 'unknown')
            job_roles[job_role] = job_roles.get(job_role, 0) + 1
            
            # Count difficulties
            difficulty = query.metadata.get('difficulty', 'unknown')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
            
            # Count ground truth assessments
            avg_ground_truth_count += len(query.ground_truth_urls)
        
        avg_ground_truth_count /= len(self.training_queries)
        
        logger.info("Training Data Summary:")
        logger.info(f"  Total queries: {len(self.training_queries)}")
        logger.info(f"  Average ground truth assessments per query: {avg_ground_truth_count:.1f}")
        logger.info(f"  Job roles distribution: {job_roles}")
        logger.info(f"  Difficulty distribution: {difficulties}")
    
    def run_baseline_evaluation(self) -> EvaluationSummary:
        """
        Run baseline evaluation with current system configuration.
        
        Returns:
            Baseline evaluation summary
        """
        if not self.training_queries:
            self.load_training_data()
        
        logger.info("Running baseline evaluation")
        
        # Evaluate current system performance
        baseline_summary = self.evaluation_engine.evaluate_dataset(self.training_queries)
        
        # Save baseline results
        baseline_file = self.results_dir / f"baseline_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.evaluation_engine.save_evaluation_results(baseline_summary, str(baseline_file))
        
        # Log baseline performance
        self._log_evaluation_summary("BASELINE", baseline_summary)
        
        return baseline_summary
    
    def run_optimization_loop(self, 
                            optimization_config: Optional[OptimizationConfig] = None) -> Dict[str, Any]:
        """
        Run comprehensive optimization loop to improve system performance.
        
        Args:
            optimization_config: Configuration for optimization experiments
            
        Returns:
            Dictionary with optimization results
        """
        if not self.training_queries:
            self.load_training_data()
        
        logger.info("Starting optimization loop")
        
        # Update optimization pipeline configuration if provided
        if optimization_config:
            self.optimization_pipeline.config = optimization_config
        
        # Run full optimization pipeline
        optimization_results = self.optimization_pipeline.run_full_optimization(self.training_queries)
        
        # Save optimization results
        results_file = self.results_dir / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.optimization_pipeline.save_optimization_results(optimization_results, str(results_file))
        
        # Generate and save optimization report
        report = self.optimization_pipeline.generate_optimization_report()
        report_file = self.results_dir / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Optimization completed. Results saved to: {results_file}")
        logger.info(f"Optimization report saved to: {report_file}")
        
        return optimization_results
    
    def run_hyperparameter_tuning(self) -> Dict[str, Any]:
        """
        Run focused hyperparameter tuning.
        
        Returns:
            Hyperparameter tuning results
        """
        if not self.training_queries:
            self.load_training_data()
        
        logger.info("Running hyperparameter tuning")
        
        # Establish baseline first
        self.optimization_pipeline.establish_baseline(self.training_queries)
        
        # Run hyperparameter optimization
        hyperopt_results = self.optimization_pipeline.optimize_hyperparameters(self.training_queries)
        
        # Log results
        best_params = hyperopt_results.get('best_parameters', {})
        improvement = hyperopt_results.get('improvement_over_baseline', 0.0)
        
        logger.info("Hyperparameter Tuning Results:")
        logger.info(f"  Best parameters: {best_params}")
        logger.info(f"  Improvement over baseline: {improvement:+.4f}")
        logger.info(f"  Experiments conducted: {hyperopt_results.get('experiments_conducted', 0)}")
        
        return hyperopt_results
    
    def run_embedding_model_comparison(self) -> Dict[str, Any]:
        """
        Compare different embedding models.
        
        Returns:
            Embedding model comparison results
        """
        if not self.training_queries:
            self.load_training_data()
        
        logger.info("Running embedding model comparison")
        
        # Establish baseline first
        self.optimization_pipeline.establish_baseline(self.training_queries)
        
        # Run embedding model optimization
        embedding_results = self.optimization_pipeline.optimize_embedding_model(self.training_queries)
        
        # Log results
        best_model = embedding_results.get('best_model')
        improvement = embedding_results.get('improvement_over_baseline', 0.0)
        
        logger.info("Embedding Model Comparison Results:")
        logger.info(f"  Best model: {best_model}")
        logger.info(f"  Improvement over baseline: {improvement:+.4f}")
        logger.info(f"  Models tested: {embedding_results.get('models_tested', 0)}")
        
        return embedding_results
    
    def run_prompt_engineering_experiments(self) -> Dict[str, Any]:
        """
        Run prompt engineering experiments.
        
        Returns:
            Prompt engineering results
        """
        if not self.training_queries:
            self.load_training_data()
        
        logger.info("Running prompt engineering experiments")
        
        # Establish baseline first
        self.optimization_pipeline.establish_baseline(self.training_queries)
        
        # Run prompt optimization
        prompt_results = self.optimization_pipeline.optimize_prompt_engineering(self.training_queries)
        
        # Log results
        best_prompt = prompt_results.get('best_prompt')
        improvement = prompt_results.get('improvement_over_baseline', 0.0)
        
        logger.info("Prompt Engineering Results:")
        logger.info(f"  Best prompt template: {best_prompt}")
        logger.info(f"  Improvement over baseline: {improvement:+.4f}")
        logger.info(f"  Prompts tested: {prompt_results.get('prompts_tested', 0)}")
        
        return prompt_results
    
    def evaluate_individual_queries(self) -> List[Dict[str, Any]]:
        """
        Evaluate system performance on individual training queries.
        
        Returns:
            List of individual query evaluation results
        """
        if not self.training_queries:
            self.load_training_data()
        
        logger.info("Evaluating individual queries")
        
        individual_results = []
        
        for i, query in enumerate(self.training_queries):
            logger.info(f"Evaluating query {i+1}/{len(self.training_queries)}: {query.query_id}")
            
            try:
                # Evaluate single query
                result = self.evaluation_engine.evaluate_single_query(query)
                
                # Create summary for this query
                query_summary = {
                    'query_id': query.query_id,
                    'query_text': query.query_text,
                    'ground_truth_count': len(query.ground_truth_urls),
                    'predicted_count': len(result.predicted_urls),
                    'recall_at_5': result.recall_at_k.get(5, 0.0),
                    'precision_at_5': result.precision_at_k.get(5, 0.0),
                    'processing_time': result.processing_time,
                    'top_predictions': result.predicted_urls[:5],
                    'ground_truth_urls': query.ground_truth_urls,
                    'metadata': query.metadata
                }
                
                individual_results.append(query_summary)
                
                # Log individual result
                logger.info(f"  Query {query.query_id} - Recall@5: {result.recall_at_k.get(5, 0.0):.3f}, "
                           f"Precision@5: {result.precision_at_k.get(5, 0.0):.3f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate query {query.query_id}: {str(e)}")
                continue
        
        # Save individual results
        results_file = self.results_dir / f"individual_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(individual_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Individual query results saved to: {results_file}")
        
        return individual_results
    
    def generate_performance_analysis(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance analysis.
        
        Returns:
            Performance analysis results
        """
        logger.info("Generating performance analysis")
        
        analysis = {}
        
        # Get performance tracking history
        tracking_history = self.performance_tracker.get_evaluation_history()
        
        if tracking_history:
            # Analyze optimization trends
            trend_analysis = self.performance_tracker.get_optimization_trend()
            analysis['optimization_trend'] = trend_analysis
            
            # Get best iteration
            best_iteration = self.performance_tracker.get_best_iteration()
            if best_iteration:
                analysis['best_iteration'] = {
                    'iteration_id': best_iteration.iteration_id,
                    'mean_recall_at_5': best_iteration.evaluation_summary.mean_recall_at_k.get(5, 0.0),
                    'configuration': best_iteration.configuration,
                    'timestamp': best_iteration.timestamp
                }
            
            # Performance improvement summary
            if len(tracking_history) >= 2:
                first_iteration = tracking_history[0]
                last_iteration = tracking_history[-1]
                
                improvement = self.performance_tracker.get_performance_improvement(
                    first_iteration.iteration_id,
                    last_iteration.iteration_id
                )
                
                analysis['overall_improvement'] = {
                    'from_iteration': improvement.from_iteration,
                    'to_iteration': improvement.to_iteration,
                    'metric_improvements': improvement.metric_improvements,
                    'significant_improvements': improvement.significant_improvements,
                    'degradations': improvement.degradations
                }
        
        # System configuration analysis
        current_config = self.recommendation_engine.get_recommendation_stats()
        analysis['current_configuration'] = current_config
        
        # Generate performance report
        performance_report = self.performance_tracker.generate_performance_report()
        analysis['performance_report'] = performance_report
        
        # Save analysis
        analysis_file = self.results_dir / f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Performance analysis saved to: {analysis_file}")
        
        return analysis
    
    def _log_evaluation_summary(self, label: str, summary: EvaluationSummary) -> None:
        """Log evaluation summary information."""
        logger.info(f"{label} EVALUATION RESULTS:")
        logger.info(f"  Total queries: {summary.total_queries}")
        logger.info(f"  Mean processing time: {summary.mean_processing_time:.3f}s")
        
        for k in sorted(summary.mean_recall_at_k.keys()):
            recall = summary.mean_recall_at_k[k]
            precision = summary.mean_precision_at_k.get(k, 0.0)
            logger.info(f"  Mean Recall@{k}: {recall:.4f}, Mean Precision@{k}: {precision:.4f}")
    
    def run_complete_evaluation_and_optimization(self) -> Dict[str, Any]:
        """
        Run complete evaluation and optimization workflow.
        
        Returns:
            Complete results dictionary
        """
        logger.info("Starting complete evaluation and optimization workflow")
        
        results = {}
        
        try:
            # Step 1: Load training data
            self.load_training_data()
            results['training_data_loaded'] = True
            results['training_queries_count'] = len(self.training_queries)
            
            # Step 2: Run baseline evaluation
            baseline_summary = self.run_baseline_evaluation()
            results['baseline_evaluation'] = {
                'mean_recall_at_5': baseline_summary.mean_recall_at_k.get(5, 0.0),
                'mean_processing_time': baseline_summary.mean_processing_time,
                'total_queries': baseline_summary.total_queries
            }
            
            # Step 3: Run individual query analysis
            individual_results = self.evaluate_individual_queries()
            results['individual_query_analysis'] = {
                'queries_evaluated': len(individual_results),
                'average_recall_at_5': sum(r['recall_at_5'] for r in individual_results) / len(individual_results) if individual_results else 0.0
            }
            
            # Step 4: Run optimization loop
            optimization_results = self.run_optimization_loop()
            results['optimization_results'] = optimization_results
            
            # Step 5: Generate performance analysis
            performance_analysis = self.generate_performance_analysis()
            results['performance_analysis'] = performance_analysis
            
            # Step 6: Calculate final improvements
            if 'final_evaluation' in optimization_results:
                final_recall = optimization_results['final_evaluation']['mean_recall_at_5']
                baseline_recall = results['baseline_evaluation']['mean_recall_at_5']
                total_improvement = final_recall - baseline_recall
                
                results['final_summary'] = {
                    'baseline_recall_at_5': baseline_recall,
                    'optimized_recall_at_5': final_recall,
                    'total_improvement': total_improvement,
                    'improvement_percentage': (total_improvement / baseline_recall * 100) if baseline_recall > 0 else 0.0
                }
            
            logger.info("Complete evaluation and optimization workflow completed successfully")
            
        except Exception as e:
            logger.error(f"Evaluation and optimization workflow failed: {str(e)}")
            results['error'] = str(e)
        
        # Save complete results
        complete_results_file = self.results_dir / f"complete_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(complete_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Complete results saved to: {complete_results_file}")
        
        return results


def main() -> None:
    """
    Main function to run training data evaluation and optimization.
    """
    # This would be called from a script or notebook
    # For now, it's just a placeholder for the main execution logic
    pass


if __name__ == "__main__":
    main()