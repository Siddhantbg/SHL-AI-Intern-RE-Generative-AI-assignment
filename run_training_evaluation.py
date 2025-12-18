#!/usr/bin/env python3
"""
Run training data evaluation and optimization for the SHL recommendation system.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run training data evaluation and optimization."""
    logger.info("Starting SHL Recommendation System Training Evaluation")
    
    try:
        # Import required modules
        from src.evaluation.training_evaluation import TrainingEvaluator
        from src.evaluation.optimization_pipeline import OptimizationConfig
        from src.recommendation.recommendation_engine import RecommendationEngine
        from src.recommendation.query_processor import QueryProcessor
        from src.recommendation.balanced_ranker import BalancedRanker
        from src.processing.vector_database import VectorDatabase
        
        # Initialize recommendation system components
        logger.info("Initializing recommendation system components...")
        
        vector_db = VectorDatabase()
        query_processor = QueryProcessor()
        balanced_ranker = BalancedRanker()
        
        recommendation_engine = RecommendationEngine(
            vector_database=vector_db,
            query_processor=query_processor,
            balanced_ranker=balanced_ranker
        )
        
        # Initialize training evaluator
        evaluator = TrainingEvaluator(
            recommendation_engine=recommendation_engine,
            training_data_path="data/evaluation/sample_training_data.json"
        )
        
        # Load training data
        logger.info("Loading training data...")
        training_queries = evaluator.load_training_data()
        logger.info(f"Loaded {len(training_queries)} training queries")
        
        # Run baseline evaluation
        logger.info("Running baseline evaluation...")
        baseline_summary = evaluator.run_baseline_evaluation()
        
        baseline_recall = baseline_summary.mean_recall_at_k.get(5, 0.0)
        logger.info(f"Baseline Mean Recall@5: {baseline_recall:.4f}")
        
        # Run individual query analysis
        logger.info("Analyzing individual queries...")
        individual_results = evaluator.evaluate_individual_queries()
        
        if individual_results:
            avg_recall = sum(r['recall_at_5'] for r in individual_results) / len(individual_results)
            logger.info(f"Average individual query Recall@5: {avg_recall:.4f}")
            
            # Show best and worst queries
            best_query = max(individual_results, key=lambda x: x['recall_at_5'])
            worst_query = min(individual_results, key=lambda x: x['recall_at_5'])
            
            logger.info(f"Best query: {best_query['query_id']} (Recall@5: {best_query['recall_at_5']:.4f})")
            logger.info(f"Worst query: {worst_query['query_id']} (Recall@5: {worst_query['recall_at_5']:.4f})")
        
        # Run hyperparameter tuning
        logger.info("Running hyperparameter optimization...")
        hyperopt_results = evaluator.run_hyperparameter_tuning()
        
        best_params = hyperopt_results.get('best_parameters', {})
        improvement = hyperopt_results.get('improvement_over_baseline', 0.0)
        
        logger.info(f"Hyperparameter optimization results:")
        logger.info(f"  Best parameters: {best_params}")
        logger.info(f"  Improvement: {improvement:+.4f}")
        
        # Generate performance analysis
        logger.info("Generating performance analysis...")
        performance_analysis = evaluator.generate_performance_analysis()
        
        if 'optimization_trend' in performance_analysis:
            trend = performance_analysis['optimization_trend']
            logger.info(f"Optimization trend: {trend['trend']}")
            logger.info(f"Best value achieved: {trend['best_value']:.4f}")
        
        # Summary
        logger.info("=" * 60)
        logger.info("TRAINING EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Training queries evaluated: {len(training_queries)}")
        logger.info(f"Baseline Mean Recall@5: {baseline_recall:.4f}")
        
        if improvement > 0:
            final_recall = baseline_recall + improvement
            improvement_pct = (improvement / baseline_recall * 100) if baseline_recall > 0 else 0
            logger.info(f"Optimized Mean Recall@5: {final_recall:.4f}")
            logger.info(f"Total improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
            logger.info("✓ Optimization successful!")
        else:
            logger.info("✗ No improvement achieved through optimization")
        
        logger.info("Check 'data/evaluation/results/' for detailed results and reports")
        logger.info("Training evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Training evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()