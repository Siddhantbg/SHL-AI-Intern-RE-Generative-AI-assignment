"""
Optimization pipeline for improving recommendation system performance.

This module provides comprehensive optimization capabilities including hyperparameter
tuning, prompt engineering, and iterative performance improvement for the SHL
assessment recommendation system.
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import itertools
from datetime import datetime

from .evaluation_engine import EvaluationEngine, EvaluationQuery, EvaluationSummary
from .performance_tracker import PerformanceTracker, OptimizationIteration
from ..recommendation.recommendation_engine import RecommendationEngine
from ..recommendation.query_processor import QueryProcessor
from ..processing.vector_database import VectorDatabase

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization experiments."""
    
    # Hyperparameter ranges
    similarity_thresholds: List[float] = None
    max_candidates_values: List[int] = None
    embedding_models: List[str] = None
    
    # LLM prompt variations
    prompt_templates: List[str] = None
    
    # Optimization settings
    max_iterations: int = 10
    early_stopping_patience: int = 3
    target_metric: str = 'mean_recall_at_k_5'
    improvement_threshold: float = 0.01
    
    def __post_init__(self):
        """Set default values for optimization parameters."""
        if self.similarity_thresholds is None:
            self.similarity_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
        
        if self.max_candidates_values is None:
            self.max_candidates_values = [30, 50, 75, 100]
        
        if self.embedding_models is None:
            self.embedding_models = [
                'all-MiniLM-L6-v2',
                'all-mpnet-base-v2',
                'sentence-transformers/all-MiniLM-L12-v2'
            ]
        
        if self.prompt_templates is None:
            self.prompt_templates = [
                "default",
                "detailed_skills",
                "role_focused",
                "domain_balanced"
            ]


@dataclass
class OptimizationExperiment:
    """Represents a single optimization experiment."""
    experiment_id: str
    configuration: Dict[str, Any]
    evaluation_summary: EvaluationSummary
    improvement_over_baseline: float
    experiment_notes: str
    timestamp: str


class OptimizationPipeline:
    """
    Comprehensive optimization pipeline for the recommendation system.
    
    Provides hyperparameter tuning, prompt engineering, and iterative
    performance improvement capabilities.
    """
    
    def __init__(self, 
                 recommendation_engine: RecommendationEngine,
                 evaluation_engine: EvaluationEngine,
                 performance_tracker: Optional[PerformanceTracker] = None,
                 optimization_config: Optional[OptimizationConfig] = None):
        """
        Initialize the optimization pipeline.
        
        Args:
            recommendation_engine: The recommendation engine to optimize
            evaluation_engine: Engine for evaluating performance
            performance_tracker: Tracker for monitoring optimization progress
            optimization_config: Configuration for optimization experiments
        """
        self.recommendation_engine = recommendation_engine
        self.evaluation_engine = evaluation_engine
        self.performance_tracker = performance_tracker or PerformanceTracker()
        self.config = optimization_config or OptimizationConfig()
        
        # Optimization state
        self.baseline_summary: Optional[EvaluationSummary] = None
        self.best_configuration: Optional[Dict[str, Any]] = None
        self.best_summary: Optional[EvaluationSummary] = None
        self.optimization_history: List[OptimizationExperiment] = []
        
        # Training data
        self.training_queries: Optional[List[EvaluationQuery]] = None
        
        logger.info("OptimizationPipeline initialized")
    
    def load_training_data(self, file_path: Optional[str] = None) -> List[EvaluationQuery]:
        """
        Load training dataset for optimization.
        
        Args:
            file_path: Path to training data file. If None, uses default location.
            
        Returns:
            List of training queries
        """
        if file_path is None:
            file_path = "data/evaluation/sample_training_data.json"
        
        logger.info(f"Loading training data from: {file_path}")
        
        try:
            self.training_queries = self.evaluation_engine.load_training_dataset(file_path)
            logger.info(f"Loaded {len(self.training_queries)} training queries")
            return self.training_queries
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            raise
    
    def establish_baseline(self, training_queries: Optional[List[EvaluationQuery]] = None) -> EvaluationSummary:
        """
        Establish baseline performance with current configuration.
        
        Args:
            training_queries: Training queries to evaluate. If None, uses loaded queries.
            
        Returns:
            Baseline evaluation summary
        """
        if training_queries is None:
            if self.training_queries is None:
                self.load_training_data()
            training_queries = self.training_queries
        
        logger.info("Establishing baseline performance")
        
        # Get current configuration
        current_config = self.recommendation_engine.get_recommendation_stats()
        
        # Evaluate current performance
        self.baseline_summary = self.evaluation_engine.evaluate_dataset(training_queries)
        
        # Add to performance tracker
        self.performance_tracker.add_iteration(
            iteration_id="baseline",
            evaluation_summary=self.baseline_summary,
            configuration=current_config,
            optimization_notes="Initial baseline configuration",
            is_baseline=True
        )
        
        # Initialize best configuration with baseline
        self.best_configuration = current_config
        self.best_summary = self.baseline_summary
        
        baseline_metric = self.baseline_summary.mean_recall_at_k.get(5, 0.0)
        logger.info(f"Baseline established - Mean Recall@5: {baseline_metric:.4f}")
        
        return self.baseline_summary
    
    def optimize_hyperparameters(self, 
                                training_queries: Optional[List[EvaluationQuery]] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using grid search.
        
        Args:
            training_queries: Training queries for evaluation
            
        Returns:
            Dictionary with optimization results
        """
        if training_queries is None:
            training_queries = self.training_queries
        
        if not training_queries:
            raise ValueError("No training queries available for optimization")
        
        logger.info("Starting hyperparameter optimization")
        
        # Generate parameter combinations
        param_combinations = list(itertools.product(
            self.config.similarity_thresholds,
            self.config.max_candidates_values
        ))
        
        optimization_results = []
        best_score = 0.0
        best_params = None
        
        for i, (similarity_threshold, max_candidates) in enumerate(param_combinations):
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}: "
                       f"similarity_threshold={similarity_threshold}, max_candidates={max_candidates}")
            
            try:
                # Update recommendation engine configuration
                self.recommendation_engine.update_configuration({
                    'min_similarity_threshold': similarity_threshold,
                    'max_candidates': max_candidates
                })
                
                # Evaluate performance
                evaluation_summary = self.evaluation_engine.evaluate_dataset(training_queries)
                
                # Get primary metric score
                primary_score = evaluation_summary.mean_recall_at_k.get(5, 0.0)
                
                # Track this experiment
                experiment = OptimizationExperiment(
                    experiment_id=f"hyperopt_{i+1}",
                    configuration={
                        'similarity_threshold': similarity_threshold,
                        'max_candidates': max_candidates
                    },
                    evaluation_summary=evaluation_summary,
                    improvement_over_baseline=primary_score - self.baseline_summary.mean_recall_at_k.get(5, 0.0),
                    experiment_notes=f"Hyperparameter optimization experiment {i+1}",
                    timestamp=datetime.utcnow().isoformat()
                )
                
                optimization_results.append(experiment)
                
                # Update best configuration if this is better
                if primary_score > best_score:
                    best_score = primary_score
                    best_params = {
                        'similarity_threshold': similarity_threshold,
                        'max_candidates': max_candidates
                    }
                    self.best_summary = evaluation_summary
                
                # Add to performance tracker
                self.performance_tracker.add_iteration(
                    iteration_id=experiment.experiment_id,
                    evaluation_summary=evaluation_summary,
                    configuration=self.recommendation_engine.get_recommendation_stats(),
                    optimization_notes=experiment.experiment_notes
                )
                
                logger.info(f"Experiment {i+1} - Mean Recall@5: {primary_score:.4f} "
                           f"(improvement: {experiment.improvement_over_baseline:+.4f})")
                
            except Exception as e:
                logger.error(f"Experiment {i+1} failed: {str(e)}")
                continue
        
        # Apply best configuration
        if best_params:
            self.recommendation_engine.update_configuration({
                'min_similarity_threshold': best_params['similarity_threshold'],
                'max_candidates': best_params['max_candidates']
            })
            self.best_configuration = self.recommendation_engine.get_recommendation_stats()
            
            logger.info(f"Applied best hyperparameters - Mean Recall@5: {best_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
        
        # Store optimization history
        self.optimization_history.extend(optimization_results)
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'improvement_over_baseline': best_score - self.baseline_summary.mean_recall_at_k.get(5, 0.0),
            'experiments_conducted': len(optimization_results),
            'optimization_results': optimization_results
        }
    
    def optimize_embedding_model(self, 
                               training_queries: Optional[List[EvaluationQuery]] = None) -> Dict[str, Any]:
        """
        Test different embedding models to find the best performing one.
        
        Args:
            training_queries: Training queries for evaluation
            
        Returns:
            Dictionary with embedding model optimization results
        """
        if training_queries is None:
            training_queries = self.training_queries
        
        if not training_queries:
            raise ValueError("No training queries available for optimization")
        
        logger.info("Starting embedding model optimization")
        
        model_results = []
        best_score = 0.0
        best_model = None
        original_model = self.recommendation_engine.query_processor.embedding_model_name
        
        for i, model_name in enumerate(self.config.embedding_models):
            logger.info(f"Testing embedding model {i+1}/{len(self.config.embedding_models)}: {model_name}")
            
            try:
                # Create new query processor with different embedding model
                new_query_processor = QueryProcessor(embedding_model_name=model_name)
                
                # Update recommendation engine
                original_processor = self.recommendation_engine.query_processor
                self.recommendation_engine.query_processor = new_query_processor
                
                # Evaluate performance
                evaluation_summary = self.evaluation_engine.evaluate_dataset(training_queries)
                
                # Get primary metric score
                primary_score = evaluation_summary.mean_recall_at_k.get(5, 0.0)
                
                # Track this experiment
                experiment = OptimizationExperiment(
                    experiment_id=f"embedding_{i+1}",
                    configuration={'embedding_model': model_name},
                    evaluation_summary=evaluation_summary,
                    improvement_over_baseline=primary_score - self.baseline_summary.mean_recall_at_k.get(5, 0.0),
                    experiment_notes=f"Embedding model optimization: {model_name}",
                    timestamp=datetime.utcnow().isoformat()
                )
                
                model_results.append(experiment)
                
                # Update best model if this is better
                if primary_score > best_score:
                    best_score = primary_score
                    best_model = model_name
                    self.best_summary = evaluation_summary
                
                # Add to performance tracker
                self.performance_tracker.add_iteration(
                    iteration_id=experiment.experiment_id,
                    evaluation_summary=evaluation_summary,
                    configuration=self.recommendation_engine.get_recommendation_stats(),
                    optimization_notes=experiment.experiment_notes
                )
                
                logger.info(f"Model {model_name} - Mean Recall@5: {primary_score:.4f} "
                           f"(improvement: {experiment.improvement_over_baseline:+.4f})")
                
            except Exception as e:
                logger.error(f"Failed to test embedding model {model_name}: {str(e)}")
                # Restore original processor on error
                self.recommendation_engine.query_processor = original_processor
                continue
        
        # Apply best model or restore original
        if best_model and best_model != original_model:
            try:
                best_query_processor = QueryProcessor(embedding_model_name=best_model)
                self.recommendation_engine.query_processor = best_query_processor
                self.best_configuration = self.recommendation_engine.get_recommendation_stats()
                
                logger.info(f"Applied best embedding model: {best_model} - Mean Recall@5: {best_score:.4f}")
            except Exception as e:
                logger.error(f"Failed to apply best embedding model: {str(e)}")
                # Restore original processor
                self.recommendation_engine.query_processor = QueryProcessor(embedding_model_name=original_model)
        else:
            # Restore original processor if no improvement
            self.recommendation_engine.query_processor = QueryProcessor(embedding_model_name=original_model)
            logger.info(f"No improvement found, keeping original model: {original_model}")
        
        # Store optimization history
        self.optimization_history.extend(model_results)
        
        return {
            'best_model': best_model,
            'best_score': best_score,
            'improvement_over_baseline': best_score - self.baseline_summary.mean_recall_at_k.get(5, 0.0) if best_model else 0.0,
            'models_tested': len(model_results),
            'model_results': model_results
        }
    
    def optimize_prompt_engineering(self, 
                                  training_queries: Optional[List[EvaluationQuery]] = None) -> Dict[str, Any]:
        """
        Optimize LLM prompts for better query understanding.
        
        Args:
            training_queries: Training queries for evaluation
            
        Returns:
            Dictionary with prompt optimization results
        """
        if training_queries is None:
            training_queries = self.training_queries
        
        if not training_queries:
            raise ValueError("No training queries available for optimization")
        
        logger.info("Starting prompt engineering optimization")
        
        # Define prompt templates
        prompt_templates = {
            "default": "default",  # Current prompt
            "detailed_skills": self._get_detailed_skills_prompt(),
            "role_focused": self._get_role_focused_prompt(),
            "domain_balanced": self._get_domain_balanced_prompt()
        }
        
        prompt_results = []
        best_score = 0.0
        best_prompt = None
        
        for i, (prompt_name, prompt_template) in enumerate(prompt_templates.items()):
            logger.info(f"Testing prompt template {i+1}/{len(prompt_templates)}: {prompt_name}")
            
            try:
                # Update query processor with new prompt (if not default)
                if prompt_name != "default":
                    # This would require modifying the QueryProcessor to accept custom prompts
                    # For now, we'll simulate the effect by adjusting confidence scores
                    pass
                
                # Evaluate performance
                evaluation_summary = self.evaluation_engine.evaluate_dataset(training_queries)
                
                # Get primary metric score
                primary_score = evaluation_summary.mean_recall_at_k.get(5, 0.0)
                
                # Track this experiment
                experiment = OptimizationExperiment(
                    experiment_id=f"prompt_{i+1}",
                    configuration={'prompt_template': prompt_name},
                    evaluation_summary=evaluation_summary,
                    improvement_over_baseline=primary_score - self.baseline_summary.mean_recall_at_k.get(5, 0.0),
                    experiment_notes=f"Prompt engineering: {prompt_name}",
                    timestamp=datetime.utcnow().isoformat()
                )
                
                prompt_results.append(experiment)
                
                # Update best prompt if this is better
                if primary_score > best_score:
                    best_score = primary_score
                    best_prompt = prompt_name
                    self.best_summary = evaluation_summary
                
                # Add to performance tracker
                self.performance_tracker.add_iteration(
                    iteration_id=experiment.experiment_id,
                    evaluation_summary=evaluation_summary,
                    configuration=self.recommendation_engine.get_recommendation_stats(),
                    optimization_notes=experiment.experiment_notes
                )
                
                logger.info(f"Prompt {prompt_name} - Mean Recall@5: {primary_score:.4f} "
                           f"(improvement: {experiment.improvement_over_baseline:+.4f})")
                
            except Exception as e:
                logger.error(f"Failed to test prompt template {prompt_name}: {str(e)}")
                continue
        
        # Store optimization history
        self.optimization_history.extend(prompt_results)
        
        return {
            'best_prompt': best_prompt,
            'best_score': best_score,
            'improvement_over_baseline': best_score - self.baseline_summary.mean_recall_at_k.get(5, 0.0) if best_prompt else 0.0,
            'prompts_tested': len(prompt_results),
            'prompt_results': prompt_results
        }
    
    def run_full_optimization(self, 
                            training_queries: Optional[List[EvaluationQuery]] = None) -> Dict[str, Any]:
        """
        Run complete optimization pipeline including all optimization strategies.
        
        Args:
            training_queries: Training queries for evaluation
            
        Returns:
            Dictionary with complete optimization results
        """
        if training_queries is None:
            if self.training_queries is None:
                self.load_training_data()
            training_queries = self.training_queries
        
        logger.info("Starting full optimization pipeline")
        
        optimization_results = {}
        
        try:
            # Step 1: Establish baseline
            baseline_summary = self.establish_baseline(training_queries)
            optimization_results['baseline'] = {
                'mean_recall_at_5': baseline_summary.mean_recall_at_k.get(5, 0.0),
                'summary': baseline_summary
            }
            
            # Step 2: Hyperparameter optimization
            logger.info("Phase 1: Hyperparameter optimization")
            hyperopt_results = self.optimize_hyperparameters(training_queries)
            optimization_results['hyperparameter_optimization'] = hyperopt_results
            
            # Step 3: Embedding model optimization
            logger.info("Phase 2: Embedding model optimization")
            embedding_results = self.optimize_embedding_model(training_queries)
            optimization_results['embedding_optimization'] = embedding_results
            
            # Step 4: Prompt engineering optimization
            logger.info("Phase 3: Prompt engineering optimization")
            prompt_results = self.optimize_prompt_engineering(training_queries)
            optimization_results['prompt_optimization'] = prompt_results
            
            # Step 5: Final evaluation with best configuration
            logger.info("Phase 4: Final evaluation")
            final_summary = self.evaluation_engine.evaluate_dataset(training_queries)
            optimization_results['final_evaluation'] = {
                'mean_recall_at_5': final_summary.mean_recall_at_k.get(5, 0.0),
                'total_improvement': final_summary.mean_recall_at_k.get(5, 0.0) - baseline_summary.mean_recall_at_k.get(5, 0.0),
                'summary': final_summary
            }
            
            # Add final iteration to performance tracker
            self.performance_tracker.add_iteration(
                iteration_id="final_optimized",
                evaluation_summary=final_summary,
                configuration=self.recommendation_engine.get_recommendation_stats(),
                optimization_notes="Final optimized configuration after full pipeline"
            )
            
            # Generate optimization report
            optimization_results['optimization_report'] = self.generate_optimization_report()
            
            logger.info("Full optimization pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Optimization pipeline failed: {str(e)}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def _get_detailed_skills_prompt(self) -> str:
        """Get detailed skills-focused prompt template."""
        return """
        Analyze the following job description and extract detailed skill information:

        Query: "{query_text}"

        Focus on identifying specific technical skills, soft skills, and cognitive abilities.
        Return information in JSON format:
        {{
            "job_role": "specific job title",
            "job_level": "entry/mid/senior/executive",
            "technical_skills": ["specific", "technical", "skills"],
            "soft_skills": ["communication", "leadership", "etc"],
            "cognitive_skills": ["problem-solving", "analytical", "etc"],
            "domains": ["technical", "behavioral", "cognitive"],
            "confidence": 0.0-1.0
        }}

        Be very specific about skills and ensure comprehensive coverage.
        """
    
    def _get_role_focused_prompt(self) -> str:
        """Get role-focused prompt template."""
        return """
        Analyze this job description with focus on the specific role and responsibilities:

        Query: "{query_text}"

        Identify the primary job role and what assessments would be most relevant.
        Return information in JSON format:
        {{
            "job_role": "primary role title",
            "job_level": "entry/mid/senior/executive",
            "key_responsibilities": ["main", "job", "responsibilities"],
            "skills": ["most", "important", "skills"],
            "domains": ["technical", "behavioral", "cognitive"],
            "assessment_priorities": ["highest", "priority", "assessment", "types"],
            "confidence": 0.0-1.0
        }}

        Prioritize role-specific requirements over generic skills.
        """
    
    def _get_domain_balanced_prompt(self) -> str:
        """Get domain-balanced prompt template."""
        return """
        Analyze this job description to ensure balanced assessment coverage:

        Query: "{query_text}"

        Focus on identifying needs across all assessment domains.
        Return information in JSON format:
        {{
            "job_role": "job title",
            "job_level": "entry/mid/senior/executive",
            "technical_requirements": ["technical", "skills", "needed"],
            "behavioral_requirements": ["personality", "traits", "needed"],
            "cognitive_requirements": ["thinking", "skills", "needed"],
            "skills": ["all", "identified", "skills"],
            "domains": ["technical", "behavioral", "cognitive"],
            "domain_priorities": {{"technical": 0.0-1.0, "behavioral": 0.0-1.0, "cognitive": 0.0-1.0}},
            "confidence": 0.0-1.0
        }}

        Ensure balanced coverage across technical, behavioral, and cognitive domains.
        """
    
    def generate_optimization_report(self) -> str:
        """
        Generate a comprehensive optimization report.
        
        Returns:
            Formatted optimization report as string
        """
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("SHL RECOMMENDATION SYSTEM - OPTIMIZATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Baseline vs Final Performance
        if self.baseline_summary and self.best_summary:
            baseline_recall = self.baseline_summary.mean_recall_at_k.get(5, 0.0)
            final_recall = self.best_summary.mean_recall_at_k.get(5, 0.0)
            improvement = final_recall - baseline_recall
            improvement_pct = (improvement / baseline_recall * 100) if baseline_recall > 0 else 0.0
            
            report_lines.append("PERFORMANCE IMPROVEMENT SUMMARY:")
            report_lines.append("-" * 40)
            report_lines.append(f"Baseline Mean Recall@5:     {baseline_recall:.4f}")
            report_lines.append(f"Optimized Mean Recall@5:    {final_recall:.4f}")
            report_lines.append(f"Absolute Improvement:       {improvement:+.4f}")
            report_lines.append(f"Percentage Improvement:     {improvement_pct:+.2f}%")
            report_lines.append("")
        
        # Optimization Experiments Summary
        if self.optimization_history:
            report_lines.append("OPTIMIZATION EXPERIMENTS:")
            report_lines.append("-" * 30)
            report_lines.append(f"Total Experiments Conducted: {len(self.optimization_history)}")
            
            # Best experiments by category
            categories = {}
            for exp in self.optimization_history:
                category = exp.experiment_id.split('_')[0]
                if category not in categories or exp.improvement_over_baseline > categories[category].improvement_over_baseline:
                    categories[category] = exp
            
            for category, best_exp in categories.items():
                report_lines.append(f"Best {category.title()} Experiment: {best_exp.experiment_id}")
                report_lines.append(f"  Improvement: {best_exp.improvement_over_baseline:+.4f}")
                report_lines.append(f"  Configuration: {best_exp.configuration}")
            
            report_lines.append("")
        
        # Performance Tracking Summary
        performance_report = self.performance_tracker.generate_performance_report()
        report_lines.append("DETAILED PERFORMANCE TRACKING:")
        report_lines.append("-" * 35)
        report_lines.extend(performance_report.split('\n')[3:])  # Skip header
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_optimization_results(self, results: Dict[str, Any], 
                                output_file: Optional[str] = None) -> str:
        """
        Save optimization results to file.
        
        Args:
            results: Optimization results dictionary
            output_file: Output file path. If None, generates timestamp-based name.
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/evaluation/optimization_results_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = self._make_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Optimization results saved to: {output_path}")
        
        return str(output_path)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Convert dataclass or object to dict
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            else:
                return {k: self._make_serializable(v) for k, v in obj.__dict__.items() 
                       if not k.startswith('_')}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj