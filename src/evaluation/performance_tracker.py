"""
Performance tracker for optimization iterations.

This module provides functionality to track performance improvements across
optimization iterations, log performance metrics, and analyze optimization trends.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import numpy as np
from .evaluation_engine import EvaluationSummary

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


@dataclass
class OptimizationIteration:
    """Represents a single optimization iteration with performance metrics."""
    iteration_id: str
    timestamp: str
    configuration: Dict[str, Any]
    evaluation_summary: EvaluationSummary
    optimization_notes: str = ""
    baseline_comparison: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceImprovement:
    """Represents performance improvement between iterations."""
    from_iteration: str
    to_iteration: str
    metric_improvements: Dict[str, float]
    significant_improvements: List[str]
    degradations: List[str]
    overall_improvement: float


class PerformanceTracker:
    """
    Tracks performance across optimization iterations and provides analysis.
    
    Maintains a history of evaluation results, tracks improvements, and provides
    visualization and analysis tools for optimization progress.
    """
    
    def __init__(self, tracking_dir: str = "data/evaluation/tracking"):
        """
        Initialize the performance tracker.
        
        Args:
            tracking_dir: Directory to store tracking data
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking data
        self.iterations: List[OptimizationIteration] = []
        self.baseline_iteration: Optional[OptimizationIteration] = None
        
        # Configuration
        self.primary_metric = 'mean_recall_at_k_5'  # Primary optimization target
        self.significance_threshold = 0.01  # Minimum improvement to be considered significant
        
        # Load existing tracking data if available
        self._load_tracking_data()
        
        logger.info(f"PerformanceTracker initialized with {len(self.iterations)} iterations")
    
    def add_iteration(self, iteration_id: str, 
                     evaluation_summary: EvaluationSummary,
                     configuration: Dict[str, Any],
                     optimization_notes: str = "",
                     is_baseline: bool = False) -> OptimizationIteration:
        """
        Add a new optimization iteration to the tracking history.
        
        Args:
            iteration_id: Unique identifier for this iteration
            evaluation_summary: Evaluation results for this iteration
            configuration: System configuration used for this iteration
            optimization_notes: Notes about what was optimized in this iteration
            is_baseline: Whether this iteration should be used as baseline
            
        Returns:
            OptimizationIteration object
        """
        # Create iteration object
        iteration = OptimizationIteration(
            iteration_id=iteration_id,
            timestamp=datetime.utcnow().isoformat(),
            configuration=configuration,
            evaluation_summary=evaluation_summary,
            optimization_notes=optimization_notes
        )
        
        # Calculate baseline comparison if we have a baseline
        if self.baseline_iteration and not is_baseline:
            iteration.baseline_comparison = self._calculate_baseline_comparison(
                iteration, self.baseline_iteration
            )
        
        # Add to iterations list
        self.iterations.append(iteration)
        
        # Set as baseline if requested
        if is_baseline or self.baseline_iteration is None:
            self.baseline_iteration = iteration
            logger.info(f"Set iteration {iteration_id} as baseline")
        
        # Save tracking data
        self._save_tracking_data()
        
        logger.info(f"Added iteration {iteration_id} to performance tracking")
        
        return iteration
    
    def get_performance_improvement(self, from_iteration_id: str, 
                                  to_iteration_id: str) -> PerformanceImprovement:
        """
        Calculate performance improvement between two iterations.
        
        Args:
            from_iteration_id: Starting iteration ID
            to_iteration_id: Ending iteration ID
            
        Returns:
            PerformanceImprovement object
            
        Raises:
            ValueError: If iteration IDs are not found
        """
        from_iteration = self._get_iteration_by_id(from_iteration_id)
        to_iteration = self._get_iteration_by_id(to_iteration_id)
        
        if not from_iteration or not to_iteration:
            raise ValueError(f"Iteration not found: {from_iteration_id} or {to_iteration_id}")
        
        # Calculate metric improvements
        metric_improvements = {}
        significant_improvements = []
        degradations = []
        
        # Compare recall metrics
        for k, from_recall in from_iteration.evaluation_summary.mean_recall_at_k.items():
            to_recall = to_iteration.evaluation_summary.mean_recall_at_k.get(k, 0.0)
            improvement = to_recall - from_recall
            
            metric_key = f'mean_recall_at_k_{k}'
            metric_improvements[metric_key] = improvement
            
            if improvement >= self.significance_threshold:
                significant_improvements.append(metric_key)
            elif improvement <= -self.significance_threshold:
                degradations.append(metric_key)
        
        # Compare precision metrics
        for k, from_precision in from_iteration.evaluation_summary.mean_precision_at_k.items():
            to_precision = to_iteration.evaluation_summary.mean_precision_at_k.get(k, 0.0)
            improvement = to_precision - from_precision
            
            metric_key = f'mean_precision_at_k_{k}'
            metric_improvements[metric_key] = improvement
            
            if improvement >= self.significance_threshold:
                significant_improvements.append(metric_key)
            elif improvement <= -self.significance_threshold:
                degradations.append(metric_key)
        
        # Compare processing time (improvement = reduction in time)
        time_improvement = from_iteration.evaluation_summary.mean_processing_time - to_iteration.evaluation_summary.mean_processing_time
        metric_improvements['processing_time'] = time_improvement
        
        if time_improvement >= 0.01:  # 10ms improvement
            significant_improvements.append('processing_time')
        elif time_improvement <= -0.01:
            degradations.append('processing_time')
        
        # Calculate overall improvement based on primary metric
        overall_improvement = metric_improvements.get(self.primary_metric, 0.0)
        
        return PerformanceImprovement(
            from_iteration=from_iteration_id,
            to_iteration=to_iteration_id,
            metric_improvements=metric_improvements,
            significant_improvements=significant_improvements,
            degradations=degradations,
            overall_improvement=overall_improvement
        )
    
    def get_best_iteration(self, metric: str = None) -> Optional[OptimizationIteration]:
        """
        Get the best performing iteration based on a specific metric.
        
        Args:
            metric: Metric to optimize for. If None, uses primary metric.
            
        Returns:
            Best performing OptimizationIteration or None if no iterations
        """
        if not self.iterations:
            return None
        
        if metric is None:
            metric = self.primary_metric
        
        best_iteration = None
        best_value = float('-inf')
        
        for iteration in self.iterations:
            value = self._extract_metric_value(iteration, metric)
            
            if value is not None and value > best_value:
                best_value = value
                best_iteration = iteration
        
        return best_iteration
    
    def get_optimization_trend(self, metric: str = None, 
                             window_size: int = 3) -> Dict[str, Any]:
        """
        Analyze optimization trend for a specific metric.
        
        Args:
            metric: Metric to analyze. If None, uses primary metric.
            window_size: Window size for moving average calculation
            
        Returns:
            Dictionary with trend analysis
        """
        if metric is None:
            metric = self.primary_metric
        
        if len(self.iterations) < 2:
            return {
                'metric': metric,
                'trend': 'insufficient_data',
                'iterations': len(self.iterations)
            }
        
        # Extract metric values over time
        values = []
        timestamps = []
        iteration_ids = []
        
        for iteration in self.iterations:
            value = self._extract_metric_value(iteration, metric)
            if value is not None:
                values.append(value)
                timestamps.append(iteration.timestamp)
                iteration_ids.append(iteration.iteration_id)
        
        if len(values) < 2:
            return {
                'metric': metric,
                'trend': 'insufficient_data',
                'iterations': len(values)
            }
        
        # Calculate trend statistics
        values_array = np.array(values)
        
        # Linear trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values_array, 1)
        
        # Moving average
        if len(values) >= window_size:
            moving_avg = np.convolve(values_array, np.ones(window_size)/window_size, mode='valid')
        else:
            moving_avg = values_array
        
        # Recent trend (last 3 iterations)
        recent_values = values_array[-min(3, len(values)):]
        recent_trend = 'stable'
        if len(recent_values) >= 2:
            recent_slope = (recent_values[-1] - recent_values[0]) / (len(recent_values) - 1)
            if recent_slope > self.significance_threshold:
                recent_trend = 'improving'
            elif recent_slope < -self.significance_threshold:
                recent_trend = 'degrading'
        
        return {
            'metric': metric,
            'trend': 'improving' if slope > 0 else 'degrading' if slope < 0 else 'stable',
            'slope': slope,
            'recent_trend': recent_trend,
            'best_value': np.max(values_array),
            'worst_value': np.min(values_array),
            'current_value': values_array[-1],
            'improvement_from_first': values_array[-1] - values_array[0],
            'values': values,
            'iteration_ids': iteration_ids,
            'moving_average': moving_avg.tolist() if len(moving_avg) > 0 else []
        }
    
    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive performance tracking report.
        
        Returns:
            Formatted performance report as string
        """
        report_lines = []
        
        report_lines.append("=" * 70)
        report_lines.append("PERFORMANCE TRACKING REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Basic statistics
        report_lines.append(f"Total Optimization Iterations: {len(self.iterations)}")
        if self.baseline_iteration:
            report_lines.append(f"Baseline Iteration: {self.baseline_iteration.iteration_id}")
        report_lines.append(f"Primary Optimization Metric: {self.primary_metric}")
        report_lines.append("")
        
        if not self.iterations:
            report_lines.append("No optimization iterations recorded.")
            return "\n".join(report_lines)
        
        # Best performing iteration
        best_iteration = self.get_best_iteration()
        if best_iteration:
            best_value = self._extract_metric_value(best_iteration, self.primary_metric)
            report_lines.append(f"Best Performing Iteration: {best_iteration.iteration_id}")
            report_lines.append(f"Best {self.primary_metric}: {best_value:.4f}")
            report_lines.append("")
        
        # Optimization trend
        trend_analysis = self.get_optimization_trend()
        report_lines.append("OPTIMIZATION TREND ANALYSIS:")
        report_lines.append("-" * 30)
        report_lines.append(f"Overall Trend: {trend_analysis['trend']}")
        report_lines.append(f"Recent Trend: {trend_analysis['recent_trend']}")
        report_lines.append(f"Improvement from First: {trend_analysis['improvement_from_first']:.4f}")
        report_lines.append(f"Best Value Achieved: {trend_analysis['best_value']:.4f}")
        report_lines.append("")
        
        # Iteration history
        report_lines.append("ITERATION HISTORY:")
        report_lines.append("-" * 20)
        report_lines.append(f"{'Iteration':<15} {'Primary Metric':<15} {'Processing Time':<15} {'Notes'}")
        report_lines.append("-" * 70)
        
        for iteration in self.iterations:
            primary_value = self._extract_metric_value(iteration, self.primary_metric)
            processing_time = iteration.evaluation_summary.mean_processing_time
            notes = iteration.optimization_notes[:30] + "..." if len(iteration.optimization_notes) > 30 else iteration.optimization_notes
            
            # Handle None values
            primary_value_str = f"{primary_value:.4f}" if primary_value is not None else "N/A"
            processing_time_str = f"{processing_time:.3f}" if processing_time is not None else "N/A"
            
            report_lines.append(
                f"{iteration.iteration_id:<15} {primary_value_str:<15} {processing_time_str:<15} {notes}"
            )
        
        report_lines.append("")
        
        # Recent improvements
        if len(self.iterations) >= 2:
            recent_improvement = self.get_performance_improvement(
                self.iterations[-2].iteration_id,
                self.iterations[-1].iteration_id
            )
            
            report_lines.append("RECENT PERFORMANCE CHANGE:")
            report_lines.append("-" * 30)
            report_lines.append(f"From: {recent_improvement.from_iteration}")
            report_lines.append(f"To: {recent_improvement.to_iteration}")
            report_lines.append(f"Overall Improvement: {recent_improvement.overall_improvement:.4f}")
            
            if recent_improvement.significant_improvements:
                report_lines.append(f"Significant Improvements: {', '.join(recent_improvement.significant_improvements)}")
            
            if recent_improvement.degradations:
                report_lines.append(f"Degradations: {', '.join(recent_improvement.degradations)}")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def plot_optimization_progress(self, metrics: List[str] = None, 
                                 save_path: Optional[str] = None) -> str:
        """
        Create a plot showing optimization progress over iterations.
        
        Args:
            metrics: List of metrics to plot. If None, uses primary metric.
            save_path: Path to save plot. If None, saves to tracking directory.
            
        Returns:
            Path to saved plot file
            
        Raises:
            ImportError: If matplotlib is not available
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        if metrics is None:
            metrics = [self.primary_metric]
        
        if not self.iterations:
            raise ValueError("No iterations to plot")
        
        # Extract data for plotting
        iteration_numbers = list(range(1, len(self.iterations) + 1))
        iteration_labels = [iter.iteration_id for iter in self.iterations]
        
        # Create plot
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = []
            for iteration in self.iterations:
                value = self._extract_metric_value(iteration, metric)
                values.append(value if value is not None else 0.0)
            
            axes[i].plot(iteration_numbers, values, marker='o', linewidth=2, markersize=6)
            axes[i].set_title(f'Optimization Progress: {metric}')
            axes[i].set_xlabel('Iteration')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
            
            # Add iteration labels
            axes[i].set_xticks(iteration_numbers)
            axes[i].set_xticklabels(iteration_labels, rotation=45, ha='right')
            
            # Highlight best value
            if values:
                best_idx = np.argmax(values)
                axes[i].scatter([iteration_numbers[best_idx]], [values[best_idx]], 
                              color='red', s=100, zorder=5, label='Best')
                axes[i].legend()
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.tracking_dir / f"optimization_progress_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Optimization progress plot saved to: {save_path}")
        
        return str(save_path)
    
    def _calculate_baseline_comparison(self, current_iteration: OptimizationIteration,
                                     baseline_iteration: OptimizationIteration) -> Dict[str, Any]:
        """Calculate comparison metrics against baseline."""
        comparison = {}
        
        # Compare recall metrics
        recall_comparison = {}
        for k in baseline_iteration.evaluation_summary.mean_recall_at_k.keys():
            baseline_recall = baseline_iteration.evaluation_summary.mean_recall_at_k[k]
            current_recall = current_iteration.evaluation_summary.mean_recall_at_k.get(k, 0.0)
            
            improvement = current_recall - baseline_recall
            improvement_pct = (improvement / baseline_recall * 100) if baseline_recall > 0 else 0.0
            
            recall_comparison[f'recall_at_{k}'] = {
                'baseline': baseline_recall,
                'current': current_recall,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            }
        
        comparison['recall_metrics'] = recall_comparison
        
        # Compare processing time
        baseline_time = baseline_iteration.evaluation_summary.mean_processing_time
        current_time = current_iteration.evaluation_summary.mean_processing_time
        time_improvement = baseline_time - current_time
        time_improvement_pct = (time_improvement / baseline_time * 100) if baseline_time > 0 else 0.0
        
        comparison['processing_time'] = {
            'baseline': baseline_time,
            'current': current_time,
            'improvement': time_improvement,
            'improvement_pct': time_improvement_pct
        }
        
        return comparison
    
    def _extract_metric_value(self, iteration: OptimizationIteration, metric: str) -> Optional[float]:
        """Extract a specific metric value from an iteration."""
        if metric.startswith('mean_recall_at_k_'):
            k = int(metric.split('_')[-1])
            return iteration.evaluation_summary.mean_recall_at_k.get(k)
        elif metric.startswith('mean_precision_at_k_'):
            k = int(metric.split('_')[-1])
            return iteration.evaluation_summary.mean_precision_at_k.get(k)
        elif metric == 'processing_time':
            return iteration.evaluation_summary.mean_processing_time
        else:
            return None
    
    def _get_iteration_by_id(self, iteration_id: str) -> Optional[OptimizationIteration]:
        """Get iteration by ID."""
        for iteration in self.iterations:
            if iteration.iteration_id == iteration_id:
                return iteration
        return None
    
    def _save_tracking_data(self) -> None:
        """Save tracking data to file."""
        tracking_file = self.tracking_dir / "performance_tracking.json"
        
        data = {
            'baseline_iteration_id': self.baseline_iteration.iteration_id if self.baseline_iteration else None,
            'primary_metric': self.primary_metric,
            'significance_threshold': self.significance_threshold,
            'iterations': []
        }
        
        # Convert iterations to serializable format
        for iteration in self.iterations:
            iteration_data = asdict(iteration)
            # Remove non-serializable evaluation_summary.individual_results
            if 'evaluation_summary' in iteration_data:
                iteration_data['evaluation_summary'].pop('individual_results', None)
            data['iterations'].append(iteration_data)
        
        with open(tracking_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_tracking_data(self) -> None:
        """Load tracking data from file."""
        tracking_file = self.tracking_dir / "performance_tracking.json"
        
        if not tracking_file.exists():
            return
        
        try:
            with open(tracking_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load configuration
            self.primary_metric = data.get('primary_metric', self.primary_metric)
            self.significance_threshold = data.get('significance_threshold', self.significance_threshold)
            
            # Load iterations (simplified - without full evaluation_summary.individual_results)
            baseline_id = data.get('baseline_iteration_id')
            
            for iteration_data in data.get('iterations', []):
                # Reconstruct EvaluationSummary (without individual_results)
                eval_data = iteration_data['evaluation_summary']
                evaluation_summary = EvaluationSummary(
                    total_queries=eval_data['total_queries'],
                    mean_recall_at_k=eval_data['mean_recall_at_k'],
                    mean_precision_at_k=eval_data['mean_precision_at_k'],
                    mean_processing_time=eval_data['mean_processing_time'],
                    evaluation_timestamp=eval_data['evaluation_timestamp'],
                    configuration=eval_data['configuration'],
                    individual_results=[]  # Not loaded
                )
                
                iteration = OptimizationIteration(
                    iteration_id=iteration_data['iteration_id'],
                    timestamp=iteration_data['timestamp'],
                    configuration=iteration_data['configuration'],
                    evaluation_summary=evaluation_summary,
                    optimization_notes=iteration_data.get('optimization_notes', ''),
                    baseline_comparison=iteration_data.get('baseline_comparison')
                )
                
                self.iterations.append(iteration)
                
                # Set baseline
                if iteration.iteration_id == baseline_id:
                    self.baseline_iteration = iteration
            
            logger.info(f"Loaded {len(self.iterations)} iterations from tracking data")
            
        except Exception as e:
            logger.warning(f"Failed to load tracking data: {str(e)}")