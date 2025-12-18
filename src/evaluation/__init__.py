"""
Evaluation module for SHL assessment recommendation system.

This module provides comprehensive evaluation capabilities including:
- EvaluationEngine for measuring recommendation system performance
- MetricsCalculator for computing various evaluation metrics
- PerformanceTracker for tracking optimization iterations
"""

from .evaluation_engine import (
    EvaluationEngine, 
    EvaluationQuery, 
    EvaluationResult, 
    EvaluationSummary
)
from .metrics_calculator import MetricsCalculator, MetricResult
from .performance_tracker import (
    PerformanceTracker, 
    OptimizationIteration, 
    PerformanceImprovement
)

__all__ = [
    'EvaluationEngine',
    'EvaluationQuery',
    'EvaluationResult', 
    'EvaluationSummary',
    'MetricsCalculator',
    'MetricResult',
    'PerformanceTracker',
    'OptimizationIteration',
    'PerformanceImprovement'
]