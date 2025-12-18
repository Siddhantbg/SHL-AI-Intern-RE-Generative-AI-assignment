"""
Data processing module for SHL assessment recommendation system.

This module provides classes for processing scraped assessment data and generating
vector embeddings for similarity search and recommendation.
"""

from .assessment_processor import AssessmentProcessor, ProcessedAssessment
from .embedding_generator import EmbeddingGenerator
from .vector_database import VectorDatabase, AssessmentVector

__all__ = [
    'AssessmentProcessor',
    'ProcessedAssessment', 
    'EmbeddingGenerator',
    'VectorDatabase',
    'AssessmentVector'
]