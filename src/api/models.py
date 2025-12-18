"""
Pydantic models for API request and response validation.

This module defines the data models used for API endpoints including
request validation and response serialization.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime, timezone


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="API status (healthy/unhealthy)")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="API uptime in seconds")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Current timestamp")
    assessment_count: int = Field(..., description="Number of assessments loaded")
    environment: str = Field(..., description="Environment (development/production)")


class RecommendationRequest(BaseModel):
    """Request model for recommendation endpoint."""
    
    query: str = Field(
        ..., 
        min_length=10, 
        max_length=5000,
        description="Job description or query text for recommendations"
    )
    max_results: Optional[int] = Field(
        default=10, 
        ge=1, 
        le=10,
        description="Maximum number of recommendations to return (1-10)"
    )
    balance_domains: Optional[bool] = Field(
        default=True,
        description="Whether to balance recommendations across skill domains"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query text."""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        
        # Remove excessive whitespace
        cleaned = ' '.join(v.split())
        if len(cleaned) < 10:
            raise ValueError('Query must be at least 10 characters long')
        
        return cleaned


class AssessmentRecommendation(BaseModel):
    """Individual assessment recommendation."""
    
    assessment_name: str = Field(..., description="Name of the recommended assessment")
    url: str = Field(..., description="URL to the assessment on SHL catalog")
    test_type: str = Field(..., description="Assessment type (K for Knowledge & Skills, P for Personality & Behavior)")
    category: str = Field(..., description="Assessment category")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0.0-1.0)")
    explanation: str = Field(..., description="Human-readable explanation for the recommendation")
    skills_matched: List[str] = Field(default_factory=list, description="Skills from query that match this assessment")


class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoint."""
    
    query: str = Field(..., description="Original query text")
    recommendations: List[AssessmentRecommendation] = Field(
        ..., 
        description="List of recommended assessments"
    )
    total_results: int = Field(..., description="Number of recommendations returned")
    processing_time: float = Field(..., description="Processing time in seconds")
    query_info: Dict[str, Any] = Field(default_factory=dict, description="Information about query processing")
    balance_info: Dict[str, Any] = Field(default_factory=dict, description="Information about domain balancing")
    
    @field_validator('recommendations')
    @classmethod
    def validate_recommendations(cls, v: List[AssessmentRecommendation]) -> List[AssessmentRecommendation]:
        """Validate recommendations list."""
        if len(v) > 10:
            raise ValueError('Cannot return more than 10 recommendations')
        return v


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Error timestamp")


class ValidationErrorResponse(BaseModel):
    """Validation error response model."""
    
    error: str = Field(default="validation_error", description="Error type")
    message: str = Field(..., description="Validation error message")
    field_errors: List[Dict[str, Any]] = Field(..., description="Field-specific validation errors")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Error timestamp")