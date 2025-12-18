"""Main FastAPI application entry point."""

import logging
import time
import os
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from ..config import get_settings
from .models import (
    HealthResponse, 
    RecommendationRequest, 
    RecommendationResponse,
    ErrorResponse,
    ValidationErrorResponse,
    AssessmentRecommendation
)
from ..recommendation.recommendation_engine import RecommendationEngine
from ..processing.vector_database import VectorDatabase
from ..recommendation.query_processor import QueryProcessor
from ..recommendation.balanced_ranker import BalancedRanker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Global variables for recommendation system components
recommendation_engine: Optional[RecommendationEngine] = None
startup_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    await startup_event()
    yield
    # Shutdown (if needed)
    pass

app = FastAPI(
    title="SHL Assessment Recommendation System",
    description="An intelligent recommendation system for SHL assessments based on job descriptions and queries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
cors_origins = ["*"] if settings.debug else [
    "https://shl-ai-intern-re-generative-ai-assi.vercel.app",
    "https://shl-ai-intern-re-generative-ai-assignment.vercel.app",
    "https://localhost:3000",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed field information."""
    field_errors = []
    for error in exc.errors():
        field_errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ValidationErrorResponse(
            message="Request validation failed",
            field_errors=field_errors
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            message="An internal server error occurred",
            details={"path": str(request.url.path)} if settings.debug else None
        ).model_dump()
    )


async def startup_event() -> None:
    """Initialize the recommendation system on startup."""
    global recommendation_engine
    
    try:
        logger.info("Initializing SHL Assessment Recommendation System...")
        
        # Initialize vector database
        vector_db = VectorDatabase(
            embedding_dim=384,  # all-MiniLM-L6-v2 dimension
            index_type='flat',
            storage_dir=str(settings.data_dir / 'vector_db')
        )
        
        # Try to load existing database
        try:
            vector_db.load_database()
            logger.info("Loaded existing vector database")
        except FileNotFoundError:
            logger.warning("No existing vector database found. System will work with empty database.")
        except Exception as e:
            logger.error(f"Failed to load vector database: {str(e)}")
            logger.info("Starting with empty vector database")
        
        # Initialize query processor
        query_processor = QueryProcessor(embedding_model_name='all-MiniLM-L6-v2')
        
        # Initialize balanced ranker
        balanced_ranker = BalancedRanker()
        
        # Initialize recommendation engine
        recommendation_engine = RecommendationEngine(
            vector_database=vector_db,
            query_processor=query_processor,
            balanced_ranker=balanced_ranker
        )
        
        logger.info("SHL Assessment Recommendation System initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize recommendation system: {str(e)}")
        # Don't fail startup, but log the error
        recommendation_engine = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint returning API status, uptime, and assessment count.
    
    Returns information about the API status, uptime, loaded assessments,
    and system configuration for monitoring purposes.
    """
    try:
        # Calculate uptime
        uptime = time.time() - startup_time
        
        # Get assessment count
        assessment_count = 0
        if recommendation_engine and recommendation_engine.vector_database:
            stats = recommendation_engine.vector_database.get_database_stats()
            assessment_count = stats.get('total_assessments', 0)
        
        # Determine status
        status_value = "healthy" if recommendation_engine is not None else "degraded"
        
        return HealthResponse(
            status=status_value,
            version="1.0.0",
            uptime=uptime,
            assessment_count=assessment_count,
            environment=settings.environment
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get assessment recommendations based on job description or query.
    
    This endpoint accepts a job description or query text and returns
    relevant SHL assessments with relevance scores and explanations.
    The system uses LLM-based query understanding and vector similarity
    search to provide balanced recommendations across skill domains.
    """
    if recommendation_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation system is not available"
        )
    
    try:
        logger.info(f"Processing recommendation request: {request.query[:100]}...")
        
        # Generate recommendations
        result = recommendation_engine.recommend(
            query=request.query,
            max_results=request.max_results,
            balance_domains=request.balance_domains
        )
        
        # Convert to API response format
        recommendations = []
        for rec in result.recommendations:
            recommendations.append(AssessmentRecommendation(
                assessment_name=rec.assessment_name,
                url=rec.assessment_url,
                test_type=rec.test_type,
                category=rec.category,
                relevance_score=rec.relevance_score,
                explanation=rec.explanation,
                skills_matched=rec.skills_matched
            ))
        
        # Prepare query info for response (remove sensitive data)
        query_info = {
            "job_role": result.query_info.job_role,
            "job_level": result.query_info.job_level,
            "extracted_skills": result.query_info.extracted_skills[:10],  # Limit for response size
            "required_domains": result.query_info.required_domains,
            "confidence_score": result.query_info.confidence_score,
            "processing_method": result.query_info.processing_method
        }
        
        response = RecommendationResponse(
            query=request.query,
            recommendations=recommendations,
            total_results=len(recommendations),
            processing_time=result.processing_time,
            query_info=query_info,
            balance_info=result.balance_info
        )
        
        logger.info(f"Returned {len(recommendations)} recommendations in {result.processing_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Recommendation request failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Root endpoint providing basic API information.
    
    Returns basic information about the SHL Assessment Recommendation System API.
    """
    return {
        "message": "SHL Assessment Recommendation System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "recommend": "/recommend"
    }


@app.get("/stats")
async def get_system_stats():
    """
    Get system statistics and configuration information.
    
    Returns detailed information about the recommendation system
    configuration, loaded assessments, and performance metrics.
    """
    if recommendation_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation system is not available"
        )
    
    try:
        stats = recommendation_engine.get_recommendation_stats()
        stats['api'] = {
            'uptime': time.time() - startup_time,
            'environment': settings.environment,
            'version': '1.0.0'
        }
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system statistics"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )