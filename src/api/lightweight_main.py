"""
Lightweight FastAPI application optimized for Render free tier (512MB RAM).
"""

import logging
import time
import os
import gc
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from ..config_lite import get_lite_settings
from .models import (
    HealthResponse, 
    RecommendationRequest, 
    RecommendationResponse,
    ErrorResponse,
    ValidationErrorResponse,
    AssessmentRecommendation
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_lite_settings()

# Global variables - keep minimal
startup_time = time.time()
simple_assessments_db: List[Dict[str, Any]] = []
gemini_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lightweight application lifespan manager."""
    # Startup
    await lightweight_startup()
    yield
    # Shutdown - cleanup
    global simple_assessments_db, gemini_client
    simple_assessments_db.clear()
    gemini_client = None
    gc.collect()

app = FastAPI(
    title="SHL Assessment Recommendation System (Lite)",
    description="Memory-optimized recommendation system for SHL assessments",
    version="1.0.0-lite",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS setup with explicit frontend URL
frontend_url = os.getenv('FRONTEND_URL', 'https://shl-ai-intern-re-generative-ai-assi.vercel.app')
cors_origins = [
    "https://shl-ai-intern-re-generative-ai-assi.vercel.app",  # Explicit frontend URL
    frontend_url,  # Environment variable
    "https://localhost:3000",
    "http://localhost:3000", 
    "http://localhost:3001",
    "*"  # Allow all for now to debug
]

# Debug logging
logger.info(f"CORS Origins configured: {cors_origins}")
logger.info(f"Frontend URL from env: {frontend_url}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "validation_error", "message": "Invalid request data"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "message": "Something went wrong"}
    )

async def lightweight_startup() -> None:
    """Initialize with minimal memory footprint."""
    global simple_assessments_db, gemini_client
    
    try:
        logger.info("Starting lightweight SHL recommendation system...")
        
        # Initialize Gemini client if API key available
        if settings.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.gemini_api_key)
                gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini client initialized")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")
        
        # Load minimal assessment data
        simple_assessments_db = load_minimal_assessments()
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"Lightweight system ready with {len(simple_assessments_db)} assessments")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")

def load_minimal_assessments() -> List[Dict[str, Any]]:
    """Load a minimal set of assessments for demonstration."""
    # This is a hardcoded minimal dataset to avoid file I/O and reduce memory
    return [
        {
            "id": "shl_001",
            "name": "Numerical Reasoning Test",
            "description": "Assess numerical and analytical skills for data-driven roles",
            "url": "https://www.shl.com/assessments/numerical-reasoning/",
            "test_type": "K",
            "category": "Cognitive",
            "skills": ["numerical analysis", "data interpretation", "problem solving"],
            "keywords": ["numerical", "data", "analysis", "math", "statistics"]
        },
        {
            "id": "shl_002", 
            "name": "Verbal Reasoning Test",
            "description": "Evaluate verbal comprehension and critical thinking abilities",
            "url": "https://www.shl.com/assessments/verbal-reasoning/",
            "test_type": "K",
            "category": "Cognitive", 
            "skills": ["verbal comprehension", "critical thinking", "reading"],
            "keywords": ["verbal", "reading", "comprehension", "language", "communication"]
        },
        {
            "id": "shl_003",
            "name": "Personality Questionnaire",
            "description": "Assess personality traits and behavioral preferences",
            "url": "https://www.shl.com/assessments/personality/",
            "test_type": "P",
            "category": "Personality",
            "skills": ["teamwork", "leadership", "communication", "adaptability"],
            "keywords": ["personality", "behavior", "teamwork", "leadership", "culture"]
        },
        {
            "id": "shl_004",
            "name": "Situational Judgment Test",
            "description": "Evaluate decision-making in workplace scenarios",
            "url": "https://www.shl.com/assessments/situational-judgment/",
            "test_type": "K",
            "category": "Behavioral",
            "skills": ["decision making", "judgment", "workplace scenarios"],
            "keywords": ["judgment", "decision", "workplace", "scenarios", "ethics"]
        },
        {
            "id": "shl_005",
            "name": "Programming Skills Assessment",
            "description": "Test technical programming and coding abilities",
            "url": "https://www.shl.com/assessments/technical-skills/",
            "test_type": "K", 
            "category": "Technical",
            "skills": ["programming", "coding", "software development", "algorithms"],
            "keywords": ["programming", "coding", "software", "technical", "development"]
        }
    ]

def simple_text_similarity(query: str, assessment: Dict[str, Any]) -> float:
    """Calculate simple text similarity without ML models."""
    query_words = set(query.lower().split())
    
    # Combine assessment text fields
    assessment_text = f"{assessment['name']} {assessment['description']} {' '.join(assessment['skills'])} {' '.join(assessment['keywords'])}"
    assessment_words = set(assessment_text.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(query_words.intersection(assessment_words))
    union = len(query_words.union(assessment_words))
    
    if union == 0:
        return 0.0
    
    return intersection / union

async def get_llm_recommendations(query: str) -> List[Dict[str, Any]]:
    """Get recommendations using LLM if available."""
    if not gemini_client:
        return []
    
    try:
        prompt = f"""
        Based on this job description/query: "{query}"
        
        Recommend the most relevant assessments from these options:
        1. Numerical Reasoning Test - for data/analytical roles
        2. Verbal Reasoning Test - for communication-heavy roles  
        3. Personality Questionnaire - for team/culture fit
        4. Situational Judgment Test - for decision-making roles
        5. Programming Skills Assessment - for technical/coding roles
        
        Return only the numbers (1-5) of the top 3 most relevant assessments, separated by commas.
        Example: 1,3,5
        """
        
        response = gemini_client.generate_content(prompt)
        
        # Parse response
        numbers = []
        for char in response.text:
            if char.isdigit() and 1 <= int(char) <= 5:
                numbers.append(int(char) - 1)  # Convert to 0-based index
        
        return [simple_assessments_db[i] for i in numbers[:3] if i < len(simple_assessments_db)]
        
    except Exception as e:
        logger.error(f"LLM recommendation failed: {e}")
        return []

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy",
        version="1.0.0-lite", 
        uptime=uptime,
        assessment_count=len(simple_assessments_db),
        environment=settings.environment
    )

@app.get("/cors-test")
async def cors_test():
    """Simple CORS test endpoint."""
    return {
        "message": "CORS is working!",
        "frontend_url": os.getenv('FRONTEND_URL', 'not-set'),
        "timestamp": time.time()
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get assessment recommendations using lightweight methods."""
    try:
        logger.info(f"Processing request: {request.query[:50]}...")
        
        start_time = time.time()
        recommendations = []
        
        # Try LLM first if available
        llm_results = await get_llm_recommendations(request.query)
        
        if llm_results:
            # Use LLM results
            for i, assessment in enumerate(llm_results):
                recommendations.append(AssessmentRecommendation(
                    assessment_name=assessment["name"],
                    url=assessment["url"],
                    test_type=assessment["test_type"],
                    category=assessment["category"],
                    relevance_score=0.9 - (i * 0.1),  # Decreasing scores
                    explanation=f"Recommended by AI analysis for your requirements",
                    skills_matched=assessment["skills"][:3]
                ))
        else:
            # Fallback to simple similarity
            scored_assessments = []
            for assessment in simple_assessments_db:
                score = simple_text_similarity(request.query, assessment)
                if score > 0:
                    scored_assessments.append((assessment, score))
            
            # Sort by score and take top results
            scored_assessments.sort(key=lambda x: x[1], reverse=True)
            
            for assessment, score in scored_assessments[:request.max_results]:
                recommendations.append(AssessmentRecommendation(
                    assessment_name=assessment["name"],
                    url=assessment["url"],
                    test_type=assessment["test_type"],
                    category=assessment["category"],
                    relevance_score=min(score * 2, 1.0),  # Scale score
                    explanation=f"Matches {int(score*100)}% of your requirements",
                    skills_matched=assessment["skills"][:3]
                ))
        
        processing_time = time.time() - start_time
        
        return RecommendationResponse(
            query=request.query,
            recommendations=recommendations,
            total_results=len(recommendations),
            processing_time=processing_time,
            query_info={
                "processing_method": "llm" if llm_results else "similarity",
                "confidence_score": 0.8 if llm_results else 0.6
            },
            balance_info={}
        )
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SHL Assessment Recommendation System (Lite)",
        "version": "1.0.0-lite",
        "memory_optimized": True,
        "docs": "/docs"
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "assessments_loaded": len(simple_assessments_db),
        "uptime": time.time() - startup_time,
        "llm_available": gemini_client is not None,
        "memory_optimized": True,
        "environment": settings.environment
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "src.api.lightweight_main:app",
        host="0.0.0.0",
        port=port,
        workers=1,  # Single worker for memory efficiency
        log_level=settings.log_level.lower(),
    )