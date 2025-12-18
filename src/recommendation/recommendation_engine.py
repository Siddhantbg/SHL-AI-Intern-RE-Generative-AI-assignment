"""
Core recommendation engine with balancing logic for SHL assessment recommendations.

This module provides the RecommendationEngine class that orchestrates the recommendation
process, including similarity-based ranking and balanced selection of assessments.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from ..processing.vector_database import VectorDatabase, AssessmentVector
from .query_processor import QueryProcessor, ProcessedQuery
from .balanced_ranker import BalancedRanker

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Represents a single assessment recommendation with metadata."""
    assessment_id: str
    assessment_name: str
    assessment_url: str
    test_type: str
    category: str
    relevance_score: float
    explanation: str
    skills_matched: List[str]


@dataclass
class RecommendationResult:
    """Complete recommendation result with metadata."""
    query: str
    recommendations: List[Recommendation]
    total_candidates: int
    processing_time: float
    query_info: ProcessedQuery
    balance_info: Dict[str, Any]


class RecommendationEngine:
    """
    Main recommendation orchestrator that combines query processing, similarity search,
    and balanced ranking to provide relevant SHL assessment recommendations.
    """
    
    def __init__(self, vector_database: VectorDatabase, 
                 query_processor: Optional[QueryProcessor] = None,
                 balanced_ranker: Optional[BalancedRanker] = None):
        """
        Initialize the recommendation engine.
        
        Args:
            vector_database: Vector database for similarity search
            query_processor: Query processor for understanding user queries
            balanced_ranker: Balanced ranker for ensuring domain balance
        """
        self.vector_database = vector_database
        self.query_processor = query_processor or QueryProcessor()
        self.balanced_ranker = balanced_ranker or BalancedRanker()
        
        # Configuration parameters
        self.max_candidates = 50  # Maximum candidates to consider before balancing
        self.min_similarity_threshold = 0.1  # Minimum similarity score to consider
        
        logger.info("RecommendationEngine initialized")
    
    def recommend(self, query: str, max_results: int = 10, 
                 balance_domains: bool = True) -> RecommendationResult:
        """
        Generate recommendations for a user query.
        
        Args:
            query: User query or job description
            max_results: Maximum number of recommendations to return (1-10)
            balance_domains: Whether to balance between technical and behavioral assessments
            
        Returns:
            RecommendationResult with recommendations and metadata
        """
        import time
        start_time = time.time()
        
        # Validate input parameters
        max_results = max(1, min(10, max_results))
        
        logger.info(f"Generating recommendations for query: {query[:100]}...")
        
        try:
            # Step 1: Process the query
            processed_query = self.query_processor.process_query(query)
            logger.debug(f"Query processed: {processed_query.job_role}, domains: {processed_query.required_domains}")
            
            # Step 2: Perform similarity search
            candidates = self._search_similar_assessments(processed_query)
            logger.debug(f"Found {len(candidates)} candidate assessments")
            
            # Step 3: Apply business logic and filtering
            filtered_candidates = self._apply_business_logic(candidates, processed_query)
            logger.debug(f"After business logic filtering: {len(filtered_candidates)} candidates")
            
            # Step 4: Balance recommendations across domains if requested
            if balance_domains:
                balanced_recommendations = self.balanced_ranker.balance_recommendations(
                    filtered_candidates, processed_query, max_results
                )
                balance_info = self.balanced_ranker.get_balance_info()
            else:
                # Just take top results without balancing
                balanced_recommendations = filtered_candidates[:max_results]
                balance_info = {"balanced": False, "reason": "balancing_disabled"}
            
            # Step 5: Create final recommendations
            recommendations = self._create_recommendations(
                balanced_recommendations, processed_query
            )
            
            processing_time = time.time() - start_time
            
            result = RecommendationResult(
                query=query,
                recommendations=recommendations,
                total_candidates=len(candidates),
                processing_time=processing_time,
                query_info=processed_query,
                balance_info=balance_info
            )
            
            logger.info(f"Generated {len(recommendations)} recommendations in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            # Return empty result on error
            return RecommendationResult(
                query=query,
                recommendations=[],
                total_candidates=0,
                processing_time=time.time() - start_time,
                query_info=ProcessedQuery(
                    original_text=query,
                    cleaned_text="",
                    extracted_skills=[],
                    job_role="unknown",
                    job_level="unknown",
                    required_domains=[],
                    confidence_score=0.0,
                    processing_method="error"
                ),
                balance_info={"error": str(e)}
            )
    
    def _search_similar_assessments(self, processed_query: ProcessedQuery) -> List[Tuple[AssessmentVector, float]]:
        """
        Search for similar assessments using vector similarity.
        
        Args:
            processed_query: Processed query with embedding
            
        Returns:
            List of (AssessmentVector, similarity_score) tuples
        """
        if processed_query.embedding is None:
            logger.warning("Query embedding is None, cannot perform similarity search")
            return []
        
        try:
            # Search for similar assessments
            candidates = self.vector_database.search_similar(
                query_embedding=processed_query.embedding,
                k=self.max_candidates
            )
            
            # Filter by minimum similarity threshold
            filtered_candidates = [
                (assessment, score) for assessment, score in candidates
                if score >= self.min_similarity_threshold
            ]
            
            logger.debug(f"Similarity search: {len(candidates)} -> {len(filtered_candidates)} after threshold filter")
            return filtered_candidates
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []
    
    def _apply_business_logic(self, candidates: List[Tuple[AssessmentVector, float]], 
                            processed_query: ProcessedQuery) -> List[Tuple[AssessmentVector, float]]:
        """
        Apply business logic and filtering to candidate assessments.
        
        Args:
            candidates: List of candidate assessments with similarity scores
            processed_query: Processed query information
            
        Returns:
            Filtered list of candidates
        """
        filtered_candidates = []
        
        for assessment, similarity_score in candidates:
            # Apply domain-specific filtering
            if self._should_include_assessment(assessment, processed_query):
                # Adjust score based on business rules
                adjusted_score = self._adjust_similarity_score(
                    assessment, similarity_score, processed_query
                )
                filtered_candidates.append((assessment, adjusted_score))
        
        # Sort by adjusted similarity score
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_candidates
    
    def _should_include_assessment(self, assessment: AssessmentVector, 
                                 processed_query: ProcessedQuery) -> bool:
        """
        Determine if an assessment should be included based on business rules.
        
        Args:
            assessment: Assessment to evaluate
            processed_query: Processed query information
            
        Returns:
            True if assessment should be included
        """
        # Always include if no domain requirements specified
        if not processed_query.required_domains:
            return True
        
        # Map test types to domains
        domain_mapping = {
            'K': ['technical', 'cognitive'],  # Knowledge & Skills assessments
            'P': ['behavioral']  # Personality & Behavior assessments
        }
        
        assessment_domains = domain_mapping.get(assessment.test_type, [])
        
        # Include if assessment matches any required domain
        return any(domain in processed_query.required_domains for domain in assessment_domains)
    
    def _adjust_similarity_score(self, assessment: AssessmentVector, 
                               similarity_score: float, 
                               processed_query: ProcessedQuery) -> float:
        """
        Adjust similarity score based on business rules and query context.
        
        Args:
            assessment: Assessment being scored
            similarity_score: Original similarity score
            processed_query: Processed query information
            
        Returns:
            Adjusted similarity score
        """
        adjusted_score = similarity_score
        
        # Boost score for exact skill matches
        if processed_query.extracted_skills:
            assessment_skills = assessment.metadata.get('skills', [])
            skill_matches = len(set(processed_query.extracted_skills) & set(assessment_skills))
            if skill_matches > 0:
                skill_boost = min(0.1, skill_matches * 0.02)  # Max 10% boost
                adjusted_score += skill_boost
        
        # Boost score for job level alignment
        job_level_boost = self._calculate_job_level_boost(assessment, processed_query)
        adjusted_score += job_level_boost
        
        # Ensure score doesn't exceed 1.0
        return min(adjusted_score, 1.0)
    
    def _calculate_job_level_boost(self, assessment: AssessmentVector, 
                                 processed_query: ProcessedQuery) -> float:
        """
        Calculate job level alignment boost for similarity score.
        
        Args:
            assessment: Assessment being evaluated
            processed_query: Processed query information
            
        Returns:
            Boost value to add to similarity score
        """
        # Simple job level mapping - could be enhanced with more sophisticated logic
        level_keywords = {
            'entry': ['entry', 'junior', 'graduate', 'trainee'],
            'mid': ['mid', 'intermediate', 'experienced'],
            'senior': ['senior', 'lead', 'principal', 'expert'],
            'executive': ['director', 'manager', 'executive', 'head']
        }
        
        assessment_text = (assessment.name + " " + assessment.metadata.get('description', '')).lower()
        query_level = processed_query.job_level
        
        if query_level in level_keywords:
            for keyword in level_keywords[query_level]:
                if keyword in assessment_text:
                    return 0.05  # 5% boost for level alignment
        
        return 0.0
    
    def _create_recommendations(self, ranked_assessments: List[Tuple[AssessmentVector, float]], 
                              processed_query: ProcessedQuery) -> List[Recommendation]:
        """
        Create final recommendation objects from ranked assessments.
        
        Args:
            ranked_assessments: List of ranked assessments with scores
            processed_query: Processed query information
            
        Returns:
            List of Recommendation objects
        """
        recommendations = []
        
        for assessment, score in ranked_assessments:
            # Calculate skill matches
            skills_matched = self._calculate_skill_matches(assessment, processed_query)
            
            # Generate explanation
            explanation = self._generate_explanation(assessment, score, skills_matched, processed_query)
            
            recommendation = Recommendation(
                assessment_id=assessment.id,
                assessment_name=assessment.name,
                assessment_url=assessment.url,
                test_type=assessment.test_type,
                category=assessment.category,
                relevance_score=score,
                explanation=explanation,
                skills_matched=skills_matched
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_skill_matches(self, assessment: AssessmentVector, 
                               processed_query: ProcessedQuery) -> List[str]:
        """
        Calculate which skills from the query match the assessment.
        
        Args:
            assessment: Assessment to check
            processed_query: Processed query information
            
        Returns:
            List of matching skills
        """
        if not processed_query.extracted_skills:
            return []
        
        assessment_skills = assessment.metadata.get('skills', [])
        if not assessment_skills:
            return []
        
        # Find exact matches
        query_skills_set = set(skill.lower() for skill in processed_query.extracted_skills)
        assessment_skills_set = set(skill.lower() for skill in assessment_skills)
        
        matches = list(query_skills_set & assessment_skills_set)
        
        # Find partial matches (substring matching)
        for query_skill in processed_query.extracted_skills:
            for assessment_skill in assessment_skills:
                if (query_skill.lower() in assessment_skill.lower() or 
                    assessment_skill.lower() in query_skill.lower()) and \
                   query_skill.lower() not in matches:
                    matches.append(query_skill.lower())
        
        return matches[:5]  # Limit to top 5 matches
    
    def _generate_explanation(self, assessment: AssessmentVector, score: float, 
                            skills_matched: List[str], processed_query: ProcessedQuery) -> str:
        """
        Generate human-readable explanation for the recommendation.
        
        Args:
            assessment: Recommended assessment
            score: Relevance score
            skills_matched: List of matching skills
            processed_query: Processed query information
            
        Returns:
            Explanation string
        """
        explanation_parts = []
        
        # Base relevance
        if score > 0.8:
            explanation_parts.append("Highly relevant match")
        elif score > 0.6:
            explanation_parts.append("Good match")
        elif score > 0.4:
            explanation_parts.append("Moderate match")
        else:
            explanation_parts.append("Basic match")
        
        # Test type explanation
        test_type_desc = {
            'K': 'knowledge and skills assessment',
            'P': 'personality and behavioral assessment'
        }
        explanation_parts.append(f"for {test_type_desc.get(assessment.test_type, 'assessment')}")
        
        # Skill matches
        if skills_matched:
            if len(skills_matched) == 1:
                explanation_parts.append(f"matching '{skills_matched[0]}' skill")
            else:
                explanation_parts.append(f"matching {len(skills_matched)} skills: {', '.join(skills_matched[:3])}")
        
        # Job role alignment
        if processed_query.job_role != 'unknown':
            explanation_parts.append(f"suitable for {processed_query.job_role} role")
        
        return "; ".join(explanation_parts) + "."
    
    def get_recommendation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the recommendation engine configuration.
        
        Returns:
            Dictionary with engine statistics
        """
        db_stats = self.vector_database.get_database_stats()
        processor_info = self.query_processor.get_processor_info()
        
        return {
            'vector_database': db_stats,
            'query_processor': processor_info,
            'configuration': {
                'max_candidates': self.max_candidates,
                'min_similarity_threshold': self.min_similarity_threshold
            },
            'balanced_ranker': self.balanced_ranker.get_ranker_info()
        }
    
    def update_configuration(self, config: Dict[str, Any]) -> None:
        """
        Update recommendation engine configuration.
        
        Args:
            config: Configuration dictionary with new settings
        """
        if 'max_candidates' in config:
            self.max_candidates = max(10, min(100, config['max_candidates']))
        
        if 'min_similarity_threshold' in config:
            self.min_similarity_threshold = max(0.0, min(1.0, config['min_similarity_threshold']))
        
        logger.info(f"Configuration updated: max_candidates={self.max_candidates}, "
                   f"min_similarity_threshold={self.min_similarity_threshold}")