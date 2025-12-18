"""
Assessment data processing module for cleaning and normalizing scraped assessment data.

This module provides the AssessmentProcessor class that handles text preprocessing,
cleaning, and normalization of assessment data scraped from SHL's catalog.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
from src.scraper.shl_catalog_scraper import Assessment

logger = logging.getLogger(__name__)


@dataclass
class ProcessedAssessment:
    """Processed assessment with cleaned and normalized data."""
    id: str
    name: str
    url: str
    category: str
    test_type: str
    description: str
    skills: List[str]
    cleaned_text: str
    normalized_description: str
    token_count: int
    quality_score: float = 0.0
    validation_errors: List[str] = field(default_factory=list)


class AssessmentProcessor:
    """
    Processes and cleans scraped assessment data for embedding generation.
    
    This class handles text preprocessing, normalization, and quality validation
    of assessment data to prepare it for vector embedding generation.
    """
    
    def __init__(self):
        """Initialize the assessment processor."""
        self.min_description_length = 10
        self.max_description_length = 5000
        self.required_fields = ['id', 'name', 'url', 'category', 'test_type', 'description']
        
    def process_assessments(self, raw_assessments: List[Assessment]) -> List[ProcessedAssessment]:
        """
        Process a list of raw assessments into cleaned, normalized format.
        
        Args:
            raw_assessments: List of Assessment objects from scraper
            
        Returns:
            List of ProcessedAssessment objects with cleaned data
        """
        processed_assessments = []
        
        logger.info(f"Processing {len(raw_assessments)} assessments")
        
        for assessment in raw_assessments:
            try:
                processed = self._process_single_assessment(assessment)
                if processed:
                    processed_assessments.append(processed)
            except Exception as e:
                logger.error(f"Error processing assessment {assessment.id}: {str(e)}")
                continue
                
        logger.info(f"Successfully processed {len(processed_assessments)} assessments")
        return processed_assessments
    
    def _process_single_assessment(self, assessment: Assessment) -> Optional[ProcessedAssessment]:
        """
        Process a single assessment with validation and cleaning.
        
        Args:
            assessment: Raw Assessment object
            
        Returns:
            ProcessedAssessment object or None if validation fails
        """
        # Validate required fields
        validation_errors = self._validate_assessment(assessment)
        
        # Clean and normalize text
        cleaned_description = self._clean_text(assessment.description)
        normalized_description = self._normalize_text(cleaned_description)
        
        # Create combined text for embedding
        cleaned_text = self._create_combined_text(assessment, normalized_description)
        
        # Calculate quality metrics
        token_count = len(normalized_description.split())
        quality_score = self._calculate_quality_score(assessment, normalized_description)
        
        # Create processed assessment
        processed = ProcessedAssessment(
            id=assessment.id,
            name=assessment.name.strip(),
            url=assessment.url,
            category=assessment.category.strip(),
            test_type=assessment.test_type,
            description=assessment.description,
            skills=self._clean_skills_list(assessment.skills),
            cleaned_text=cleaned_text,
            normalized_description=normalized_description,
            token_count=token_count,
            quality_score=quality_score,
            validation_errors=validation_errors
        )
        
        # Filter out low-quality assessments
        if quality_score < 0.3:
            logger.warning(f"Low quality assessment filtered out: {assessment.id}")
            return None
            
        return processed
    
    def _validate_assessment(self, assessment: Assessment) -> List[str]:
        """
        Validate assessment data quality and completeness.
        
        Args:
            assessment: Assessment object to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if not hasattr(assessment, field) or not getattr(assessment, field):
                errors.append(f"Missing required field: {field}")
        
        # Validate description length
        if hasattr(assessment, 'description') and assessment.description:
            desc_len = len(assessment.description.strip())
            if desc_len < self.min_description_length:
                errors.append(f"Description too short: {desc_len} characters")
            elif desc_len > self.max_description_length:
                errors.append(f"Description too long: {desc_len} characters")
        
        # Validate test type
        if hasattr(assessment, 'test_type') and assessment.test_type not in ['K', 'P']:
            errors.append(f"Invalid test type: {assessment.test_type}")
        
        # Validate URL format
        if hasattr(assessment, 'url') and assessment.url:
            if not assessment.url.startswith('http'):
                errors.append("Invalid URL format")
        
        return errors
    
    def _clean_text(self, text: str) -> str:
        """
        Clean raw text by removing unwanted characters and formatting.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', '', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[,]{2,}', ',', text)
        
        return text.strip()
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent processing.
        
        Args:
            text: Cleaned text to normalize
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Normalize common abbreviations and terms
        replacements = {
            'assessment': 'assessment',
            'test': 'test',
            'evaluation': 'evaluation',
            'skill': 'skill',
            'ability': 'ability',
            'personality': 'personality',
            'behavior': 'behavior',
            'behaviour': 'behavior',  # UK to US spelling
            'cognitive': 'cognitive',
            'aptitude': 'aptitude'
        }
        
        for old, new in replacements.items():
            text = re.sub(r'\b' + old + r'\b', new, text)
        
        return text.strip()
    
    def _create_combined_text(self, assessment: Assessment, normalized_description: str) -> str:
        """
        Create combined text for embedding generation.
        
        Args:
            assessment: Original assessment object
            normalized_description: Normalized description text
            
        Returns:
            Combined text string for embedding
        """
        components = []
        
        # Add assessment name
        if assessment.name:
            components.append(assessment.name.lower())
        
        # Add category
        if assessment.category:
            components.append(assessment.category.lower())
        
        # Add skills
        if assessment.skills:
            skills_text = ' '.join(assessment.skills).lower()
            components.append(skills_text)
        
        # Add normalized description
        if normalized_description:
            components.append(normalized_description)
        
        return ' '.join(components)
    
    def _clean_skills_list(self, skills: List[str]) -> List[str]:
        """
        Clean and normalize skills list.
        
        Args:
            skills: List of skill strings
            
        Returns:
            Cleaned list of skills
        """
        if not skills:
            return []
        
        cleaned_skills = []
        for skill in skills:
            if skill and isinstance(skill, str):
                # Clean and normalize skill text
                cleaned_skill = self._clean_text(skill).lower()
                if cleaned_skill and len(cleaned_skill) > 2:
                    cleaned_skills.append(cleaned_skill)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in cleaned_skills:
            if skill not in seen:
                seen.add(skill)
                unique_skills.append(skill)
        
        return unique_skills
    
    def _calculate_quality_score(self, assessment: Assessment, normalized_description: str) -> float:
        """
        Calculate quality score for an assessment.
        
        Args:
            assessment: Assessment object
            normalized_description: Normalized description text
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Base score for having required fields
        if assessment.name and assessment.description and assessment.url:
            score += 0.3
        
        # Score for description quality
        if normalized_description:
            desc_len = len(normalized_description.split())
            if desc_len >= 10:
                score += 0.2
            if desc_len >= 20:
                score += 0.1
        
        # Score for having skills
        if assessment.skills and len(assessment.skills) > 0:
            score += 0.2
        
        # Score for having category
        if assessment.category and assessment.category.strip():
            score += 0.1
        
        # Score for valid test type
        if assessment.test_type in ['K', 'P']:
            score += 0.1
        
        return min(score, 1.0)
    
    def get_processing_stats(self, processed_assessments: List[ProcessedAssessment]) -> Dict[str, Any]:
        """
        Get statistics about processed assessments.
        
        Args:
            processed_assessments: List of processed assessments
            
        Returns:
            Dictionary with processing statistics
        """
        if not processed_assessments:
            return {}
        
        df = pd.DataFrame([
            {
                'test_type': a.test_type,
                'token_count': a.token_count,
                'quality_score': a.quality_score,
                'has_skills': len(a.skills) > 0,
                'validation_errors': len(a.validation_errors)
            }
            for a in processed_assessments
        ])
        
        stats = {
            'total_assessments': len(processed_assessments),
            'test_type_distribution': df['test_type'].value_counts().to_dict(),
            'avg_token_count': df['token_count'].mean(),
            'avg_quality_score': df['quality_score'].mean(),
            'assessments_with_skills': df['has_skills'].sum(),
            'assessments_with_errors': (df['validation_errors'] > 0).sum(),
            'token_count_stats': {
                'min': df['token_count'].min(),
                'max': df['token_count'].max(),
                'median': df['token_count'].median()
            }
        }
        
        return stats