"""
SHL Catalog Scraper Module

This module provides functionality to scrape assessment data from SHL's product catalog.
It extracts individual test solutions while filtering out pre-packaged job solutions.
"""

import time
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup, Tag
import re


@dataclass
class Assessment:
    """Data class representing an SHL assessment."""
    id: str
    name: str
    url: str
    category: str
    test_type: str  # 'K' for Knowledge & Skills, 'P' for Personality & Behavior
    description: str
    skills: List[str]


class SHLCatalogScraper:
    """
    Scraper for SHL's product catalog to extract individual test solutions.
    
    This class navigates the SHL product catalog and extracts assessment data
    while filtering out pre-packaged job solutions to focus on individual tests.
    """
    
    BASE_URL = "https://www.shl.com"
    CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the scraper with retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic and error handling.
        
        Args:
            url: URL to request
            
        Returns:
            Response object if successful, None if all retries failed
        """
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.info(f"Requesting URL: {url} (attempt {attempt + 1})")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"All retry attempts failed for URL: {url}")
                    return None
        
        return None
    
    def _parse_assessment_details(self, assessment_url: str) -> Optional[Dict[str, Any]]:
        """
        Parse detailed information from an individual assessment page.
        
        Args:
            assessment_url: URL of the assessment page
            
        Returns:
            Dictionary containing assessment details or None if parsing failed
        """
        response = self._make_request(assessment_url)
        if not response:
            return None
            
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract assessment name
            name_element = soup.find('h1') or soup.find('title')
            name = name_element.get_text(strip=True) if name_element else "Unknown Assessment"
            
            # Extract description
            description = ""
            desc_selectors = [
                '.hero-content p',
                '.content-section p',
                '.description p',
                'meta[name="description"]'
            ]
            
            for selector in desc_selectors:
                if selector.startswith('meta'):
                    desc_element = soup.select_one(selector)
                    if desc_element:
                        description = desc_element.get('content', '')
                        break
                else:
                    desc_elements = soup.select(selector)
                    if desc_elements:
                        description = ' '.join([elem.get_text(strip=True) for elem in desc_elements[:2]])
                        break
            
            # Determine test type based on content analysis
            content_text = soup.get_text().lower()
            test_type = self._determine_test_type(content_text, name.lower())
            
            # Extract skills/categories
            skills = self._extract_skills(soup, content_text)
            
            return {
                'name': name,
                'description': description,
                'test_type': test_type,
                'skills': skills
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing assessment details from {assessment_url}: {e}")
            return None
    
    def _determine_test_type(self, content_text: str, name_text: str) -> str:
        """
        Determine if assessment is Knowledge & Skills (K) or Personality & Behavior (P).
        
        Args:
            content_text: Full text content of the page
            name_text: Assessment name text
            
        Returns:
            'K' for Knowledge & Skills, 'P' for Personality & Behavior
        """
        # Keywords that indicate Knowledge & Skills assessments
        knowledge_keywords = [
            'technical', 'coding', 'programming', 'software', 'numerical', 'verbal',
            'logical', 'reasoning', 'aptitude', 'cognitive', 'ability', 'skill',
            'competency', 'knowledge', 'mathematics', 'language', 'analytical'
        ]
        
        # Keywords that indicate Personality & Behavior assessments
        personality_keywords = [
            'personality', 'behavior', 'behaviour', 'motivation', 'values',
            'leadership', 'teamwork', 'communication', 'emotional', 'social',
            'cultural', 'fit', 'style', 'preference', 'trait', 'character'
        ]
        
        # Count keyword matches
        knowledge_score = sum(1 for keyword in knowledge_keywords if keyword in content_text or keyword in name_text)
        personality_score = sum(1 for keyword in personality_keywords if keyword in content_text or keyword in name_text)
        
        # Default to Knowledge & Skills if unclear
        return 'P' if personality_score > knowledge_score else 'K'
    
    def _extract_skills(self, soup: BeautifulSoup, content_text: str) -> List[str]:
        """
        Extract relevant skills from the assessment page.
        
        Args:
            soup: BeautifulSoup object of the page
            content_text: Full text content of the page
            
        Returns:
            List of extracted skills
        """
        skills = []
        
        # Look for skills in structured elements
        skill_selectors = [
            '.skills li',
            '.competencies li',
            '.measures li',
            '.tags .tag'
        ]
        
        for selector in skill_selectors:
            elements = soup.select(selector)
            for element in elements:
                skill = element.get_text(strip=True)
                if skill and len(skill) < 50:  # Reasonable skill name length
                    skills.append(skill)
        
        # If no structured skills found, extract from common skill patterns
        if not skills:
            skill_patterns = [
                r'measures?\s+([^.]+)',
                r'assesses?\s+([^.]+)',
                r'evaluates?\s+([^.]+)'
            ]
            
            for pattern in skill_patterns:
                matches = re.findall(pattern, content_text, re.IGNORECASE)
                for match in matches[:3]:  # Limit to first 3 matches
                    clean_skill = re.sub(r'[^\w\s]', '', match).strip()
                    if clean_skill and len(clean_skill) < 50:
                        skills.append(clean_skill)
        
        return list(set(skills))  # Remove duplicates
    
    def _is_individual_test_solution(self, assessment_data: Dict[str, Any]) -> bool:
        """
        Check if an assessment is an individual test solution (not pre-packaged).
        
        Args:
            assessment_data: Dictionary containing assessment information
            
        Returns:
            True if it's an individual test solution, False otherwise
        """
        name = assessment_data.get('name', '').lower()
        description = assessment_data.get('description', '').lower()
        
        # Exclude pre-packaged job solutions
        exclusion_keywords = [
            'pre-packaged', 'job solution', 'package', 'bundle',
            'suite', 'collection', 'set of assessments'
        ]
        
        for keyword in exclusion_keywords:
            if keyword in name or keyword in description:
                return False
        
        return True
    
    def scrape_catalog(self) -> List[Assessment]:
        """
        Scrape the complete SHL product catalog for individual test solutions.
        
        Returns:
            List of Assessment objects containing scraped data
        """
        self.logger.info("Starting SHL catalog scraping...")
        
        # Get the main catalog page
        response = self._make_request(self.CATALOG_URL)
        if not response:
            self.logger.error("Failed to fetch catalog page")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        assessments = []
        
        # Find assessment links - try multiple selectors
        link_selectors = [
            'a[href*="/assessment"]',
            'a[href*="/product"]',
            'a[href*="/solution"]',
            '.product-card a',
            '.assessment-card a',
            '.card a'
        ]
        
        assessment_links = set()
        for selector in link_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(self.BASE_URL, href)
                    assessment_links.add(full_url)
        
        self.logger.info(f"Found {len(assessment_links)} potential assessment links")
        
        # Process each assessment link
        for i, assessment_url in enumerate(assessment_links, 1):
            self.logger.info(f"Processing assessment {i}/{len(assessment_links)}: {assessment_url}")
            
            try:
                # Parse assessment details
                details = self._parse_assessment_details(assessment_url)
                if not details:
                    continue
                
                # Check if it's an individual test solution
                if not self._is_individual_test_solution(details):
                    self.logger.info(f"Skipping pre-packaged solution: {details.get('name', 'Unknown')}")
                    continue
                
                # Create Assessment object
                assessment = Assessment(
                    id=str(len(assessments) + 1),
                    name=details['name'],
                    url=assessment_url,
                    category=self._extract_category(assessment_url),
                    test_type=details['test_type'],
                    description=details['description'],
                    skills=details['skills']
                )
                
                assessments.append(assessment)
                self.logger.info(f"Added assessment: {assessment.name}")
                
                # Add small delay to be respectful
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error processing assessment {assessment_url}: {e}")
                continue
        
        self.logger.info(f"Scraping completed. Found {len(assessments)} individual test solutions")
        return assessments
    
    def _extract_category(self, url: str) -> str:
        """
        Extract category from URL path or return default.
        
        Args:
            url: Assessment URL
            
        Returns:
            Category string
        """
        try:
            path = urlparse(url).path
            path_parts = [part for part in path.split('/') if part]
            
            # Look for category indicators in URL
            category_indicators = ['cognitive', 'personality', 'behavioral', 'technical', 'skills']
            for part in path_parts:
                for indicator in category_indicators:
                    if indicator in part.lower():
                        return indicator.title()
            
            return "General"
            
        except Exception:
            return "General"
    
    def get_individual_tests(self) -> List[Assessment]:
        """
        Get individual test solutions from SHL catalog.
        
        Returns:
            List of individual test Assessment objects
        """
        return self.scrape_catalog()
    
    def validate_minimum_count(self, assessments: List[Assessment]) -> bool:
        """
        Validate that minimum required number of assessments were collected.
        
        Args:
            assessments: List of scraped assessments
            
        Returns:
            True if minimum count (377) is met, False otherwise
        """
        min_required = 377
        actual_count = len(assessments)
        
        self.logger.info(f"Validation: Found {actual_count} assessments (minimum required: {min_required})")
        
        if actual_count >= min_required:
            self.logger.info("✓ Minimum assessment count validation passed")
            return True
        else:
            self.logger.warning(f"✗ Minimum assessment count validation failed: {actual_count} < {min_required}")
            return False


class DataValidator:
    """Utility class for validating scraped assessment data."""
    
    @staticmethod
    def validate_assessment(assessment: Assessment) -> bool:
        """
        Validate that an assessment has all required fields.
        
        Args:
            assessment: Assessment object to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['id', 'name', 'url', 'category', 'test_type', 'description']
        
        for field in required_fields:
            value = getattr(assessment, field, None)
            if not value or (isinstance(value, str) and not value.strip()):
                return False
        
        # Validate test_type is either 'K' or 'P'
        if assessment.test_type not in ['K', 'P']:
            return False
        
        # Validate URL format
        if not assessment.url.startswith(('http://', 'https://')):
            return False
        
        return True
    
    @staticmethod
    def validate_assessment_list(assessments: List[Assessment]) -> Dict[str, Any]:
        """
        Validate a list of assessments and return validation report.
        
        Args:
            assessments: List of Assessment objects
            
        Returns:
            Dictionary containing validation results
        """
        total_count = len(assessments)
        valid_count = sum(1 for assessment in assessments if DataValidator.validate_assessment(assessment))
        invalid_count = total_count - valid_count
        
        # Count by test type
        k_type_count = sum(1 for assessment in assessments if assessment.test_type == 'K')
        p_type_count = sum(1 for assessment in assessments if assessment.test_type == 'P')
        
        return {
            'total_assessments': total_count,
            'valid_assessments': valid_count,
            'invalid_assessments': invalid_count,
            'validation_rate': valid_count / total_count if total_count > 0 else 0,
            'knowledge_skills_count': k_type_count,
            'personality_behavior_count': p_type_count,
            'meets_minimum_requirement': total_count >= 377
        }