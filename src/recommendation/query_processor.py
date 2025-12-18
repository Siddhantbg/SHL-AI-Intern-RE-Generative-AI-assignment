"""
Query processing module for understanding and extracting information from user queries.

This module provides the QueryProcessor class that uses LLM-based techniques and
sentence transformers to understand job descriptions and extract relevant information
for assessment recommendations.
"""

import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    # Try the new google-genai package first
    import google.genai as genai
    GENAI_PACKAGE = "google.genai"
except ImportError:
    try:
        # Fallback to the deprecated google-generativeai package
        import google.generativeai as genai
        GENAI_PACKAGE = "google.generativeai"
    except ImportError:
        genai = None
        GENAI_PACKAGE = None
from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Represents a processed user query with extracted information."""
    
    original_text: str
    cleaned_text: str
    extracted_skills: List[str]
    job_role: str
    job_level: str
    required_domains: List[str]  # 'technical', 'behavioral', 'cognitive'
    embedding: Optional[np.ndarray] = None
    confidence_score: float = 0.0
    processing_method: str = "unknown"  # 'llm', 'fallback', 'hybrid'


class QueryProcessor:
    """
    Processes user queries to extract skills, job roles, and generate embeddings.
    
    This class uses LLM-based techniques for enhanced query understanding and
    falls back to rule-based methods when LLM services are unavailable.
    """
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the query processor.
        
        Args:
            embedding_model_name: Name of the sentence transformer model for embeddings
        """
        self.settings = get_settings()
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.gemini_model = None
        
        # Initialize LLM if API key is available
        self._initialize_llm()
        
        # Skill extraction patterns for fallback
        self._skill_patterns = self._load_skill_patterns()
        self._job_role_patterns = self._load_job_role_patterns()
    
    def _initialize_llm(self) -> None:
        """Initialize the Gemini LLM if API key is available."""
        try:
            if genai is None:
                logger.warning("No Gemini package available, will use fallback methods only")
                return
                
            if self.settings.gemini_api_key:
                if GENAI_PACKAGE == "google.genai":
                    # New package initialization
                    client = genai.Client(api_key=self.settings.gemini_api_key)
                    self.gemini_model = client
                else:
                    # Legacy package initialization
                    genai.configure(api_key=self.settings.gemini_api_key)
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info(f"Gemini LLM initialized successfully using {GENAI_PACKAGE}")
            else:
                logger.warning("Gemini API key not found, will use fallback methods only")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
            self.gemini_model = None
    
    def _load_embedding_model(self) -> None:
        """Load the sentence transformer model for embeddings."""
        if not self.embedding_model:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {str(e)}")
                raise
    
    def _load_skill_patterns(self) -> Dict[str, List[str]]:
        """Load skill extraction patterns for fallback processing."""
        return {
            'technical_skills': [
                r'\b(?:python|java|javascript|c\+\+|sql|html|css|react|angular|vue)\b',
                r'\b(?:machine learning|data science|artificial intelligence|ai|ml)\b',
                r'\b(?:cloud|aws|azure|gcp|docker|kubernetes)\b',
                r'\b(?:database|mongodb|postgresql|mysql|redis)\b',
                r'\b(?:api|rest|graphql|microservices)\b',
                r'\b(?:testing|unit test|integration test|qa)\b',
                r'\b(?:agile|scrum|devops|ci/cd)\b'
            ],
            'soft_skills': [
                r'\b(?:leadership|management|communication|teamwork)\b',
                r'\b(?:problem solving|analytical|critical thinking)\b',
                r'\b(?:creativity|innovation|adaptability)\b',
                r'\b(?:time management|organization|planning)\b',
                r'\b(?:customer service|client relations|stakeholder)\b'
            ],
            'cognitive_skills': [
                r'\b(?:reasoning|logic|analytical thinking|problem solving)\b',
                r'\b(?:attention to detail|accuracy|precision)\b',
                r'\b(?:memory|learning|comprehension)\b',
                r'\b(?:decision making|judgment|evaluation)\b'
            ]
        }
    
    def _load_job_role_patterns(self) -> Dict[str, List[str]]:
        """Load job role patterns for fallback processing."""
        return {
            'software_engineer': [
                r'\b(?:software engineer|developer|programmer|coder)\b',
                r'\b(?:frontend|backend|fullstack|full stack)\b',
                r'\b(?:web developer|mobile developer|app developer)\b'
            ],
            'data_scientist': [
                r'\b(?:data scientist|data analyst|data engineer)\b',
                r'\b(?:machine learning engineer|ml engineer|ai engineer)\b',
                r'\b(?:business analyst|research analyst)\b'
            ],
            'manager': [
                r'\b(?:manager|director|lead|supervisor|head)\b',
                r'\b(?:team lead|project manager|product manager)\b',
                r'\b(?:executive|ceo|cto|cfo|vp)\b'
            ],
            'sales': [
                r'\b(?:sales|account manager|business development)\b',
                r'\b(?:sales representative|account executive)\b'
            ],
            'marketing': [
                r'\b(?:marketing|digital marketing|content marketing)\b',
                r'\b(?:marketing manager|brand manager|growth)\b'
            ],
            'hr': [
                r'\b(?:human resources|hr|recruiter|talent)\b',
                r'\b(?:hr manager|people operations|talent acquisition)\b'
            ]
        }
    
    def process_query(self, query_text: str) -> ProcessedQuery:
        """
        Process a user query to extract skills, job role, and generate embedding.
        
        Args:
            query_text: The user's query or job description
            
        Returns:
            ProcessedQuery object with extracted information
        """
        logger.info(f"Processing query: {query_text[:100]}...")
        
        # Clean the input text
        cleaned_text = self._clean_text(query_text)
        
        # Try LLM-based processing first
        if self.gemini_model:
            try:
                processed_query = self._process_with_llm(query_text, cleaned_text)
                processed_query.processing_method = "llm"
                logger.info("Query processed using LLM")
            except Exception as e:
                logger.warning(f"LLM processing failed: {str(e)}, falling back to rule-based")
                processed_query = self._process_with_fallback(query_text, cleaned_text)
                processed_query.processing_method = "fallback"
        else:
            processed_query = self._process_with_fallback(query_text, cleaned_text)
            processed_query.processing_method = "fallback"
        
        # Generate embedding
        processed_query.embedding = self._generate_query_embedding(processed_query.cleaned_text)
        
        logger.info(f"Query processed successfully using {processed_query.processing_method} method")
        return processed_query
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep important punctuation
        cleaned = re.sub(r'[^\w\s\-\+\#\.\,\;\:]', ' ', cleaned)
        
        # Convert to lowercase for processing
        cleaned = cleaned.lower()
        
        return cleaned
    
    def _process_with_llm(self, original_text: str, cleaned_text: str) -> ProcessedQuery:
        """Process query using Gemini LLM for enhanced understanding."""
        
        # Create a structured prompt for the LLM
        prompt = self._create_llm_prompt(original_text)
        
        try:
            if GENAI_PACKAGE == "google.genai":
                # New package API
                response = self.gemini_model.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=prompt
                )
                response_text = response.text
            else:
                # Legacy package API
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text
            
            # Parse the LLM response
            extracted_info = self._parse_llm_response(response_text)
            
            return ProcessedQuery(
                original_text=original_text,
                cleaned_text=cleaned_text,
                extracted_skills=extracted_info.get('skills', []),
                job_role=extracted_info.get('job_role', 'unknown'),
                job_level=extracted_info.get('job_level', 'unknown'),
                required_domains=extracted_info.get('domains', []),
                confidence_score=extracted_info.get('confidence', 0.8)
            )
            
        except Exception as e:
            logger.error(f"LLM processing error: {str(e)}")
            raise
    
    def _create_llm_prompt(self, query_text: str) -> str:
        """Create a structured prompt for the LLM to extract information."""
        prompt = f"""
        Analyze the following job description or query and extract structured information:

        Query: "{query_text}"

        Please extract and return the following information in JSON format:
        {{
            "job_role": "primary job role/title",
            "job_level": "entry/mid/senior/executive",
            "skills": ["list", "of", "required", "skills"],
            "domains": ["technical", "behavioral", "cognitive"],
            "confidence": 0.0-1.0
        }}

        Guidelines:
        - job_role: Identify the main job role or position
        - job_level: Determine experience level (entry, mid, senior, executive)
        - skills: Extract both technical and soft skills mentioned
        - domains: Identify which assessment domains are needed:
          * "technical" for programming, technical knowledge, job-specific skills
          * "behavioral" for personality, teamwork, leadership, communication
          * "cognitive" for reasoning, problem-solving, analytical thinking
        - confidence: Your confidence in the extraction (0.0-1.0)

        Return only the JSON object, no additional text.
        """
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response to extract structured information."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                extracted_info = json.loads(json_str)
                
                # Validate and clean the extracted information
                return self._validate_llm_extraction(extracted_info)
            else:
                logger.warning("No JSON found in LLM response")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {str(e)}")
            return {}
    
    def _validate_llm_extraction(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean LLM extracted information."""
        validated = {}
        
        # Validate job_role
        validated['job_role'] = str(extracted_info.get('job_role', 'unknown')).lower()
        
        # Validate job_level
        valid_levels = ['entry', 'mid', 'senior', 'executive']
        job_level = str(extracted_info.get('job_level', 'unknown')).lower()
        validated['job_level'] = job_level if job_level in valid_levels else 'unknown'
        
        # Validate skills
        skills = extracted_info.get('skills', [])
        if isinstance(skills, list):
            validated['skills'] = [str(skill).lower().strip() for skill in skills if skill]
        else:
            validated['skills'] = []
        
        # Validate domains
        valid_domains = ['technical', 'behavioral', 'cognitive']
        domains = extracted_info.get('domains', [])
        if isinstance(domains, list):
            validated['domains'] = [d for d in domains if d in valid_domains]
        else:
            validated['domains'] = []
        
        # Validate confidence
        confidence = extracted_info.get('confidence', 0.5)
        try:
            validated['confidence'] = max(0.0, min(1.0, float(confidence)))
        except (ValueError, TypeError):
            validated['confidence'] = 0.5
        
        return validated
    
    def _process_with_fallback(self, original_text: str, cleaned_text: str) -> ProcessedQuery:
        """Process query using rule-based fallback methods."""
        
        # Extract skills using pattern matching
        extracted_skills = self._extract_skills_fallback(cleaned_text)
        
        # Extract job role using pattern matching
        job_role = self._extract_job_role_fallback(cleaned_text)
        
        # Determine job level
        job_level = self._extract_job_level_fallback(cleaned_text)
        
        # Determine required domains based on extracted skills
        required_domains = self._determine_domains_fallback(extracted_skills, cleaned_text)
        
        return ProcessedQuery(
            original_text=original_text,
            cleaned_text=cleaned_text,
            extracted_skills=extracted_skills,
            job_role=job_role,
            job_level=job_level,
            required_domains=required_domains,
            confidence_score=0.6  # Lower confidence for rule-based extraction
        )
    
    def _extract_skills_fallback(self, text: str) -> List[str]:
        """Extract skills using pattern matching."""
        skills = []
        
        for category, patterns in self._skill_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                skills.extend(matches)
        
        # Remove duplicates and clean
        unique_skills = list(set([skill.lower().strip() for skill in skills if skill]))
        return unique_skills
    
    def _extract_job_role_fallback(self, text: str) -> str:
        """Extract job role using pattern matching."""
        for role, patterns in self._job_role_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return role
        
        return 'unknown'
    
    def _extract_job_level_fallback(self, text: str) -> str:
        """Extract job level using pattern matching."""
        level_patterns = {
            'entry': [r'\b(?:entry|junior|intern|graduate|trainee)\b'],
            'mid': [r'\b(?:mid|middle|intermediate|experienced)\b'],
            'senior': [r'\b(?:senior|lead|principal|expert)\b'],
            'executive': [r'\b(?:director|manager|executive|head|vp|ceo|cto|cfo)\b']
        }
        
        for level, patterns in level_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return level
        
        return 'unknown'
    
    def _determine_domains_fallback(self, skills: List[str], text: str) -> List[str]:
        """Determine required assessment domains based on skills and text."""
        domains = []
        
        # Check for technical domain
        technical_indicators = [
            'programming', 'coding', 'software', 'technical', 'engineering',
            'development', 'database', 'api', 'cloud', 'testing'
        ]
        if any(indicator in text for indicator in technical_indicators) or \
           any(skill in self._skill_patterns['technical_skills'][0] for skill in skills):
            domains.append('technical')
        
        # Check for behavioral domain
        behavioral_indicators = [
            'leadership', 'management', 'communication', 'teamwork', 'collaboration',
            'customer service', 'interpersonal', 'personality'
        ]
        if any(indicator in text for indicator in behavioral_indicators) or \
           any(skill in ' '.join(self._skill_patterns['soft_skills']) for skill in skills):
            domains.append('behavioral')
        
        # Check for cognitive domain
        cognitive_indicators = [
            'problem solving', 'analytical', 'reasoning', 'critical thinking',
            'decision making', 'attention to detail', 'logic'
        ]
        if any(indicator in text for indicator in cognitive_indicators) or \
           any(skill in ' '.join(self._skill_patterns['cognitive_skills']) for skill in skills):
            domains.append('cognitive')
        
        # Default to all domains if none detected
        if not domains:
            domains = ['technical', 'behavioral', 'cognitive']
        
        return domains
    
    def _generate_query_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for the query text."""
        if not self.embedding_model:
            self._load_embedding_model()
        
        try:
            embedding = self.embedding_model.encode([text], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise
    
    def batch_process_queries(self, queries: List[str]) -> List[ProcessedQuery]:
        """Process multiple queries in batch for efficiency."""
        logger.info(f"Batch processing {len(queries)} queries")
        
        processed_queries = []
        for i, query in enumerate(queries):
            try:
                processed_query = self.process_query(query)
                processed_queries.append(processed_query)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(queries)} queries")
                    
            except Exception as e:
                logger.error(f"Failed to process query {i}: {str(e)}")
                # Create a minimal processed query for failed cases
                processed_queries.append(ProcessedQuery(
                    original_text=query,
                    cleaned_text=self._clean_text(query),
                    extracted_skills=[],
                    job_role='unknown',
                    job_level='unknown',
                    required_domains=['technical', 'behavioral', 'cognitive'],
                    confidence_score=0.0,
                    processing_method='error'
                ))
        
        logger.info(f"Batch processing completed: {len(processed_queries)} queries processed")
        return processed_queries
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about the query processor configuration."""
        return {
            'embedding_model': self.embedding_model_name,
            'llm_available': self.gemini_model is not None,
            'embedding_model_loaded': self.embedding_model is not None,
            'fallback_patterns': {
                'skill_categories': list(self._skill_patterns.keys()),
                'job_role_categories': list(self._job_role_patterns.keys())
            }
        }