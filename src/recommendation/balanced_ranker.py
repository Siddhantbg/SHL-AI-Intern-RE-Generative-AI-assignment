"""
Balanced ranking module for ensuring balanced recommendations across assessment domains.

This module provides the BalancedRanker class that ensures recommendations include
a balanced mix of Knowledge & Skills (Type K) and Personality & Behavior (Type P) assessments.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from ..processing.vector_database import AssessmentVector
from .query_processor import ProcessedQuery

logger = logging.getLogger(__name__)


@dataclass
class BalanceConfig:
    """Configuration for balanced ranking."""
    min_technical_ratio: float = 0.3  # Minimum ratio of technical assessments
    max_technical_ratio: float = 0.8  # Maximum ratio of technical assessments
    min_behavioral_ratio: float = 0.2  # Minimum ratio of behavioral assessments
    max_behavioral_ratio: float = 0.7  # Maximum ratio of behavioral assessments
    prefer_balance: bool = True  # Whether to prefer balanced results
    quality_threshold: float = 0.3  # Minimum quality score for inclusion


class BalancedRanker:
    """
    Ensures balanced recommendations across Knowledge & Skills (K) and Personality & Behavior (P) assessments.
    
    This class implements business logic for multi-domain queries to provide balanced
    coverage of both technical and behavioral assessments based on query requirements.
    """
    
    def __init__(self, balance_config: Optional[BalanceConfig] = None):
        """
        Initialize the balanced ranker.
        
        Args:
            balance_config: Configuration for balancing behavior
        """
        self.config = balance_config or BalanceConfig()
        self.last_balance_info = {}
        
        logger.info("BalancedRanker initialized")
    
    def balance_recommendations(self, candidates: List[Tuple[AssessmentVector, float]], 
                              processed_query: ProcessedQuery, 
                              max_results: int) -> List[Tuple[AssessmentVector, float]]:
        """
        Balance recommendations to ensure appropriate mix of assessment types.
        
        Args:
            candidates: List of candidate assessments with similarity scores
            processed_query: Processed query information
            max_results: Maximum number of results to return
            
        Returns:
            Balanced list of assessments with scores
        """
        if max_results <= 0:
            self.last_balance_info = {"balanced": False, "reason": "invalid_max_results"}
            return []
        
        if not candidates:
            self.last_balance_info = {"balanced": False, "reason": "no_candidates"}
            return []
        
        logger.debug(f"Balancing {len(candidates)} candidates for {max_results} results")
        
        # Separate candidates by test type
        k_type_candidates = [(a, s) for a, s in candidates if a.test_type == 'K']
        p_type_candidates = [(a, s) for a, s in candidates if a.test_type == 'P']
        
        logger.debug(f"Candidates by type: K={len(k_type_candidates)}, P={len(p_type_candidates)}")
        
        # Determine balancing strategy based on query requirements
        balance_strategy = self._determine_balance_strategy(processed_query, k_type_candidates, p_type_candidates)
        
        # Apply balancing strategy
        balanced_results = self._apply_balance_strategy(
            balance_strategy, k_type_candidates, p_type_candidates, max_results
        )
        
        # Store balance information for reporting
        self._update_balance_info(balance_strategy, balanced_results, k_type_candidates, p_type_candidates)
        
        logger.debug(f"Balanced results: {len(balanced_results)} assessments")
        return balanced_results
    
    def _determine_balance_strategy(self, processed_query: ProcessedQuery, 
                                  k_candidates: List[Tuple[AssessmentVector, float]], 
                                  p_candidates: List[Tuple[AssessmentVector, float]]) -> Dict[str, Any]:
        """
        Determine the appropriate balancing strategy based on query requirements.
        
        Args:
            processed_query: Processed query information
            k_candidates: Knowledge & Skills candidates
            p_candidates: Personality & Behavior candidates
            
        Returns:
            Dictionary with balancing strategy
        """
        strategy = {
            'type': 'balanced',
            'k_ratio': 0.5,
            'p_ratio': 0.5,
            'reasoning': []
        }
        
        required_domains = processed_query.required_domains
        
        # Analyze query requirements
        has_technical = any(domain in required_domains for domain in ['technical', 'cognitive'])
        has_behavioral = 'behavioral' in required_domains
        
        # Determine strategy based on domain requirements
        if has_technical and has_behavioral:
            # Multi-domain query - use balanced approach
            strategy['type'] = 'multi_domain'
            strategy['k_ratio'] = 0.6  # Slightly favor technical for multi-domain
            strategy['p_ratio'] = 0.4
            strategy['reasoning'].append("Multi-domain query detected")
            
        elif has_technical and not has_behavioral:
            # Technical-focused query
            strategy['type'] = 'technical_focused'
            strategy['k_ratio'] = 0.8
            strategy['p_ratio'] = 0.2
            strategy['reasoning'].append("Technical-focused query")
            
        elif has_behavioral and not has_technical:
            # Behavioral-focused query
            strategy['type'] = 'behavioral_focused'
            strategy['k_ratio'] = 0.2
            strategy['p_ratio'] = 0.8
            strategy['reasoning'].append("Behavioral-focused query")
            
        else:
            # No clear domain preference - use job role heuristics
            strategy = self._infer_strategy_from_job_role(processed_query.job_role)
        
        # Adjust based on candidate availability (only if we have some candidates)
        if k_candidates or p_candidates:
            if not k_candidates:
                strategy['k_ratio'] = 0.0
                strategy['p_ratio'] = 1.0
                strategy['reasoning'].append("No K-type candidates available")
            elif not p_candidates:
                strategy['k_ratio'] = 1.0
                strategy['p_ratio'] = 0.0
                strategy['reasoning'].append("No P-type candidates available")
        
        # Apply configuration constraints only if we have candidates
        if k_candidates or p_candidates:
            strategy['k_ratio'] = max(self.config.min_technical_ratio, 
                                    min(self.config.max_technical_ratio, strategy['k_ratio']))
            strategy['p_ratio'] = max(self.config.min_behavioral_ratio, 
                                    min(self.config.max_behavioral_ratio, strategy['p_ratio']))
        
        # Normalize ratios
        total_ratio = strategy['k_ratio'] + strategy['p_ratio']
        if total_ratio > 0:
            strategy['k_ratio'] /= total_ratio
            strategy['p_ratio'] /= total_ratio
        
        return strategy
    
    def _infer_strategy_from_job_role(self, job_role: str) -> Dict[str, Any]:
        """
        Infer balancing strategy from job role when domain requirements are unclear.
        
        Args:
            job_role: Job role from processed query
            
        Returns:
            Dictionary with inferred strategy
        """
        # Job role to strategy mapping
        role_strategies = {
            'software_engineer': {'k_ratio': 0.7, 'p_ratio': 0.3, 'type': 'technical_heavy'},
            'data_scientist': {'k_ratio': 0.7, 'p_ratio': 0.3, 'type': 'technical_heavy'},
            'manager': {'k_ratio': 0.3, 'p_ratio': 0.7, 'type': 'behavioral_heavy'},
            'sales': {'k_ratio': 0.2, 'p_ratio': 0.8, 'type': 'behavioral_heavy'},
            'marketing': {'k_ratio': 0.4, 'p_ratio': 0.6, 'type': 'behavioral_focused'},
            'hr': {'k_ratio': 0.3, 'p_ratio': 0.7, 'type': 'behavioral_heavy'}
        }
        
        if job_role in role_strategies:
            strategy = role_strategies[job_role].copy()
            strategy['reasoning'] = [f"Inferred from job role: {job_role}"]
            return strategy
        
        # Default balanced strategy for unknown roles
        return {
            'type': 'balanced_default',
            'k_ratio': 0.5,
            'p_ratio': 0.5,
            'reasoning': ['Default balanced strategy for unknown role']
        }
    
    def _apply_balance_strategy(self, strategy: Dict[str, Any], 
                              k_candidates: List[Tuple[AssessmentVector, float]], 
                              p_candidates: List[Tuple[AssessmentVector, float]], 
                              max_results: int) -> List[Tuple[AssessmentVector, float]]:
        """
        Apply the determined balancing strategy to select final recommendations.
        
        Args:
            strategy: Balancing strategy dictionary
            k_candidates: Knowledge & Skills candidates
            p_candidates: Personality & Behavior candidates
            max_results: Maximum number of results
            
        Returns:
            Balanced list of recommendations
        """
        # Calculate target counts for each type
        k_target = int(max_results * strategy['k_ratio'])
        p_target = int(max_results * strategy['p_ratio'])
        
        # Ensure we don't exceed available candidates
        k_available = len(k_candidates)
        p_available = len(p_candidates)
        
        k_count = min(k_target, k_available)
        p_count = min(p_target, p_available)
        
        # If we have remaining slots, distribute them proportionally
        remaining_slots = max_results - k_count - p_count
        if remaining_slots > 0:
            if k_available > k_count and p_available > p_count:
                # Distribute remaining slots based on strategy ratios
                additional_k = int(remaining_slots * strategy['k_ratio'])
                additional_p = remaining_slots - additional_k
                
                k_count = min(k_count + additional_k, k_available)
                p_count = min(p_count + additional_p, p_available)
            elif k_available > k_count:
                # Only K-type candidates available
                k_count = min(k_count + remaining_slots, k_available)
            elif p_available > p_count:
                # Only P-type candidates available
                p_count = min(p_count + remaining_slots, p_available)
        
        # Select top candidates from each type
        selected_k = k_candidates[:k_count] if k_count > 0 else []
        selected_p = p_candidates[:p_count] if p_count > 0 else []
        
        # Combine and sort by similarity score
        combined_results = selected_k + selected_p
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results
    
    def _update_balance_info(self, strategy: Dict[str, Any], 
                           results: List[Tuple[AssessmentVector, float]], 
                           k_candidates: List[Tuple[AssessmentVector, float]], 
                           p_candidates: List[Tuple[AssessmentVector, float]]) -> None:
        """
        Update balance information for reporting and debugging.
        
        Args:
            strategy: Applied balancing strategy
            results: Final balanced results
            k_candidates: Available K-type candidates
            p_candidates: Available P-type candidates
        """
        # Count actual results by type
        k_results = sum(1 for assessment, _ in results if assessment.test_type == 'K')
        p_results = sum(1 for assessment, _ in results if assessment.test_type == 'P')
        total_results = len(results)
        
        # Calculate actual ratios
        actual_k_ratio = k_results / total_results if total_results > 0 else 0
        actual_p_ratio = p_results / total_results if total_results > 0 else 0
        
        self.last_balance_info = {
            'balanced': True,
            'strategy': strategy,
            'target_ratios': {
                'k_ratio': strategy['k_ratio'],
                'p_ratio': strategy['p_ratio']
            },
            'actual_ratios': {
                'k_ratio': actual_k_ratio,
                'p_ratio': actual_p_ratio
            },
            'counts': {
                'k_results': k_results,
                'p_results': p_results,
                'total_results': total_results,
                'k_candidates': len(k_candidates),
                'p_candidates': len(p_candidates)
            },
            'balance_quality': self._calculate_balance_quality(strategy, actual_k_ratio, actual_p_ratio)
        }
    
    def _calculate_balance_quality(self, strategy: Dict[str, Any], 
                                 actual_k_ratio: float, actual_p_ratio: float) -> Dict[str, Any]:
        """
        Calculate how well the actual results match the intended balance.
        
        Args:
            strategy: Intended balancing strategy
            actual_k_ratio: Actual ratio of K-type assessments
            actual_p_ratio: Actual ratio of P-type assessments
            
        Returns:
            Dictionary with balance quality metrics
        """
        target_k_ratio = strategy['k_ratio']
        target_p_ratio = strategy['p_ratio']
        
        # Calculate deviations
        k_deviation = abs(actual_k_ratio - target_k_ratio)
        p_deviation = abs(actual_p_ratio - target_p_ratio)
        
        # Calculate overall balance score (0-1, where 1 is perfect balance)
        max_deviation = max(k_deviation, p_deviation)
        balance_score = max(0, 1 - (max_deviation * 2))  # Scale deviation to 0-1
        
        # Determine balance quality level
        if balance_score >= 0.9:
            quality_level = 'excellent'
        elif balance_score >= 0.7:
            quality_level = 'good'
        elif balance_score >= 0.5:
            quality_level = 'fair'
        else:
            quality_level = 'poor'
        
        return {
            'score': balance_score,
            'level': quality_level,
            'k_deviation': k_deviation,
            'p_deviation': p_deviation,
            'max_deviation': max_deviation
        }
    
    def get_balance_info(self) -> Dict[str, Any]:
        """
        Get information about the last balancing operation.
        
        Returns:
            Dictionary with balance information
        """
        return self.last_balance_info.copy()
    
    def get_ranker_info(self) -> Dict[str, Any]:
        """
        Get information about the ranker configuration.
        
        Returns:
            Dictionary with ranker information
        """
        return {
            'config': {
                'min_technical_ratio': self.config.min_technical_ratio,
                'max_technical_ratio': self.config.max_technical_ratio,
                'min_behavioral_ratio': self.config.min_behavioral_ratio,
                'max_behavioral_ratio': self.config.max_behavioral_ratio,
                'prefer_balance': self.config.prefer_balance,
                'quality_threshold': self.config.quality_threshold
            },
            'last_operation': self.last_balance_info
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update ranker configuration.
        
        Args:
            new_config: Dictionary with new configuration values
        """
        if 'min_technical_ratio' in new_config:
            self.config.min_technical_ratio = max(0.0, min(1.0, new_config['min_technical_ratio']))
        
        if 'max_technical_ratio' in new_config:
            self.config.max_technical_ratio = max(0.0, min(1.0, new_config['max_technical_ratio']))
        
        if 'min_behavioral_ratio' in new_config:
            self.config.min_behavioral_ratio = max(0.0, min(1.0, new_config['min_behavioral_ratio']))
        
        if 'max_behavioral_ratio' in new_config:
            self.config.max_behavioral_ratio = max(0.0, min(1.0, new_config['max_behavioral_ratio']))
        
        if 'prefer_balance' in new_config:
            self.config.prefer_balance = bool(new_config['prefer_balance'])
        
        if 'quality_threshold' in new_config:
            self.config.quality_threshold = max(0.0, min(1.0, new_config['quality_threshold']))
        
        logger.info("BalancedRanker configuration updated")
    
    def analyze_query_balance_needs(self, processed_query: ProcessedQuery) -> Dict[str, Any]:
        """
        Analyze a query to determine its balance requirements.
        
        Args:
            processed_query: Processed query to analyze
            
        Returns:
            Dictionary with balance analysis
        """
        analysis = {
            'query_type': 'unknown',
            'recommended_k_ratio': 0.5,
            'recommended_p_ratio': 0.5,
            'confidence': 0.5,
            'reasoning': []
        }
        
        # Analyze required domains
        required_domains = processed_query.required_domains
        
        if 'technical' in required_domains and 'behavioral' in required_domains:
            analysis['query_type'] = 'multi_domain'
            analysis['recommended_k_ratio'] = 0.6
            analysis['recommended_p_ratio'] = 0.4
            analysis['confidence'] = 0.8
            analysis['reasoning'].append("Both technical and behavioral domains required")
            
        elif 'technical' in required_domains or 'cognitive' in required_domains:
            analysis['query_type'] = 'technical_focused'
            analysis['recommended_k_ratio'] = 0.75
            analysis['recommended_p_ratio'] = 0.25
            analysis['confidence'] = 0.7
            analysis['reasoning'].append("Technical/cognitive focus detected")
            
        elif 'behavioral' in required_domains:
            analysis['query_type'] = 'behavioral_focused'
            analysis['recommended_k_ratio'] = 0.25
            analysis['recommended_p_ratio'] = 0.75
            analysis['confidence'] = 0.7
            analysis['reasoning'].append("Behavioral focus detected")
        
        # Analyze job role
        job_role_analysis = self._analyze_job_role_balance_needs(processed_query.job_role)
        if job_role_analysis['confidence'] > analysis['confidence']:
            analysis.update(job_role_analysis)
        
        return analysis
    
    def _analyze_job_role_balance_needs(self, job_role: str) -> Dict[str, Any]:
        """
        Analyze balance needs based on job role.
        
        Args:
            job_role: Job role to analyze
            
        Returns:
            Dictionary with job role balance analysis
        """
        role_analysis = {
            'software_engineer': {
                'query_type': 'technical_role',
                'recommended_k_ratio': 0.7,
                'recommended_p_ratio': 0.3,
                'confidence': 0.8,
                'reasoning': ['Software engineering role requires strong technical assessment']
            },
            'manager': {
                'query_type': 'leadership_role',
                'recommended_k_ratio': 0.3,
                'recommended_p_ratio': 0.7,
                'confidence': 0.8,
                'reasoning': ['Management role requires strong behavioral assessment']
            },
            'sales': {
                'query_type': 'interpersonal_role',
                'recommended_k_ratio': 0.2,
                'recommended_p_ratio': 0.8,
                'confidence': 0.9,
                'reasoning': ['Sales role heavily emphasizes personality and behavior']
            }
        }
        
        if job_role in role_analysis:
            return role_analysis[job_role]
        
        return {
            'query_type': 'unknown_role',
            'recommended_k_ratio': 0.5,
            'recommended_p_ratio': 0.5,
            'confidence': 0.3,
            'reasoning': ['Unknown job role, using balanced approach']
        }