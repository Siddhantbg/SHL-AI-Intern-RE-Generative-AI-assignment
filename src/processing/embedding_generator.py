"""
Embedding generation module for converting assessment descriptions into vector representations.

This module provides the EmbeddingGenerator class that uses sentence transformer models
to create vector embeddings from processed assessment data.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import pickle
import os
from sentence_transformers import SentenceTransformer
from .assessment_processor import ProcessedAssessment

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates vector embeddings for assessment descriptions using sentence transformers.
    
    This class handles the conversion of processed assessment text into dense vector
    representations suitable for similarity search and recommendation systems.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
            cache_dir: Directory to cache the model and embeddings
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or './models/embeddings'
        self.model = None
        self.embedding_dim = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode(["test"])
            self.embedding_dim = test_embedding.shape[1]
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    def generate_embeddings(self, assessments: List[ProcessedAssessment]) -> np.ndarray:
        """
        Generate embeddings for a list of processed assessments.
        
        Args:
            assessments: List of ProcessedAssessment objects
            
        Returns:
            NumPy array of embeddings with shape (n_assessments, embedding_dim)
        """
        if not self.model:
            self.load_model()
        
        if not assessments:
            return np.array([]).reshape(0, self.embedding_dim)
        
        logger.info(f"Generating embeddings for {len(assessments)} assessments")
        
        # Extract text for embedding
        texts = [assessment.cleaned_text for assessment in assessments]
        
        try:
            # Generate embeddings in batches for memory efficiency
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=True if i == 0 else False
                )
                embeddings.append(batch_embeddings)
                
                if i % (batch_size * 10) == 0:
                    logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} assessments")
            
            # Concatenate all embeddings
            all_embeddings = np.vstack(embeddings)
            
            logger.info(f"Generated embeddings with shape: {all_embeddings.shape}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            NumPy array with embedding vector
        """
        if not self.model:
            self.load_model()
        
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to generate single embedding: {str(e)}")
            raise
    
    def save_embeddings(self, embeddings: np.ndarray, assessments: List[ProcessedAssessment], 
                       filename: str = 'assessment_embeddings.pkl') -> str:
        """
        Save embeddings and associated assessment data to disk.
        
        Args:
            embeddings: NumPy array of embeddings
            assessments: List of ProcessedAssessment objects
            filename: Name of the file to save
            
        Returns:
            Path to the saved file
        """
        filepath = os.path.join(self.cache_dir, filename)
        
        # Create data structure for saving
        embedding_data = {
            'embeddings': embeddings,
            'assessments': assessments,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'metadata': {
                'num_assessments': len(assessments),
                'embedding_shape': embeddings.shape,
                'generated_at': pd.Timestamp.now().isoformat()
            }
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(embedding_data, f)
            
            logger.info(f"Embeddings saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, filename: str = 'assessment_embeddings.pkl') -> Tuple[np.ndarray, List[ProcessedAssessment]]:
        """
        Load embeddings and assessment data from disk.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Tuple of (embeddings array, list of assessments)
        """
        filepath = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                embedding_data = pickle.load(f)
            
            embeddings = embedding_data['embeddings']
            assessments = embedding_data['assessments']
            
            # Verify model compatibility
            if embedding_data.get('model_name') != self.model_name:
                logger.warning(f"Model mismatch: saved with {embedding_data.get('model_name')}, "
                             f"current model is {self.model_name}")
            
            logger.info(f"Loaded embeddings from: {filepath}")
            logger.info(f"Shape: {embeddings.shape}, Assessments: {len(assessments)}")
            
            return embeddings, assessments
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}")
            raise
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity matrix for embeddings.
        
        Args:
            embeddings: NumPy array of embeddings
            
        Returns:
            Similarity matrix with shape (n_assessments, n_assessments)
        """
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix
    
    def find_similar_assessments(self, query_embedding: np.ndarray, 
                                embeddings: np.ndarray, 
                                assessments: List[ProcessedAssessment],
                                top_k: int = 10) -> List[Tuple[ProcessedAssessment, float]]:
        """
        Find most similar assessments to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            embeddings: Assessment embeddings matrix
            assessments: List of ProcessedAssessment objects
            top_k: Number of top similar assessments to return
            
        Returns:
            List of tuples (assessment, similarity_score) sorted by similarity
        """
        # Normalize embeddings for cosine similarity
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(normalized_embeddings, normalized_query)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return assessments with similarity scores
        results = []
        for idx in top_indices:
            assessment = assessments[idx]
            similarity = similarities[idx]
            results.append((assessment, float(similarity)))
        
        return results
    
    def validate_embeddings(self, embeddings: np.ndarray, assessments: List[ProcessedAssessment]) -> Dict[str, Any]:
        """
        Validate generated embeddings for quality and consistency.
        
        Args:
            embeddings: NumPy array of embeddings
            assessments: List of ProcessedAssessment objects
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check shape consistency
        if len(embeddings) != len(assessments):
            validation_results['is_valid'] = False
            validation_results['errors'].append(
                f"Embedding count ({len(embeddings)}) doesn't match assessment count ({len(assessments)})"
            )
        
        # Check for NaN or infinite values
        if np.isnan(embeddings).any():
            validation_results['is_valid'] = False
            validation_results['errors'].append("Embeddings contain NaN values")
        
        if np.isinf(embeddings).any():
            validation_results['is_valid'] = False
            validation_results['errors'].append("Embeddings contain infinite values")
        
        # Check embedding dimension consistency
        if embeddings.ndim != 2:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Expected 2D embeddings, got {embeddings.ndim}D")
        elif self.embedding_dim is not None and embeddings.shape[1] != self.embedding_dim:
            validation_results['warnings'].append(
                f"Embedding dimension ({embeddings.shape[1]}) doesn't match expected ({self.embedding_dim})"
            )
        
        # Calculate statistics
        if validation_results['is_valid']:
            validation_results['stats'] = {
                'shape': embeddings.shape,
                'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
                'std_norm': np.std(np.linalg.norm(embeddings, axis=1)),
                'min_value': np.min(embeddings),
                'max_value': np.max(embeddings),
                'mean_value': np.mean(embeddings),
                'std_value': np.std(embeddings)
            }
        
        return validation_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {'model_name': self.model_name, 'loaded': False}
        
        return {
            'model_name': self.model_name,
            'loaded': True,
            'embedding_dim': self.embedding_dim,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown'),
            'cache_dir': self.cache_dir
        }