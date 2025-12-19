"""
Vector database module for efficient storage and similarity search of assessment embeddings.

This module provides the VectorDatabase class that uses FAISS for fast similarity search
and manages assessment metadata for the recommendation system.
"""

import logging
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import faiss
from dataclasses import dataclass, asdict
from .assessment_processor import ProcessedAssessment

logger = logging.getLogger(__name__)


@dataclass
class AssessmentVector:
    """Assessment vector with metadata for storage in vector database."""
    id: str
    name: str
    url: str
    category: str
    test_type: str  # K (Knowledge & Skills) or P (Personality & Behavior)
    embedding: np.ndarray
    metadata: Dict[str, Any]


class VectorDatabase:
    """
    Vector database for storing and querying assessment embeddings using FAISS.
    
    This class provides efficient similarity search capabilities for assessment
    recommendations using cosine similarity and FAISS indexing.
    """
    
    def __init__(self, embedding_dim: int = 384, index_type: str = 'flat', 
                 storage_dir: str = './data/vector_db'):
        """
        Initialize the vector database with memory optimization.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            storage_dir: Directory to store the database files
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.storage_dir = storage_dir
        
        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize FAISS index with memory optimization
        self.index = None
        self.assessment_metadata = []  # List of AssessmentVector objects
        self.id_to_index = {}  # Map assessment ID to index position
        
        # Memory optimization: Use lazy loading
        self._index_initialized = False
        
    def _initialize_index(self) -> None:
        """Initialize the FAISS index based on the specified type with memory optimization."""
        if self._index_initialized:
            return
            
        try:
            if self.index_type == 'flat':
                # Flat index for exact search (good for smaller datasets)
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
                
            elif self.index_type == 'ivf':
                # IVF index for approximate search (good for larger datasets)
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                # Reduce nlist for memory efficiency
                nlist = min(50, max(10, len(self.assessment_metadata) // 10))  # Adaptive nlist
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                
            elif self.index_type == 'hnsw':
                # HNSW index for very fast approximate search
                M = 8  # Reduced connections for memory efficiency
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
                
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
                
            self._index_initialized = True
            logger.info(f"Initialized FAISS index: {self.index_type} with dimension {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {str(e)}")
            raise
    
    def add_assessments(self, assessments: List[ProcessedAssessment], 
                       embeddings: np.ndarray) -> None:
        """
        Add assessments and their embeddings to the vector database with memory optimization.
        
        Args:
            assessments: List of ProcessedAssessment objects
            embeddings: NumPy array of embeddings with shape (n_assessments, embedding_dim)
        """
        if len(assessments) != len(embeddings):
            raise ValueError(f"Number of assessments ({len(assessments)}) must match "
                           f"number of embeddings ({len(embeddings)})")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) must match "
                           f"index dimension ({self.embedding_dim})")
        
        try:
            # Initialize index lazily
            if not self._index_initialized:
                self._initialize_index()
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = self._normalize_embeddings(embeddings)
            
            # Add to FAISS index
            start_idx = len(self.assessment_metadata)
            
            # Train index if needed (for IVF)
            if self.index_type == 'ivf' and not self.index.is_trained:
                logger.info("Training IVF index...")
                self.index.train(normalized_embeddings)
            
            self.index.add(normalized_embeddings)
            
            # Store metadata with memory optimization - store only essential data
            for i, assessment in enumerate(assessments):
                # Use float32 instead of float64 for embeddings to save memory
                embedding_f32 = embeddings[i].astype(np.float32)
                
                assessment_vector = AssessmentVector(
                    id=assessment.id,
                    name=assessment.name,
                    url=assessment.url,
                    category=assessment.category,
                    test_type=assessment.test_type,
                    embedding=embedding_f32,  # Use float32
                    metadata={
                        'description': assessment.description[:500],  # Truncate long descriptions
                        'skills': assessment.skills[:10],  # Limit skills list
                        'quality_score': assessment.quality_score,
                        'token_count': assessment.token_count
                        # Remove 'cleaned_text' to save memory
                    }
                )
                
                self.assessment_metadata.append(assessment_vector)
                self.id_to_index[assessment.id] = start_idx + i
            
            logger.info(f"Added {len(assessments)} assessments to vector database. "
                       f"Total assessments: {len(self.assessment_metadata)}")
            
        except Exception as e:
            logger.error(f"Failed to add assessments to vector database: {str(e)}")
            raise
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10, 
                      filter_test_type: Optional[str] = None) -> List[Tuple[AssessmentVector, float]]:
        """
        Search for similar assessments using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar assessments to return
            filter_test_type: Optional filter by test type ('K' or 'P')
            
        Returns:
            List of tuples (AssessmentVector, similarity_score) sorted by similarity
        """
        if self.index.ntotal == 0:
            logger.warning("Vector database is empty")
            return []
        
        try:
            # Normalize query embedding
            normalized_query = self._normalize_embeddings(query_embedding.reshape(1, -1))
            
            # Search in FAISS index
            search_k = min(k * 3, self.index.ntotal)  # Search more to allow for filtering
            similarities, indices = self.index.search(normalized_query, search_k)
            
            # Convert to results with metadata
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                    
                assessment_vector = self.assessment_metadata[idx]
                
                # Apply test type filter if specified
                if filter_test_type and assessment_vector.test_type != filter_test_type:
                    continue
                
                results.append((assessment_vector, float(similarity)))
                
                # Stop when we have enough results
                if len(results) >= k:
                    break
            
            logger.debug(f"Found {len(results)} similar assessments for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar assessments: {str(e)}")
            raise
    
    def search_by_id(self, assessment_id: str) -> Optional[AssessmentVector]:
        """
        Retrieve an assessment by its ID.
        
        Args:
            assessment_id: ID of the assessment to retrieve
            
        Returns:
            AssessmentVector if found, None otherwise
        """
        if assessment_id in self.id_to_index:
            idx = self.id_to_index[assessment_id]
            return self.assessment_metadata[idx]
        return None
    
    def get_all_assessments(self) -> List[AssessmentVector]:
        """
        Get all assessments in the database.
        
        Returns:
            List of all AssessmentVector objects
        """
        return self.assessment_metadata.copy()
    
    def get_assessments_by_type(self, test_type: str) -> List[AssessmentVector]:
        """
        Get all assessments of a specific test type.
        
        Args:
            test_type: Test type to filter by ('K' or 'P')
            
        Returns:
            List of AssessmentVector objects matching the test type
        """
        return [av for av in self.assessment_metadata if av.test_type == test_type]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.assessment_metadata:
            return {
                'total_assessments': 0,
                'index_type': self.index_type,
                'embedding_dim': self.embedding_dim
            }
        
        # Count by test type
        type_counts = {}
        category_counts = {}
        
        for av in self.assessment_metadata:
            type_counts[av.test_type] = type_counts.get(av.test_type, 0) + 1
            category_counts[av.category] = category_counts.get(av.category, 0) + 1
        
        return {
            'total_assessments': len(self.assessment_metadata),
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'test_type_distribution': type_counts,
            'category_distribution': category_counts,
            'index_trained': getattr(self.index, 'is_trained', True),
            'index_total': self.index.ntotal
        }
    
    def save_database(self, filename: str = 'vector_database.pkl') -> str:
        """
        Save the vector database to disk.
        
        Args:
            filename: Name of the file to save
            
        Returns:
            Path to the saved file
        """
        filepath = os.path.join(self.storage_dir, filename)
        
        try:
            # Save FAISS index
            index_path = filepath.replace('.pkl', '.faiss')
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            database_data = {
                'assessment_metadata': self.assessment_metadata,
                'id_to_index': self.id_to_index,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'index_path': index_path,
                'stats': self.get_database_stats()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(database_data, f)
            
            logger.info(f"Vector database saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save vector database: {str(e)}")
            raise
    
    def load_database(self, filename: str = 'vector_database.pkl') -> None:
        """
        Load the vector database from disk.
        
        Args:
            filename: Name of the file to load
        """
        filepath = os.path.join(self.storage_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Database file not found: {filepath}")
        
        try:
            # Load metadata
            with open(filepath, 'rb') as f:
                database_data = pickle.load(f)
            
            # Load FAISS index
            index_path = database_data['index_path']
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index file not found: {index_path}")
            
            self.index = faiss.read_index(index_path)
            
            # Restore metadata
            self.assessment_metadata = database_data['assessment_metadata']
            self.id_to_index = database_data['id_to_index']
            self.embedding_dim = database_data['embedding_dim']
            self.index_type = database_data['index_type']
            
            logger.info(f"Vector database loaded from: {filepath}")
            logger.info(f"Loaded {len(self.assessment_metadata)} assessments")
            
        except Exception as e:
            logger.error(f"Failed to load vector database: {str(e)}")
            raise
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def rebuild_index(self) -> None:
        """
        Rebuild the FAISS index from stored embeddings.
        
        Useful when changing index type or parameters.
        """
        if not self.assessment_metadata:
            logger.warning("No assessments to rebuild index from")
            return
        
        try:
            # Extract embeddings
            embeddings = np.array([av.embedding for av in self.assessment_metadata])
            
            # Reinitialize index
            self._initialize_index()
            
            # Normalize and add embeddings
            normalized_embeddings = self._normalize_embeddings(embeddings)
            
            # Train if needed
            if self.index_type == 'ivf':
                logger.info("Training IVF index...")
                self.index.train(normalized_embeddings)
            
            self.index.add(normalized_embeddings)
            
            logger.info(f"Rebuilt index with {len(self.assessment_metadata)} assessments")
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {str(e)}")
            raise
    
    def add_assessment(self, assessment_id: str, name: str, description: str, 
                      url: str, test_type: str, category: str, skills: List[str]) -> None:
        """
        Add a single assessment directly to the database (memory optimized).
        
        Args:
            assessment_id: Unique identifier for the assessment
            name: Assessment name
            description: Assessment description
            url: Assessment URL
            test_type: Test type ('K' or 'P')
            category: Assessment category
            skills: List of skills
        """
        try:
            # Create a simple embedding from text (fallback method)
            text_for_embedding = f"{name} {description} {' '.join(skills)}"
            
            # Simple text-based embedding (fallback if sentence transformer not available)
            # This is much more memory efficient
            simple_embedding = self._create_simple_embedding(text_for_embedding)
            
            # Initialize index if needed
            if not self._index_initialized:
                self._initialize_index()
            
            # Normalize and add to index
            normalized_embedding = self._normalize_embeddings(simple_embedding.reshape(1, -1))
            self.index.add(normalized_embedding)
            
            # Create assessment vector
            assessment_vector = AssessmentVector(
                id=assessment_id,
                name=name,
                url=url,
                category=category,
                test_type=test_type,
                embedding=simple_embedding.astype(np.float32),
                metadata={
                    'description': description[:300],  # Truncate for memory
                    'skills': skills[:8],  # Limit skills
                    'quality_score': 0.8,  # Default score
                    'token_count': len(text_for_embedding.split())
                }
            )
            
            # Add to metadata
            idx = len(self.assessment_metadata)
            self.assessment_metadata.append(assessment_vector)
            self.id_to_index[assessment_id] = idx
            
            logger.debug(f"Added assessment: {name}")
            
        except Exception as e:
            logger.error(f"Failed to add assessment {assessment_id}: {str(e)}")
            raise
    
    def _create_simple_embedding(self, text: str) -> np.ndarray:
        """Create a simple embedding using basic text features (memory efficient fallback)."""
        # This is a very basic embedding method that doesn't require ML models
        # It's not as good as sentence transformers but uses minimal memory
        
        words = text.lower().split()
        
        # Create a simple feature vector based on word presence and frequency
        # This creates a 384-dimensional vector to match sentence transformer output
        embedding = np.zeros(384, dtype=np.float32)
        
        # Simple hash-based features
        for i, word in enumerate(words[:100]):  # Limit to first 100 words
            # Use hash to map words to dimensions
            hash_val = hash(word) % 384
            embedding[hash_val] += 1.0 / (i + 1)  # Weight by position
        
        # Add some basic text statistics
        embedding[0] = len(words)  # Word count
        embedding[1] = len(set(words))  # Unique word count
        embedding[2] = sum(len(word) for word in words) / max(len(words), 1)  # Avg word length
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding

    def clear_database(self) -> None:
        """Clear all data from the vector database."""
        if not self._index_initialized:
            self._initialize_index()
        else:
            self._initialize_index()  # Reinitialize
        self.assessment_metadata.clear()
        self.id_to_index.clear()
        logger.info("Vector database cleared")
    
    def validate_database(self) -> Dict[str, Any]:
        """
        Validate the consistency of the vector database.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check index consistency
        if self.index.ntotal != len(self.assessment_metadata):
            validation_results['is_valid'] = False
            validation_results['errors'].append(
                f"Index size ({self.index.ntotal}) doesn't match metadata size ({len(self.assessment_metadata)})"
            )
        
        # Check ID mapping consistency
        if len(self.id_to_index) != len(self.assessment_metadata):
            validation_results['is_valid'] = False
            validation_results['errors'].append(
                f"ID mapping size ({len(self.id_to_index)}) doesn't match metadata size ({len(self.assessment_metadata)})"
            )
        
        # Check for duplicate IDs
        ids = [av.id for av in self.assessment_metadata]
        if len(set(ids)) != len(ids):
            validation_results['is_valid'] = False
            validation_results['errors'].append("Duplicate assessment IDs found")
        
        # Check embedding dimensions
        for i, av in enumerate(self.assessment_metadata):
            if av.embedding.shape[0] != self.embedding_dim:
                validation_results['is_valid'] = False
                validation_results['errors'].append(
                    f"Assessment {av.id} has wrong embedding dimension: {av.embedding.shape[0]} != {self.embedding_dim}"
                )
                break  # Don't check all if we find one error
        
        return validation_results