#!/usr/bin/env python3
"""
Process scraped assessment data and populate vector database.
"""

import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Process scraped data and populate vector database."""
    try:
        # Import after adding to path
        from src.scraper.shl_catalog_scraper import Assessment
        from src.processing.assessment_processor import AssessmentProcessor
        from src.processing.embedding_generator import EmbeddingGenerator
        from src.processing.vector_database import VectorDatabase
        from src.config import get_settings
        
        settings = get_settings()
        
        # Load scraped data
        scraped_file = Path("data/scraped/assessments.json")
        if not scraped_file.exists():
            logger.error(f"Scraped data file not found: {scraped_file}")
            return
        
        logger.info(f"Loading scraped data from: {scraped_file}")
        with open(scraped_file, 'r') as f:
            scraped_data = json.load(f)
        
        # Convert to Assessment objects
        assessments = []
        for item in scraped_data:
            assessment = Assessment(
                id=item['id'],
                name=item['name'],
                url=item['url'],
                category=item['category'],
                test_type=item['test_type'],
                description=item['description'],
                skills=item['skills']
            )
            assessments.append(assessment)
        
        logger.info(f"Loaded {len(assessments)} assessments")
        
        # Process assessments
        logger.info("Processing assessments...")
        processor = AssessmentProcessor()
        processed_assessments = processor.process_assessments(assessments)
        
        stats = processor.get_processing_stats(processed_assessments)
        logger.info(f"Processing complete: {stats['total_assessments']} assessments processed")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embedding_generator = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
        embedding_generator.load_model()
        
        embeddings = embedding_generator.generate_embeddings(processed_assessments)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Initialize vector database
        logger.info("Initializing vector database...")
        vector_db = VectorDatabase(
            embedding_dim=384,
            index_type='flat',
            storage_dir=str(settings.data_dir / 'vector_db')
        )
        
        # Add assessments to database
        logger.info("Adding assessments to vector database...")
        vector_db.add_assessments(processed_assessments, embeddings)
        
        # Save database
        logger.info("Saving vector database...")
        vector_db.save_database()
        
        # Get final stats
        db_stats = vector_db.get_database_stats()
        logger.info(f"Vector database populated with {db_stats['total_assessments']} assessments")
        
        logger.info("Data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()