# Implementation Plan

- [-] 1. Set up project structure and development environment



  - Create Python project with proper directory structure (src/, tests/, docs/, etc.)
  - Set up virtual environment and install core dependencies (FastAPI, requests, beautifulsoup4, sentence-transformers, faiss-cpu, pytest)
  - Configure development tools (linting, formatting, type checking with mypy)
  - Create Docker configuration files (Dockerfile, docker-compose.yml)
  - Initialize Git repository with proper .gitignore
  - _Requirements: 1.5, 6.4_

- [ ] 2. Implement web scraping module for SHL catalog
  - Create `SHLCatalogScraper` class to navigate and extract data from SHL product catalog
  - Implement parsing logic to extract assessment name, URL, category, test type, and description
  - Add filtering to exclude "Pre-packaged Job Solutions" and focus on individual test solutions
  - Implement retry logic and error handling for network requests
  - Add validation to ensure minimum 377 assessments are collected
  - Create unit tests for scraper functionality with mock responses
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 3. Build data processing and embedding pipeline
  - Create `AssessmentProcessor` class to clean and normalize scraped assessment data
  - Implement text preprocessing (cleaning, tokenization, normalization)
  - Integrate sentence transformer model (e.g., all-MiniLM-L6-v2) for generating embeddings
  - Create `EmbeddingGenerator` to convert assessment descriptions into vector representations
  - Implement data validation and quality checks for processed assessments
  - Write unit tests for data processing pipeline with sample data
  - _Requirements: 1.5, 2.3_

- [ ] 4. Set up vector database and similarity search
  - Implement vector storage using FAISS for efficient similarity search
  - Create `VectorDatabase` class with methods for storing and querying embeddings
  - Implement cosine similarity search functionality
  - Add indexing and retrieval methods for assessment vectors
  - Create database schema for storing assessment metadata
  - Write integration tests for vector database operations
  - _Requirements: 2.3, 2.4_

- [ ] 5. Develop LLM-based query processing
  - Create `QueryProcessor` class to understand and extract information from user queries
  - Implement query embedding generation using the same sentence transformer model
  - Add skill extraction and job role identification from natural language queries
  - Integrate with LLM API (Gemini) for enhanced query understanding and context extraction
  - Implement fallback mechanisms for when LLM service is unavailable
  - Write unit tests for query processing with various input types
  - _Requirements: 2.1, 2.2_

- [ ] 6. Build core recommendation engine with balancing logic
  - Create `RecommendationEngine` class as the main recommendation orchestrator
  - Implement similarity-based ranking using vector search results
  - Create `BalancedRanker` to ensure mix of Knowledge & Skills (Type K) and Personality & Behavior (Type P) assessments
  - Add business logic for handling multi-domain queries (technical + behavioral skills)
  - Implement recommendation filtering and result limiting (1-10 assessments)
  - Write comprehensive unit tests for recommendation logic with various query scenarios
  - _Requirements: 2.4, 2.5_

- [ ] 7. Implement REST API with required endpoints
  - Create FastAPI application with proper project structure
  - Implement `/health` endpoint returning API status, uptime, and assessment count
  - Implement `/recommend` endpoint accepting queries and returning JSON recommendations
  - Add request/response models with proper validation using Pydantic
  - Implement error handling with appropriate HTTP status codes
  - Add API documentation with OpenAPI/Swagger integration
  - Write API integration tests for both endpoints with various input scenarios
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 8. Create evaluation module and metrics calculation
  - Create `EvaluationEngine` class to measure recommendation system performance
  - Implement Mean Recall@K calculation for measuring recommendation accuracy
  - Add functionality to load and process training dataset (10 labeled queries)
  - Create evaluation pipeline to test system performance against labeled data
  - Implement performance tracking and logging for optimization iterations
  - Write unit tests for evaluation metrics and ensure correct calculation
  - _Requirements: 3.1, 3.2, 3.4_

- [ ] 9. Build web frontend application
  - Set up React.js project with TypeScript and Tailwind CSS
  - Create main components: QueryInput, RecommendationTable, LoadingSpinner, ErrorBoundary
  - Implement user interface for entering job descriptions and queries
  - Add API integration to call the /recommend endpoint
  - Create responsive table display for showing assessment names and URLs
  - Implement error handling and loading states in the UI
  - Write component tests using React Testing Library
  - _Requirements: 5.1, 5.2, 5.3, 5.5_

- [ ] 10. Integrate training data evaluation and optimization
  - Load provided training dataset with 10 human-labeled queries
  - Implement evaluation loop to test current system performance
  - Create optimization pipeline for improving recommendation accuracy
  - Add hyperparameter tuning for embedding models and similarity thresholds
  - Implement prompt engineering for better LLM query understanding
  - Document performance improvements and optimization methodology
  - _Requirements: 3.2, 3.5_

- [ ] 11. Generate test set predictions and prepare submission files
  - Load unlabeled test dataset with 9 queries
  - Run recommendation system on test queries to generate predictions
  - Format predictions in required CSV format (Query, Assessment_url columns)
  - Validate CSV format matches exact specification for automated evaluation
  - Create submission preparation script to generate all required files
  - _Requirements: 3.3, 6.3_

- [ ] 12. Set up deployment and hosting infrastructure
  - Configure Docker containerization for both API and frontend
  - Set up Google Cloud Platform project with necessary services
  - Deploy API to Cloud Run with proper environment configuration
  - Deploy frontend to Cloud Storage with CDN or similar hosting service
  - Configure domain names and SSL certificates for public access
  - Implement health monitoring and logging for deployed services
  - _Requirements: 5.4, 6.1_

- [ ] 13. Create comprehensive documentation
  - Write 2-page approach document detailing solution methodology and performance optimization
  - Document API endpoints with examples and response formats
  - Create README with setup instructions, usage examples, and deployment guide
  - Add inline code documentation and type hints throughout codebase
  - Document evaluation results and performance metrics achieved
  - Create troubleshooting guide for common issues
  - _Requirements: 6.2, 6.5_

- [ ] 14. Perform final testing and quality assurance
  - Run complete end-to-end testing of the entire system
  - Verify all API endpoints are functional and return correct responses
  - Test web application functionality and user experience
  - Validate CSV predictions format and content
  - Perform load testing to ensure system handles concurrent requests
  - Conduct final code review and security audit
  - _Requirements: 6.1, 6.4, 6.5_

- [ ] 15. Prepare and submit final deliverables
  - Verify all three URLs are functional: API endpoint, GitHub repository, web application
  - Upload code to GitHub repository with complete implementation and experiments
  - Finalize approach documentation with performance optimization details
  - Submit CSV file with test set predictions in exact required format
  - Complete submission form with all required URLs and documents
  - Perform final verification that all submission requirements are met
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_