# SHL Assessment Recommendation System - Requirements Document

## Introduction

This project involves building an intelligent recommendation system that helps hiring managers and recruiters find the right SHL assessments for their roles. The system takes natural language queries or job descriptions and returns relevant SHL assessments from their product catalog. The solution must leverage modern LLM-based or retrieval-augmented techniques and include web scraping, data processing, recommendation engine, API development, and web application components.

## Requirements

### Requirement 1: Data Ingestion and Web Scraping

**User Story:** As a data engineer, I want to scrape and process SHL's assessment catalog to build a comprehensive database of individual test solutions, so that the recommendation system has accurate and up-to-date assessment data.

#### Acceptance Criteria

1. WHEN the system scrapes SHL's product catalog THEN it SHALL retrieve at least 377 Individual Test Solutions from https://www.shl.com/solutions/products/product-catalog/
2. WHEN parsing assessment data THEN the system SHALL extract assessment name, URL, category, test type, and description for each assessment
3. WHEN processing scraped data THEN the system SHALL ignore "Pre-packaged Job Solutions" category and focus only on individual test solutions
4. IF scraping encounters errors THEN the system SHALL implement retry logic and error handling
5. WHEN data ingestion completes THEN the system SHALL store structured assessment data in an efficient retrieval format

### Requirement 2: LLM-Based Recommendation Engine

**User Story:** As an AI engineer, I want to develop a recommendation engine using modern LLM and retrieval-augmented techniques, so that the system can intelligently match job queries with relevant SHL assessments.

#### Acceptance Criteria

1. WHEN the recommendation engine is built THEN it SHALL use modern LLM-based or retrieval-augmented techniques for query understanding
2. WHEN processing queries THEN the system SHALL handle natural language job descriptions, job description URLs, and free-form text queries
3. WHEN generating embeddings THEN the system SHALL create vector representations of both queries and assessment data for similarity matching
4. WHEN making recommendations THEN the system SHALL return between 1-10 most relevant assessments with balanced coverage across skill domains
5. WHEN query spans multiple domains THEN the system SHALL provide balanced recommendations covering both technical and behavioral assessments

### Requirement 3: Evaluation and Performance Optimization

**User Story:** As a quality assurance engineer, I want to evaluate and optimize the recommendation system's performance using provided datasets, so that it achieves high accuracy and relevance in assessment recommendations.

#### Acceptance Criteria

1. WHEN evaluation is performed THEN the system SHALL use Mean Recall@K metrics to measure recommendation accuracy
2. WHEN using training data THEN the system SHALL leverage the provided labeled train set with 10 human-labeled queries for iteration and improvement
3. WHEN testing system performance THEN the system SHALL evaluate against the unlabeled test set with 9 queries
4. WHEN measuring recommendation balance THEN the system SHALL ensure balanced coverage of assessment types (Knowledge & Skills vs Personality & Behavior)
5. WHEN optimization is complete THEN the system SHALL document performance improvements and methodology used

### Requirement 4: API Development and Endpoints

**User Story:** As a software developer, I want to create a robust REST API that provides access to the recommendation engine, so that it can be evaluated and integrated into existing workflows.

#### Acceptance Criteria

1. WHEN API is developed THEN it SHALL implement exactly two required endpoints: health check (/health) and recommendation (/recommend)
2. WHEN /health endpoint is called THEN it SHALL return API status and uptime information in JSON format
3. WHEN /recommend endpoint receives a query THEN it SHALL return 1-10 relevant assessments with assessment name and URL in tabular JSON format
4. WHEN API handles requests THEN it SHALL use proper HTTP status codes and JSON responses for all interactions
5. WHEN API is deployed THEN it SHALL be accessible via HTTP/HTTPS and handle concurrent requests efficiently

### Requirement 5: Web Application Frontend

**User Story:** As an end user, I want a user-friendly web interface to test the recommendation system, so that I can easily input queries and view recommended assessments.

#### Acceptance Criteria

1. WHEN web application is developed THEN it SHALL provide an intuitive interface for entering job descriptions or queries
2. WHEN user submits a query THEN the application SHALL display recommended assessments in a clear, tabular format
3. WHEN displaying results THEN the application SHALL show assessment names and provide clickable URLs to SHL's catalog
4. WHEN application is deployed THEN it SHALL be accessible via a public URL for evaluation purposes
5. WHEN user interacts with the interface THEN it SHALL provide responsive design and proper error handling

### Requirement 6: Submission and Deliverables

**User Story:** As a project evaluator, I want all required deliverables in the specified format, so that I can properly assess the technical completeness and robustness of the solution.

#### Acceptance Criteria

1. WHEN submission is prepared THEN it SHALL include three functional URLs: API endpoint, GitHub repository, and web application frontend
2. WHEN documentation is created THEN it SHALL include a 2-page approach document detailing solution methodology and performance optimization efforts
3. WHEN test predictions are generated THEN they SHALL be provided in a CSV file with exact format: Query, Assessment_url columns
4. WHEN code is submitted THEN it SHALL be accessible via public or shared GitHub repository with complete implementation including experiments
5. WHEN final submission occurs THEN all components SHALL be functional and accessible for evaluation

## Success Criteria

- Successful scraping and processing of at least 377 Individual Test Solutions from SHL's catalog
- Development of an LLM-based recommendation engine with measurable accuracy using Mean Recall@K metrics
- Creation of functional REST API with /health and /recommend endpoints
- Web application frontend for testing and demonstration
- Comprehensive evaluation using provided training and test datasets
- Complete submission package including URLs, documentation, and CSV predictions
- Balanced recommendation system that covers both technical and behavioral assessments appropriately

## Constraints and Assumptions

- Must use modern LLM-based or retrieval-augmented techniques (solutions without LLM integration will be rejected)
- Must implement web scraping of SHL catalog (solutions without scraping will be rejected)
- Must include measurable evaluation methods (solutions lacking evaluation will be rejected)
- API must follow exact endpoint specifications for automated evaluation
- Submission format must be followed precisely for scoring
- Free tier cloud platforms and APIs can be leveraged for deployment
- System must handle queries spanning multiple skill domains with balanced recommendations