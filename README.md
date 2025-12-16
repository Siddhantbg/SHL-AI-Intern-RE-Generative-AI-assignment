# SHL Assessment Recommendation System

An intelligent recommendation system that helps hiring managers and recruiters find relevant SHL assessments based on natural language queries or job descriptions.

## Features

- **Web Scraping**: Extracts 377+ individual test solutions from SHL's product catalog
- **LLM-Based Recommendations**: Uses modern LLM and retrieval-augmented techniques
- **Balanced Results**: Ensures mix of Knowledge & Skills and Personality & Behavior assessments
- **REST API**: Provides `/health` and `/recommend` endpoints
- **Web Interface**: User-friendly frontend for testing recommendations
- **Evaluation System**: Measures performance using Mean Recall@K metrics

## Project Structure

```
├── src/                    # Source code
│   ├── scraper/           # Web scraping module
│   ├── processing/        # Data processing pipeline
│   ├── recommendation/    # Recommendation engine
│   ├── api/              # REST API
│   └── evaluation/       # Evaluation module
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── docs/                 # Documentation
├── data/                 # Data files (gitignored)
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Multi-service setup
└── pyproject.toml       # Project configuration
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional)
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd shl-assessment-recommender
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the Application

#### Using Python directly:
```bash
# Start the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Using Docker:
```bash
# Build and run with docker-compose
docker-compose up --build
```

### API Endpoints

- **Health Check**: `GET /health`
- **Get Recommendations**: `POST /recommend`

### Development

#### Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

#### Run tests:
```bash
pytest
```

#### Code formatting:
```bash
black src/ tests/
isort src/ tests/
```

#### Type checking:
```bash
mypy src/
```

#### Pre-commit hooks:
```bash
pre-commit install
pre-commit run --all-files
```

## Architecture

The system consists of several key components:

1. **Web Scraper**: Extracts assessment data from SHL catalog
2. **Data Processor**: Cleans and prepares data for embedding
3. **Vector Database**: Stores embeddings for similarity search
4. **LLM Service**: Processes queries and generates understanding
5. **Recommendation Engine**: Core matching and ranking logic
6. **REST API**: Provides programmatic access
7. **Web Frontend**: User interface for testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details.