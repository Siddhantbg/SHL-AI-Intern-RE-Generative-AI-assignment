# SHL Catalog Scraper

This module provides functionality to scrape assessment data from SHL's product catalog, focusing on individual test solutions while filtering out pre-packaged job solutions.

## Features

- **Web Scraping**: Extracts assessment data from SHL's product catalog
- **Intelligent Filtering**: Excludes pre-packaged job solutions, focuses on individual tests
- **Retry Logic**: Robust error handling with exponential backoff
- **Data Validation**: Ensures minimum 377 assessments are collected
- **Test Type Classification**: Automatically categorizes assessments as Knowledge & Skills (K) or Personality & Behavior (P)
- **Comprehensive Testing**: Full unit test coverage with mock responses

## Classes

### SHLCatalogScraper

Main scraper class that handles the extraction of assessment data.

```python
from src.scraper import SHLCatalogScraper

# Initialize scraper
scraper = SHLCatalogScraper(max_retries=3, retry_delay=1.0)

# Scrape all individual test solutions
assessments = scraper.scrape_catalog()

# Validate minimum count requirement
is_valid = scraper.validate_minimum_count(assessments)
```

#### Key Methods

- `scrape_catalog()`: Main method to scrape the complete catalog
- `get_individual_tests()`: Alias for scrape_catalog()
- `validate_minimum_count(assessments)`: Validates minimum 377 assessments
- `_parse_assessment_details(url)`: Extracts details from individual assessment pages
- `_determine_test_type(content, name)`: Classifies assessment type (K or P)
- `_is_individual_test_solution(data)`: Filters out pre-packaged solutions

### Assessment

Data class representing an SHL assessment.

```python
@dataclass
class Assessment:
    id: str
    name: str
    url: str
    category: str
    test_type: str  # 'K' for Knowledge & Skills, 'P' for Personality & Behavior
    description: str
    skills: List[str]
```

### DataValidator

Utility class for validating scraped assessment data.

```python
from src.scraper import DataValidator

# Validate individual assessment
is_valid = DataValidator.validate_assessment(assessment)

# Validate list of assessments with detailed report
report = DataValidator.validate_assessment_list(assessments)
```

## Usage Examples

### Basic Scraping

```python
from src.scraper import SHLCatalogScraper, DataValidator

# Initialize scraper
scraper = SHLCatalogScraper()

# Scrape assessments
assessments = scraper.scrape_catalog()

# Validate results
validation_report = DataValidator.validate_assessment_list(assessments)
print(f"Scraped {len(assessments)} assessments")
print(f"Validation rate: {validation_report['validation_rate']:.2%}")
```

### Using the CLI Script

```bash
# Run scraper and save to JSON
python scripts/run_scraper.py --output data/scraped/assessments.json

# Run with verbose logging
python scripts/run_scraper.py --verbose

# Validate existing data without scraping
python scripts/run_scraper.py --validate-only --output data/scraped/assessments.json
```

### Demo Script

```bash
# Run the demo to see scraper functionality
python examples/scraper_demo.py
```

## Configuration

### Scraper Parameters

- `max_retries`: Maximum number of retry attempts for failed requests (default: 3)
- `retry_delay`: Base delay between retry attempts in seconds (default: 1.0)
- Uses exponential backoff for retries

### URL Configuration

- `BASE_URL`: "https://www.shl.com"
- `CATALOG_URL`: "https://www.shl.com/solutions/products/product-catalog/"

## Data Extraction

### Assessment Information Extracted

1. **Name**: Assessment title
2. **URL**: Direct link to assessment page
3. **Category**: Extracted from URL path or content
4. **Test Type**: 
   - 'K' for Knowledge & Skills assessments
   - 'P' for Personality & Behavior assessments
5. **Description**: Assessment description and purpose
6. **Skills**: List of skills/competencies measured

### Filtering Logic

The scraper excludes assessments containing these keywords:
- "pre-packaged"
- "job solution"
- "package"
- "bundle"
- "suite"
- "collection"
- "set of assessments"

### Test Type Classification

**Knowledge & Skills (K) Keywords:**
- technical, coding, programming, software
- numerical, verbal, logical, reasoning
- aptitude, cognitive, ability, skill
- competency, knowledge, mathematics
- language, analytical

**Personality & Behavior (P) Keywords:**
- personality, behavior, motivation, values
- leadership, teamwork, communication
- emotional, social, cultural, fit
- style, preference, trait, character

## Error Handling

### Network Errors
- Automatic retry with exponential backoff
- Configurable retry attempts and delays
- Graceful handling of timeouts and connection errors

### Parsing Errors
- Continues processing other assessments if one fails
- Logs detailed error information
- Returns partial results when possible

### Validation Errors
- Comprehensive validation of required fields
- URL format validation
- Test type validation (must be 'K' or 'P')

## Testing

### Running Tests

```bash
# Run all scraper tests
python -m pytest tests/unit/test_shl_catalog_scraper.py -v

# Run with coverage
python -m pytest tests/unit/test_shl_catalog_scraper.py --cov=src.scraper --cov-report=html
```

### Test Coverage

The test suite includes:
- Unit tests for all public methods
- Mock HTTP responses for network testing
- Edge case handling
- Integration tests
- Data validation tests

Current coverage: 90%+ on scraper module

## Requirements

### Dependencies

- `requests`: HTTP client for web scraping
- `beautifulsoup4`: HTML parsing
- `pytest`: Testing framework (dev dependency)

### System Requirements

- Python 3.8+
- Network access to SHL website
- Minimum 377 assessments for validation to pass

## Logging

The scraper uses Python's built-in logging module:

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

# The scraper will log:
# - Request attempts and failures
# - Parsing progress
# - Validation results
# - Error details
```

## Performance Considerations

### Rate Limiting
- Built-in 0.5-second delay between requests
- Respectful of server resources
- Can be adjusted if needed

### Memory Usage
- Processes assessments incrementally
- Stores results in memory during scraping
- Consider batch processing for very large catalogs

### Network Efficiency
- Reuses HTTP session for connection pooling
- Implements proper timeout handling
- Uses appropriate User-Agent headers

## Troubleshooting

### Common Issues

1. **Network Timeouts**
   - Increase `retry_delay` parameter
   - Check network connectivity
   - Verify SHL website accessibility

2. **Insufficient Assessments**
   - Check if SHL catalog structure has changed
   - Verify filtering logic isn't too restrictive
   - Review scraping selectors

3. **Parsing Errors**
   - Enable verbose logging to see detailed errors
   - Check if HTML structure has changed
   - Verify BeautifulSoup selectors

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create scraper with verbose settings
scraper = SHLCatalogScraper(max_retries=1, retry_delay=0.1)
```

## Contributing

When contributing to the scraper module:

1. Maintain test coverage above 90%
2. Add tests for new functionality
3. Follow existing code style and patterns
4. Update documentation for API changes
5. Test with mock responses to avoid hitting live servers during development

## License

This module is part of the SHL Assessment Recommendation System project.