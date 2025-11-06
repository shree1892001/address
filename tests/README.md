# 🧪 OCR Table Extraction API - Test Suite

This directory contains comprehensive unit tests and integration tests for the OCR Table Extraction API.

## 📁 Test Structure

```
tests/
├── __init__.py                 # Package initialization
├── conftest.py                 # Pytest configuration and shared fixtures
├── test_extract_table_api.py   # Main API endpoint tests
├── test_other_endpoints.py     # Tests for other API endpoints
├── run_tests.py               # Test runner script
├── requirements-test.txt      # Testing dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Testing Dependencies

```bash
pip install -r tests/requirements-test.txt
```

### 2. Run All Tests

```bash
# Using the test runner script
python tests/run_tests.py

# Or using pytest directly
pytest tests/ -v
```

### 3. Check Test Environment

```bash
python tests/run_tests.py --check-env
```

## 🎯 Test Categories

### Unit Tests (`@pytest.mark.unit`)
- **Purpose**: Test individual components in isolation
- **Coverage**: Service methods, utility functions, validation logic
- **Speed**: Fast execution
- **Dependencies**: Mocked external dependencies

### Integration Tests (`@pytest.mark.integration`)
- **Purpose**: Test component interactions
- **Coverage**: Service-to-service communication, database operations
- **Speed**: Medium execution time
- **Dependencies**: Some real dependencies, others mocked

### API Tests (`@pytest.mark.api`)
- **Purpose**: Test HTTP endpoints end-to-end
- **Coverage**: Request/response handling, status codes, data validation
- **Speed**: Fast to medium execution
- **Dependencies**: FastAPI TestClient, mocked services

## 📋 Test Coverage

### Main API Endpoint (`/api/v1/extract-table`)

#### ✅ Success Scenarios
- [x] Valid PDF file upload
- [x] Successful table extraction
- [x] Correct response structure
- [x] Multiple concurrent requests
- [x] Various filename patterns

#### ❌ Error Scenarios
- [x] Invalid file type (non-PDF)
- [x] Empty file upload
- [x] File size exceeding limits
- [x] No table found in document
- [x] Processing errors
- [x] File validation errors
- [x] Service exceptions

#### 🔍 Edge Cases
- [x] Large files (>10MB)
- [x] Corrupted PDF files
- [x] Files with no text content
- [x] Complex table layouts
- [x] Bilingual headers (English/Arabic)

### Other API Endpoints

#### Health Check (`/api/v1/health`)
- [x] Successful health check
- [x] Service error handling

#### Download (`/api/v1/download/{file_id}`)
- [x] Successful CSV download
- [x] File not found scenarios
- [x] Download service errors

#### Simple Extraction (`/api/v1/extract-simple`)
- [x] Successful simple extraction
- [x] Invalid file handling
- [x] Response structure validation

#### Upload Interface (`/api/v1/`)
- [x] Interface rendering
- [x] Template service errors

#### Test Download (`/api/v1/test-download`)
- [x] Test file download
- [x] Service error handling

## 🛠️ Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test file
pytest tests/test_extract_table_api.py

# Run specific test class
pytest tests/test_extract_table_api.py::TestExtractTableAPI

# Run specific test method
pytest tests/test_extract_table_api.py::TestExtractTableAPI::test_extract_table_success
```

### Using the Test Runner Script

```bash
# Run all tests
python tests/run_tests.py

# Run specific test types
python tests/run_tests.py --type unit
python tests/run_tests.py --type integration
python tests/run_tests.py --type api

# Run with coverage
python tests/run_tests.py --coverage

# Run in parallel
python tests/run_tests.py --parallel

# Run specific test file
python tests/run_tests.py --file tests/test_extract_table_api.py

# Check environment
python tests/run_tests.py --check-env
```

### Advanced Options

```bash
# Run tests with HTML report
pytest tests/ --html=test_report.html

# Run tests with JSON report
pytest tests/ --json-report

# Run tests with performance benchmarking
pytest tests/ --benchmark-only

# Run tests and stop on first failure
pytest tests/ -x

# Run tests and show local variables on failure
pytest tests/ -l

# Run tests with maximum verbosity
pytest tests/ -vvv
```

## 📊 Test Fixtures

### Shared Fixtures (conftest.py)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `test_client` | session | FastAPI TestClient instance |
| `temp_dir` | function | Temporary directory for test files |
| `sample_pdf_file` | function | Sample PDF file for testing |
| `mock_table_extraction_service` | function | Mocked table extraction service |
| `mock_file_validation_service` | function | Mocked file validation service |
| `mock_file_upload_service` | function | Mocked file upload service |
| `mock_response_service` | function | Mocked response service |
| `sample_table_data` | function | Sample table data for testing |
| `successful_extraction_response` | function | Mock successful extraction response |
| `error_response` | function | Mock error response |

### Test-Specific Fixtures

| Fixture | File | Description |
|---------|------|-------------|
| `client` | test_*.py | TestClient for each test class |
| `sample_pdf_content` | test_*.py | PDF content bytes for testing |
| `mock_table_extraction_response` | test_*.py | Mock extraction response |
| `mock_table_data` | test_*.py | Mock table data |

## 🔧 Test Configuration

### Pytest Configuration

The test suite uses the following pytest configuration:

```python
# conftest.py
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "api: mark test as an API test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
```

### Environment Variables

```bash
# Set during test execution
TESTING=true
```

### Test Directories

The test suite automatically creates and manages these directories:
- `app/temp_files/` - Temporary files during testing
- `app/temp_uploads/` - Temporary upload files
- `temp_uploads/` - Additional temporary files

## 📈 Coverage Reporting

### Generate Coverage Report

```bash
# HTML coverage report
pytest tests/ --cov=app --cov-report=html

# Terminal coverage report
pytest tests/ --cov=app --cov-report=term-missing

# XML coverage report (for CI/CD)
pytest tests/ --cov=app --cov-report=xml
```

### Coverage Targets

- **Overall Coverage**: >90%
- **API Endpoints**: >95%
- **Service Layer**: >85%
- **Error Handling**: >90%

## 🐛 Debugging Tests

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the project root
   cd /path/to/ocr-project
   python -m pytest tests/
   ```

2. **Missing Dependencies**
   ```bash
   # Install test dependencies
   pip install -r tests/requirements-test.txt
   ```

3. **Test Environment Issues**
   ```bash
   # Check test environment
   python tests/run_tests.py --check-env
   ```

### Debug Mode

```bash
# Run tests with debug output
pytest tests/ -v -s --tb=long

# Run specific failing test
pytest tests/test_extract_table_api.py::TestExtractTableAPI::test_extract_table_success -v -s
```

### Test Data

The test suite includes:
- Sample PDF files for testing
- Mock table data
- Mock service responses
- Error scenarios

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt
      - name: Run tests
        run: |
          python tests/run_tests.py --coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## 📝 Adding New Tests

### Test File Structure

```python
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

class TestNewFeature:
    """Test suite for new feature"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app1)
    
    def test_new_feature_success(self, client):
        """Test successful new feature"""
        # Test implementation
        pass
    
    def test_new_feature_error(self, client):
        """Test new feature error handling"""
        # Test implementation
        pass
```

### Test Naming Conventions

- **Test Classes**: `Test{FeatureName}`
- **Test Methods**: `test_{scenario}_{expected_result}`
- **Fixtures**: `{purpose}_{type}`

### Best Practices

1. **Use descriptive test names**
2. **Test both success and failure scenarios**
3. **Mock external dependencies**
4. **Use appropriate test markers**
5. **Keep tests independent**
6. **Clean up test data**

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [Python Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

## 🤝 Contributing

When adding new tests:

1. Follow the existing test structure
2. Add appropriate test markers
3. Include both positive and negative test cases
4. Update this README if adding new test categories
5. Ensure tests pass before submitting

---

*Last updated: January 2024*
*Test Suite Version: 1.0.0*
