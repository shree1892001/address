"""
Pytest configuration and shared fixtures for OCR Table Extraction API tests
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Add the parent directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Main import app1


@pytest.fixture(scope="session")
def test_client():
    """Create a test client for the FastAPI app"""
    return TestClient(app1)


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_pdf_file(temp_dir):
    """Create a sample PDF file for testing"""
    pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test Table) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF'
    
    pdf_path = os.path.join(temp_dir, "test_document.pdf")
    with open(pdf_path, 'wb') as f:
        f.write(pdf_content)
    
    return pdf_path


@pytest.fixture(scope="function")
def mock_table_extraction_service():
    """Mock the table extraction service"""
    with patch('app.Services.TableExtractionService.TableExtractionService') as mock_service:
        mock_instance = Mock()
        mock_service.return_value = mock_instance
        yield mock_instance


@pytest.fixture(scope="function")
def mock_file_validation_service():
    """Mock the file validation service"""
    with patch('app.Services.FileValidationService.FileValidationService') as mock_service:
        mock_instance = Mock()
        mock_service.return_value = mock_instance
        yield mock_instance


@pytest.fixture(scope="function")
def mock_file_upload_service():
    """Mock the file upload service"""
    with patch('app.Services.FileUploadService.FileUploadService') as mock_service:
        mock_instance = Mock()
        mock_service.return_value = mock_instance
        yield mock_instance


@pytest.fixture(scope="function")
def mock_response_service():
    """Mock the response service"""
    with patch('app.Services.ResponseService.ResponseService') as mock_service:
        mock_instance = Mock()
        mock_service.return_value = mock_instance
        yield mock_instance


@pytest.fixture(scope="function")
def sample_table_data():
    """Sample table data for testing"""
    return [
        ["Name", "Age", "City", "Salary", "Department"],
        ["John Doe", "30", "New York", "75000", "Engineering"],
        ["Jane Smith", "28", "Los Angeles", "70000", "Marketing"],
        ["Bob Johnson", "35", "Chicago", "80000", "Sales"],
        ["Alice Brown", "32", "Houston", "72000", "HR"],
        ["Charlie Wilson", "29", "Phoenix", "68000", "Engineering"]
    ]


@pytest.fixture(scope="function")
def successful_extraction_response():
    """Mock successful extraction response"""
    return {
        "success": True,
        "message": "Successfully extracted table from test_document.pdf",
        "rows_extracted": 5,
        "columns": 5,
        "excel_file": "/temp/test_document.xlsx",
        "timestamp": "2024-01-15T10:30:45.123456"
    }


@pytest.fixture(scope="function")
def error_response():
    """Mock error response"""
    return {
        "detail": "An error occurred during processing"
    }


# Test markers for different types of tests
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Test environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests"""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    
    # Create test directories if they don't exist
    test_dirs = [
        "app/temp_files",
        "app/temp_uploads",
        "temp_uploads"
    ]
    
    for test_dir in test_dirs:
        os.makedirs(test_dir, exist_ok=True)
    
    yield
    
    # Cleanup after all tests
    print("Test session completed")


# Test data cleanup
@pytest.fixture(scope="function", autouse=True)
def cleanup_test_data():
    """Clean up test data after each test"""
    yield
    
    # Clean up any temporary files created during tests
    temp_dirs = [
        "app/temp_files",
        "app/temp_uploads",
        "temp_uploads"
    ]
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception:
                    pass
