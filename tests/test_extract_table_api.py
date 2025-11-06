import pytest
import os
import tempfile
import io
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
import json

# Import the main app
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Main import app1


class TestExtractTableAPI:
    """Test suite for the /api/v1/extract-table endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app1)
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing"""
        # This is a minimal PDF content for testing
        return b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test Table) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF'
    
    @pytest.fixture
    def mock_table_extraction_response(self):
        """Mock response for successful table extraction"""
        return {
            "success": True,
            "message": "Successfully extracted table from test_document.pdf",
            "rows_extracted": 10,
            "columns": 5,
            "excel_file": "/temp/test_document.xlsx",
            "timestamp": "2024-01-15T10:30:45.123456"
        }
    
    @pytest.fixture
    def mock_table_data(self):
        """Mock table data for testing"""
        return [
            ["Name", "Age", "City", "Salary", "Department"],
            ["John Doe", "30", "New York", "75000", "Engineering"],
            ["Jane Smith", "28", "Los Angeles", "70000", "Marketing"],
            ["Bob Johnson", "35", "Chicago", "80000", "Sales"],
            ["Alice Brown", "32", "Houston", "72000", "HR"],
            ["Charlie Wilson", "29", "Phoenix", "68000", "Engineering"],
            ["Diana Davis", "31", "Philadelphia", "75000", "Marketing"],
            ["Edward Miller", "33", "San Antonio", "78000", "Sales"],
            ["Fiona Garcia", "27", "San Diego", "65000", "HR"],
            ["George Martinez", "34", "Dallas", "82000", "Engineering"],
            ["Helen Rodriguez", "30", "San Jose", "76000", "Marketing"]
        ]

    def test_extract_table_success(self, client, sample_pdf_content, mock_table_extraction_response):
        """Test successful table extraction from PDF"""
        with patch('app.Services.TableExtractionService.TableExtractionService.extract_table_from_pdf') as mock_extract:
            mock_extract.return_value = mock_table_extraction_response
            
            # Create a mock PDF file
            files = {"file": ("test_document.pdf", sample_pdf_content, "application/pdf")}
            
            response = client.post("/api/v1/extract-table", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "Successfully extracted table from test_document.pdf"
            assert data["rows_extracted"] == 10
            assert data["columns"] == 5
            assert "excel_file" in data
            assert "timestamp" in data

    def test_extract_table_invalid_file_type(self, client):
        """Test API with invalid file type (non-PDF)"""
        # Create a text file instead of PDF
        text_content = b"This is a text file, not a PDF"
        files = {"file": ("test_document.txt", text_content, "text/plain")}
        
        response = client.post("/api/v1/extract-table", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid file type" in data["detail"]

    def test_extract_table_no_file_provided(self, client):
        """Test API without providing a file"""
        response = client.post("/api/v1/extract-table")
        
        assert response.status_code == 422  # Validation error

    def test_extract_table_empty_file(self, client):
        """Test API with empty file"""
        files = {"file": ("empty.pdf", b"", "application/pdf")}
        
        response = client.post("/api/v1/extract-table", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid file" in data["detail"] or "No table found" in data["detail"]

    def test_extract_table_large_file(self, client):
        """Test API with file exceeding size limit"""
        # Create a large file content (simulating > 10MB)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        files = {"file": ("large_document.pdf", large_content, "application/pdf")}
        
        response = client.post("/api/v1/extract-table", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "File size exceeds" in data["detail"]

    def test_extract_table_no_table_found(self, client, sample_pdf_content):
        """Test API when no table is found in the document"""
        with patch('app.Services.TableExtractionService.TableExtractionService.extract_table_from_pdf') as mock_extract:
            from app.Exceptions.custom_exceptions import NoTableFoundException
            mock_extract.side_effect = NoTableFoundException("test.pdf")
            
            files = {"file": ("test_document.pdf", sample_pdf_content, "application/pdf")}
            
            response = client.post("/api/v1/extract-table", files=files)
            
            assert response.status_code == 404
            data = response.json()
            assert "No table found" in data["detail"]

    def test_extract_table_processing_error(self, client, sample_pdf_content):
        """Test API when processing fails"""
        with patch('app.Services.TableExtractionService.TableExtractionService.extract_table_from_pdf') as mock_extract:
            mock_extract.side_effect = Exception("Processing failed")
            
            files = {"file": ("test_document.pdf", sample_pdf_content, "application/pdf")}
            
            response = client.post("/api/v1/extract-table", files=files)
            
            assert response.status_code == 500
            data = response.json()
            assert "Failed to process document" in data["detail"]

    def test_extract_table_file_validation_error(self, client):
        """Test API with file validation error"""
        with patch('app.Services.FileValidationService.FileValidationService.validate_pdf_file') as mock_validate:
            from app.Exceptions.custom_exceptions import InvalidFileTypeException
            mock_validate.side_effect = InvalidFileTypeException("Invalid file type")
            
            files = {"file": ("test_document.pdf", b"fake_pdf_content", "application/pdf")}
            
            response = client.post("/api/v1/extract-table", files=files)
            
            assert response.status_code == 400
            data = response.json()
            assert "Invalid file type" in data["detail"]

    def test_extract_table_with_actual_table_data(self, client, sample_pdf_content, mock_table_data):
        """Test API with realistic table data"""
        with patch('app.Services.TableExtractionService.TableExtractionService.extract_table_from_pdf') as mock_extract:
            mock_response = {
                "success": True,
                "message": "Successfully extracted table from employee_data.pdf",
                "rows_extracted": len(mock_table_data) - 1,  # Exclude header
                "columns": len(mock_table_data[0]) if mock_table_data else 0,
                "excel_file": "/temp/employee_data.xlsx",
                "timestamp": "2024-01-15T10:30:45.123456"
            }
            mock_extract.return_value = mock_response
            
            files = {"file": ("employee_data.pdf", sample_pdf_content, "application/pdf")}
            
            response = client.post("/api/v1/extract-table", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["rows_extracted"] == 10  # 11 rows - 1 header
            assert data["columns"] == 5

    def test_extract_table_response_structure(self, client, sample_pdf_content, mock_table_extraction_response):
        """Test that response has correct structure"""
        with patch('app.Services.TableExtractionService.TableExtractionService.extract_table_from_pdf') as mock_extract:
            mock_extract.return_value = mock_table_extraction_response
            
            files = {"file": ("test_document.pdf", sample_pdf_content, "application/pdf")}
            
            response = client.post("/api/v1/extract-table", files=files)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check required fields
            required_fields = ["success", "message", "rows_extracted", "columns", "excel_file", "timestamp"]
            for field in required_fields:
                assert field in data
            
            # Check data types
            assert isinstance(data["success"], bool)
            assert isinstance(data["message"], str)
            assert isinstance(data["rows_extracted"], int)
            assert isinstance(data["columns"], int)
            assert isinstance(data["excel_file"], str)
            assert isinstance(data["timestamp"], str)

    def test_extract_table_multiple_requests(self, client, sample_pdf_content, mock_table_extraction_response):
        """Test multiple concurrent requests"""
        with patch('app.Services.TableExtractionService.TableExtractionService.extract_table_from_pdf') as mock_extract:
            mock_extract.return_value = mock_table_extraction_response
            
            files = {"file": ("test_document.pdf", sample_pdf_content, "application/pdf")}
            
            # Make multiple requests
            responses = []
            for i in range(3):
                response = client.post("/api/v1/extract-table", files=files)
                responses.append(response)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

    def test_extract_table_content_type_validation(self, client):
        """Test API with different content types"""
        # Test with correct content type
        files = {"file": ("test.pdf", b"fake_pdf_content", "application/pdf")}
        response = client.post("/api/v1/extract-table", files=files)
        # Should fail due to invalid PDF content, but not due to content type
        assert response.status_code in [400, 500]
        
        # Test with wrong content type
        files = {"file": ("test.pdf", b"fake_pdf_content", "text/plain")}
        response = client.post("/api/v1/extract-table", files=files)
        assert response.status_code == 400

    def test_extract_table_filename_validation(self, client, sample_pdf_content):
        """Test API with different filename patterns"""
        with patch('app.Services.TableExtractionService.TableExtractionService.extract_table_from_pdf') as mock_extract:
            mock_extract.return_value = {
                "success": True,
                "message": "Successfully extracted table from document.pdf",
                "rows_extracted": 5,
                "columns": 3,
                "excel_file": "/temp/document.xlsx",
                "timestamp": "2024-01-15T10:30:45.123456"
            }
            
            # Test with various filename patterns
            test_filenames = [
                "document.pdf",
                "DOCUMENT.PDF",
                "test-document.pdf",
                "test_document.pdf",
                "test document.pdf",
                "123_document.pdf"
            ]
            
            for filename in test_filenames:
                files = {"file": (filename, sample_pdf_content, "application/pdf")}
                response = client.post("/api/v1/extract-table", files=files)
                assert response.status_code == 200

    def test_extract_table_error_handling_edge_cases(self, client, sample_pdf_content):
        """Test various error handling edge cases"""
        with patch('app.Services.TableExtractionService.TableExtractionService.extract_table_from_pdf') as mock_extract:
            # Test with various exceptions
            exceptions_to_test = [
                (Exception("Generic error"), 500),
                (ValueError("Invalid value"), 500),
                (OSError("File system error"), 500),
            ]
            
            for exception, expected_status in exceptions_to_test:
                mock_extract.side_effect = exception
                files = {"file": ("test.pdf", sample_pdf_content, "application/pdf")}
                response = client.post("/api/v1/extract-table", files=files)
                assert response.status_code == expected_status


class TestExtractTableAPIIntegration:
    """Integration tests for the extract-table API"""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app1)
    
    def test_api_endpoint_exists(self, client):
        """Test that the API endpoint is properly registered"""
        # Test that the endpoint exists by checking the OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        paths = schema.get("paths", {})
        
        # Check if our endpoint exists
        assert "/api/v1/extract-table" in paths
        assert "post" in paths["/api/v1/extract-table"]
    
    def test_api_documentation_accessible(self, client):
        """Test that API documentation is accessible"""
        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
