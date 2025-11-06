import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

# Import the main app
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Main import app1


class TestHealthCheckAPI:
    """Test suite for the health check endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app1)
    
    def test_health_check_success(self, client):
        """Test successful health check"""
        with patch('app.Services.HealthCheckService.HealthCheckService.get_health_status') as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:45.123456",
                "version": "1.0.0"
            }
            
            response = client.get("/api/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "version" in data

    def test_health_check_service_error(self, client):
        """Test health check when service fails"""
        with patch('app.Services.HealthCheckService.HealthCheckService.get_health_status') as mock_health:
            mock_health.side_effect = Exception("Health check failed")
            
            response = client.get("/api/v1/health")
            
            assert response.status_code == 500


class TestDownloadAPI:
    """Test suite for the download endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app1)
    
    def test_download_csv_success(self, client):
        """Test successful CSV download"""
        with patch('app.Services.FileDownloadService.FileDownloadService.download_csv_file') as mock_download:
            mock_response = Mock()
            mock_response.headers = {"Content-Type": "text/csv"}
            mock_response.content = b"Name,Age,City\nJohn,30,NYC\nJane,25,LA"
            mock_download.return_value = mock_response
            
            response = client.get("/api/v1/download/test-file-id")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/csv"

    def test_download_csv_file_not_found(self, client):
        """Test download when file is not found"""
        with patch('app.Services.FileDownloadService.FileDownloadService.download_csv_file') as mock_download:
            from app.Exceptions.custom_exceptions import FileNotFoundException
            mock_download.side_effect = FileNotFoundException("File not found")
            
            response = client.get("/api/v1/download/non-existent-file")
            
            assert response.status_code == 404
            data = response.json()
            assert "File not found" in data["detail"]

    def test_download_csv_service_error(self, client):
        """Test download when service fails"""
        with patch('app.Services.FileDownloadService.FileDownloadService.download_csv_file') as mock_download:
            mock_download.side_effect = Exception("Download failed")
            
            response = client.get("/api/v1/download/test-file-id")
            
            assert response.status_code == 500
            data = response.json()
            assert "Failed to download file" in data["detail"]


class TestSimpleExtractionAPI:
    """Test suite for the simple extraction endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app1)
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing"""
        return b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test Table) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF'
    
    def test_simple_extraction_success(self, client, sample_pdf_content):
        """Test successful simple table extraction"""
        with patch('app.Services.TableExtractionService.TableExtractionService.extract_table_simple') as mock_extract:
            mock_response = {
                "success": True,
                "message": "Successfully extracted table from test_document.pdf",
                "data": [
                    ["Name", "Age", "City"],
                    ["John", "30", "NYC"],
                    ["Jane", "25", "LA"]
                ],
                "rows": 2,
                "columns": 3
            }
            mock_extract.return_value = mock_response
            
            files = {"file": ("test_document.pdf", sample_pdf_content, "application/pdf")}
            
            response = client.post("/api/v1/extract-simple", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["rows"] == 2
            assert data["columns"] == 3
            assert "data" in data

    def test_simple_extraction_no_file(self, client):
        """Test simple extraction without file"""
        response = client.post("/api/v1/extract-simple")
        
        assert response.status_code == 422  # Validation error

    def test_simple_extraction_invalid_file(self, client):
        """Test simple extraction with invalid file"""
        text_content = b"This is not a PDF file"
        files = {"file": ("test.txt", text_content, "text/plain")}
        
        response = client.post("/api/v1/extract-simple", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid file type" in data["detail"]


class TestUploadInterfaceAPI:
    """Test suite for the upload interface endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app1)
    
    def test_upload_interface_success(self, client):
        """Test successful upload interface rendering"""
        with patch('app.Services.TemplateService.TemplateService.render_upload_interface') as mock_render:
            mock_render.return_value = "<html><body>Upload Form</body></html>"
            
            response = client.get("/api/v1/")
            
            assert response.status_code == 200
            assert "Upload Form" in response.text

    def test_upload_interface_service_error(self, client):
        """Test upload interface when service fails"""
        with patch('app.Services.TemplateService.TemplateService.render_upload_interface') as mock_render:
            mock_render.side_effect = Exception("Template rendering failed")
            
            response = client.get("/api/v1/")
            
            assert response.status_code == 500


class TestTestDownloadAPI:
    """Test suite for the test download endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app1)
    
    def test_test_download_success(self, client):
        """Test successful test download"""
        with patch('app.Services.FileDownloadService.FileDownloadService.download_test_file') as mock_download:
            mock_response = Mock()
            mock_response.headers = {"Content-Type": "application/octet-stream"}
            mock_response.content = b"test file content"
            mock_download.return_value = mock_response
            
            response = client.get("/api/v1/test-download")
            
            assert response.status_code == 200

    def test_test_download_service_error(self, client):
        """Test test download when service fails"""
        with patch('app.Services.FileDownloadService.FileDownloadService.download_test_file') as mock_download:
            mock_download.side_effect = Exception("Test download failed")
            
            response = client.get("/api/v1/test-download")
            
            assert response.status_code == 500
            data = response.json()
            assert "Failed to create test download" in data["detail"]


class TestAPIEndpointsIntegration:
    """Integration tests for all API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app1)
    
    def test_all_endpoints_exist(self, client):
        """Test that all expected endpoints are registered"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        paths = schema.get("paths", {})
        
        # Check all expected endpoints
        expected_endpoints = [
            "/api/v1/",
            "/api/v1/extract-table",
            "/api/v1/extract-simple",
            "/api/v1/download/{file_id}",
            "/api/v1/health",
            "/api/v1/test-download"
        ]
        
        for endpoint in expected_endpoints:
            assert endpoint in paths, f"Endpoint {endpoint} not found in API schema"
    
    def test_api_documentation_endpoints(self, client):
        """Test that API documentation endpoints are accessible"""
        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
        
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
    
    def test_cors_headers(self, client):
        """Test that CORS headers are properly set"""
        response = client.options("/api/v1/health")
        
        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
