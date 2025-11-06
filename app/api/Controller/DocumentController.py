from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import HTMLResponse

from app.Services.TableExtractionService import TableExtractionService
from app.Services.FileDownloadService import FileDownloadService
from app.Services.HealthCheckService import HealthCheckService
from app.Services.ResponseService import ResponseService
from app.Services.TemplateService import TemplateService
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import *

# Initialize router
router = APIRouter()
logger = get_standard_logger("DocumentController")

# Initialize services
table_extraction_service = TableExtractionService()
file_download_service = FileDownloadService()
health_check_service = HealthCheckService()
response_service = ResponseService()
template_service = TemplateService()


@router.get("/", response_class=HTMLResponse)
@log_method_entry_exit
@handle_general_operations(severity=ExceptionSeverity.LOW)
async def upload_interface(request: Request):
    """Serve the upload interface"""
    return template_service.render_upload_interface(request)


@router.post("/extract-table")
@monitor_performance
@log_method_entry_exit
@handle_general_operations(severity=ExceptionSeverity.HIGH)
async def extract_table_from_document(file: UploadFile = File(...)):
    """Extract table from uploaded PDF document"""
    try:
        response = await table_extraction_service.extract_table_from_pdf(file)
        return response_service.handle_table_extraction_response(response)
    except Exception as e:
        return response_service.handle_table_extraction_error(e, "table extraction")


@router.get("/download/{file_id}")
@log_method_entry_exit
@handle_file_operations(severity=ExceptionSeverity.MEDIUM)
async def extract_table_as_csv(file_id: str):
    """Download extracted table as CSV file"""
    try:
        response = file_download_service.download_csv_file(file_id)
        return response_service.handle_file_download_response(response)
    except Exception as e:
        return response_service.handle_file_download_error(e)


@router.get("/health")
@log_method_entry_exit
@handle_general_operations(severity=ExceptionSeverity.LOW)
async def health_check():
    """Health check endpoint"""
    response = health_check_service.get_health_status()
    return response_service.handle_health_check_response(response)


@router.get("/test-download")
@log_method_entry_exit
@handle_general_operations(severity=ExceptionSeverity.LOW)
async def test_download():
    """Test download endpoint"""
    try:
        response = file_download_service.download_test_file()
        return response_service.handle_file_download_response(response)
    except Exception as e:
        return response_service.handle_test_download_error(e)


@router.post("/extract-simple")
@monitor_performance
@log_method_entry_exit
@handle_general_operations(severity=ExceptionSeverity.MEDIUM)
async def extract_table_simple(file: UploadFile = File(...)):
    """Simplified table extraction endpoint"""
    try:
        response = await table_extraction_service.extract_table_simple(file)
        return response_service.handle_table_extraction_response(response)
    except Exception as e:
        return response_service.handle_table_extraction_error(e, "simple extraction")


@router.post("/extract-all-content")
@monitor_performance
@log_method_entry_exit
@handle_general_operations(severity=ExceptionSeverity.HIGH)
async def extract_all_content(file: UploadFile = File(...)):
    """Extract all content from PDF with table prioritization"""
    try:
        response = await table_extraction_service.extract_all_content_from_pdf(file)
        return response_service.handle_table_extraction_response(response)
    except Exception as e:
        return response_service.handle_table_extraction_error(e, "comprehensive extraction")
