from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import HTMLResponse

from app.Services.TableExtractionService import TableExtractionService
from app.Services.FileDownloadService import FileDownloadService
from app.Services.HealthCheckService import HealthCheckService
from app.Services.ResponseService import ResponseService
from app.Services.TemplateService import TemplateService
from app.Services.TextExtractionService import TextExtractionService

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
text_extraction_service = TextExtractionService()



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


@router.get("/test-extract-endpoint")
@log_method_entry_exit
@handle_general_operations(severity=ExceptionSeverity.LOW)
async def test_extract_endpoint():
	"""Test endpoint to verify extract-all-content is using new code"""
	logger.info("=" * 80)
	logger.info("TEST ENDPOINT CALLED - This confirms new code is running")
	logger.info("=" * 80)
	return {
		"status": "success",
		"message": "New extract-all-content endpoint is active",
		"endpoint_version": "comprehensive_text_extraction_service",
		"timestamp": "2025-11-06"
	}


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
	"""Extract ALL text content from PDF using comprehensive extraction service"""
	logger.info("=" * 80)
	logger.info("EXTRACT-ALL-CONTENT ENDPOINT CALLED - NEW VERSION")
	logger.info(f"File received: {file.filename}, Size: {file.size}")
	logger.info("=" * 80)
	
	try:
		from app.Services.FileValidationService import FileValidationService
		from app.Services.FileUploadService import FileUploadService
		
		file_validation_service = FileValidationService()
		file_upload_service = FileUploadService()
		
		file_validation_service.validate_pdf_file(file)
		file_paths = await file_upload_service.save_uploaded_file(file)
		
		logger.info(f"File saved to: {file_paths['pdf_path']}")
		logger.info("Calling text_extraction_service.extract_all_text_from_pdf_comprehensive...")
		
		# IMPORTANT: Use the comprehensive extraction service directly (bypasses region classification)
		# This method extracts ALL text from the PDF, not just classified regions
		extracted_text = await text_extraction_service.extract_all_text_from_pdf_comprehensive(file_paths["pdf_path"])
		
		logger.info(f"Extraction complete. Text length: {len(extracted_text)} characters")
		
		# Return as JSON with the full text
		response = {
			"success": True,
			"message": f"Successfully extracted all content from {file.filename}",
			"full_text": extracted_text,
			"character_count": len(extracted_text) if extracted_text else 0,
			"word_count": len(extracted_text.split()) if extracted_text else 0,
			"extraction_method": "comprehensive_text_extraction_service"
		}
		
		logger.info(f"Returning response with {response['character_count']} characters")
		return response
			
	except Exception as e:
		logger.error(f"Error extracting text: {e}", exc_info=True)
		return {
			"success": False,
			"error": f"Error extracting text: {str(e)}",
			"extraction_method": "comprehensive_text_extraction_service"
		}
