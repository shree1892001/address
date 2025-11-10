from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
import os
import tempfile
from typing import Dict, Any

from app.Services.TableExtractionServiceV2 import TableExtractionServiceV2
from app.Logger.ocr_logger import get_standard_logger

router = APIRouter()
logger = get_standard_logger("table_extraction_v2")

@router.post("/extract", response_model=Dict[str, Any])
async def extract_table_v2(
    file: UploadFile = File(..., description="Document file (PDF, PNG, JPEG, DOCX)")
):
    """
    Extract table data from uploaded document.
    
    Supports multiple file formats including PDF, PNG, JPEG, and DOCX.
    For image files and scanned documents, OCR will be automatically applied.
    """
    temp_dir = None
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.docx', '.doc'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Supported types: {', '.join(allowed_extensions)}"
            )
        
        # Initialize the extraction service
        extractor = TableExtractionServiceV2()
        
        # Process the file
        result = await extractor.extract_table(file)
        
        # If we have an Excel file, return it as a file download
        if result.get("excel_file") and os.path.exists(result["excel_file"]):
            return FileResponse(
                result["excel_file"],
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=f"extracted_table_{os.path.basename(file.filename)}.xlsx"
            )
        
        # Otherwise return the result as JSON
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in table extraction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
        
    finally:
        # Clean up any temporary files
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temp directory: {e}")
