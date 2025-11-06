from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
import os
import tempfile
from app.Services.DocxExtractionService import docx_extraction_service
from app.Logger.ocr_logger import get_standard_logger

router = APIRouter(
    prefix="/api/v1/docx",
    tags=["DOCX Processing"],
    responses={404: {"description": "Not found"}},
)

logger = get_standard_logger("DOCXRouter")

@router.post("/extract-tables", 
            summary="Extract tables from DOCX file",
            description="Upload a DOCX file to extract tables with formatting preserved.")
async def extract_tables_from_docx(file: UploadFile = File(...)):
    """
    Extract tables from an uploaded DOCX file.
    
    This endpoint handles DOCX files and extracts tables while preserving formatting.
    It works alongside the existing PDF/Image processing without modifying it.
    """
    try:
        # Validate file extension
        if not file.filename.lower().endswith(('.docx', '.doc')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only .docx and .doc files are supported."
            )
        
        # Create a temporary file to save the uploaded DOCX
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process the DOCX file
            tables = docx_extraction_service.extract_tables_from_docx(temp_file_path)
            
            return {
                "status": "success",
                "file_name": file.filename,
                "tables_found": len(tables),
                "tables": tables
            }
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error deleting temporary file {temp_file_path}: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing DOCX file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing DOCX file: {str(e)}"
        )

# Add the router to the main router in app/api/router/__init__.py
