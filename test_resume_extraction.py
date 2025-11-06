#!/usr/bin/env python3
"""
Test script to verify the enhanced extract-all-content functionality
with the provided resume PDF
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from Services.TableExtractionService import TableExtractionService
from Services.FileValidationService import FileValidationService
from Services.FileUploadService import FileUploadService

class MockUploadFile:
    """Mock UploadFile for testing"""
    def __init__(self, filename, file_path):
        self.filename = filename
        self.file_path = file_path
        self.content_type = "application/pdf"
    
    async def read(self):
        with open(self.file_path, 'rb') as f:
            return f.read()

async def test_resume_extraction():
    """Test the enhanced extraction with the resume PDF"""
    
    # Path to the resume PDF
    resume_path = r"e:\OCRMAIN - Copy - Copy - Copy (2)\Rohan Lute Resume 1.pdf"
    
    if not os.path.exists(resume_path):
        print(f"Resume file not found: {resume_path}")
        return False
    
    print(f"Testing extraction with: {resume_path}")
    
    try:
        # Create mock upload file
        mock_file = MockUploadFile("Rohan Lute Resume 1.pdf", resume_path)
        
        # Initialize service
        extraction_service = TableExtractionService()
        
        # Test the enhanced extract_all_content_from_pdf method
        print("Starting comprehensive content extraction...")
        result = await extraction_service.extract_all_content_from_pdf(mock_file)
        
        if result and "extracted_text" in result:
            extracted_text = result["extracted_text"]
            print(f"Extraction successful!")
            print(f"Total characters extracted: {len(extracted_text)}")
            print(f"Total lines: {len(extracted_text.splitlines())}")
            
            # Show first 500 characters as preview
            print("\nContent Preview (first 500 characters):")
            print("-" * 60)
            print(extracted_text[:500])
            if len(extracted_text) > 500:
                print("...")
            print("-" * 60)
            
            # Check for key resume sections
            sections_found = []
            key_sections = [
                "ROHAN LUTE", "SUMMARY", "TECHNICAL SKILLS", "WORK EXPERIENCE", 
                "PROJECTS", "EDUCATION", "CERTIFICATIONS", "Associate Developer",
                "Python", "Django", "JavaScript", "HTML", "CSS"
            ]
            
            for section in key_sections:
                if section.upper() in extracted_text.upper():
                    sections_found.append(section)
            
            print(f"\nKey sections found ({len(sections_found)}/{len(key_sections)}):")
            for section in sections_found:
                print(f"  + {section}")
            
            missing_sections = [s for s in key_sections if s not in sections_found]
            if missing_sections:
                print(f"\nMissing sections:")
                for section in missing_sections:
                    print(f"  - {section}")
            
            # Save extracted content to file for review
            output_file = "extracted_resume_content.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print(f"\nFull extracted content saved to: {output_file}")
            
            return True
        else:
            print("No content extracted")
            return False
            
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Enhanced PDF Content Extraction")
    print("=" * 50)
    
    success = asyncio.run(test_resume_extraction())
    
    if success:
        print("\nTest completed successfully!")
        print("The enhanced extraction method should now work with various PDF templates and formats.")
    else:
        print("\nTest failed. Check the error messages above.")