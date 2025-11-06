#!/usr/bin/env python3
"""
Test the complete text extraction functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

class MockUploadFile:
    """Mock UploadFile for testing"""
    def __init__(self, filename, file_path):
        self.filename = filename
        self.file_path = file_path
        self.content_type = "application/pdf"
    
    async def read(self):
        with open(self.file_path, 'rb') as f:
            return f.read()

async def test_complete_extraction():
    """Test the complete extraction functionality"""
    
    resume_path = r"e:\OCRMAIN - Copy - Copy - Copy (2)\Rohan Lute Resume 1.pdf"
    
    if not os.path.exists(resume_path):
        print(f"Resume file not found: {resume_path}")
        return False
    
    print(f"Testing complete extraction with: {resume_path}")
    
    try:
        from Services.TableExtractionService import TableExtractionService
        
        # Create mock upload file
        mock_file = MockUploadFile("Rohan Lute Resume 1.pdf", resume_path)
        
        # Initialize service
        extraction_service = TableExtractionService()
        
        # Test the extract_all_content_from_pdf method
        print("Starting complete content extraction...")
        result = await extraction_service.extract_all_content_from_pdf(mock_file)
        
        if result and "extracted_text" in result:
            extracted_text = result["extracted_text"]
            print(f"Extraction successful!")
            print(f"Total characters extracted: {len(extracted_text)}")
            print(f"Total lines: {len(extracted_text.splitlines())}")
            
            # Save full content to file
            with open("complete_extraction_result.txt", "w", encoding="utf-8") as f:
                f.write(extracted_text)
            print(f"Full extracted content saved to: complete_extraction_result.txt")
            
            # Show preview
            print(f"\nContent Preview (first 800 characters):")
            print("-" * 60)
            print(extracted_text[:800])
            if len(extracted_text) > 800:
                print("...")
            print("-" * 60)
            
            # Check for key resume sections
            key_sections = [
                "ROHAN LUTE", "SUMMARY", "TECHNICAL SKILLS", "WORK EXPERIENCE", 
                "PROJECTS", "EDUCATION", "CERTIFICATIONS", "Associate Developer",
                "Python", "Django", "JavaScript", "HTML", "CSS", "Bootstrap",
                "Vitric Business Solutions", "Ideal Energy Projects"
            ]
            
            found_sections = []
            for section in key_sections:
                if section.upper() in extracted_text.upper():
                    found_sections.append(section)
            
            print(f"\nKey sections found ({len(found_sections)}/{len(key_sections)}):")
            for section in found_sections:
                print(f"  + {section}")
            
            missing_sections = [s for s in key_sections if s not in found_sections]
            if missing_sections:
                print(f"\nMissing sections:")
                for section in missing_sections:
                    print(f"  - {section}")
            
            # Check content quality
            word_count = len(extracted_text.split())
            print(f"\nContent Quality:")
            print(f"  Word count: {word_count}")
            print(f"  Average words per line: {word_count / len(extracted_text.splitlines()):.1f}")
            
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
    print("Testing Complete PDF Text Extraction")
    print("=" * 50)
    
    success = asyncio.run(test_complete_extraction())
    
    if success:
        print("\nComplete extraction test successful!")
        print("The enhanced method should now extract all readable text from PDFs.")
    else:
        print("\nComplete extraction test failed.")