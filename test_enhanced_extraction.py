#!/usr/bin/env python3
"""
Simple test for the enhanced PDF extraction functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

def test_direct_extraction():
    """Test direct PDF extraction methods"""
    
    resume_path = r"e:\OCRMAIN - Copy - Copy - Copy (2)\Rohan Lute Resume 1.pdf"
    
    if not os.path.exists(resume_path):
        print(f"Resume file not found: {resume_path}")
        return False
    
    print(f"Testing direct extraction with: {resume_path}")
    
    # Test Method 1: PyMuPDF
    try:
        import fitz
        doc = fitz.open(resume_path)
        
        all_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Try different extraction methods
            methods = []
            
            # Standard text
            try:
                text1 = page.get_text()
                methods.append(("standard", text1, len(text1)))
            except:
                pass
            
            # Dictionary method
            try:
                text_dict = page.get_text("dict")
                dict_text = ""
                for block in text_dict.get("blocks", []):
                    if block.get("type") == 0:
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                dict_text += span.get("text", "") + " "
                            dict_text += "\n"
                methods.append(("dict", dict_text, len(dict_text)))
            except:
                pass
            
            # Blocks method
            try:
                blocks = page.get_text("blocks")
                block_text = ""
                for block in blocks:
                    if len(block) >= 5:
                        block_text += block[4] + "\n"
                methods.append(("blocks", block_text, len(block_text)))
            except:
                pass
            
            if methods:
                # Use method with most text
                best_method, best_text, best_len = max(methods, key=lambda x: x[2])
                all_text += f"\n--- Page {page_num + 1} ({best_method}) ---\n{best_text}\n"
                print(f"Page {page_num + 1}: {best_method} method extracted {best_len} characters")
        
        doc.close()
        
        if all_text.strip():
            print(f"\nTotal extracted: {len(all_text)} characters")
            
            # Check for key resume content
            key_terms = ["ROHAN LUTE", "SUMMARY", "TECHNICAL SKILLS", "WORK EXPERIENCE", "PROJECTS", "EDUCATION"]
            found_terms = [term for term in key_terms if term.upper() in all_text.upper()]
            
            print(f"Key terms found: {len(found_terms)}/{len(key_terms)}")
            for term in found_terms:
                print(f"  ✓ {term}")
            
            # Save to file
            with open("direct_extraction_result.txt", "w", encoding="utf-8") as f:
                f.write(all_text)
            print(f"\nFull text saved to: direct_extraction_result.txt")
            
            # Show preview
            print(f"\nPreview (first 300 characters):")
            print("-" * 50)
            print(all_text[:300])
            print("-" * 50)
            
            return True
        else:
            print("No text extracted")
            return False
            
    except Exception as e:
        print(f"PyMuPDF extraction failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Enhanced PDF Extraction")
    print("=" * 40)
    
    success = test_direct_extraction()
    
    if success:
        print("\nDirect extraction test completed successfully!")
    else:
        print("\nDirect extraction test failed.")