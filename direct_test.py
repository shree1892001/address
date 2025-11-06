#!/usr/bin/env python3
"""
Direct test of PDF text extraction
"""

import os

def test_direct_pdf_extraction():
    """Test direct PDF extraction using PyMuPDF"""
    
    resume_path = r"e:\OCRMAIN - Copy - Copy - Copy (2)\Rohan Lute Resume 1.pdf"
    
    if not os.path.exists(resume_path):
        print(f"Resume file not found: {resume_path}")
        return False
    
    try:
        import fitz  # PyMuPDF
        
        print(f"Opening PDF: {resume_path}")
        doc = fitz.open(resume_path)
        
        all_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"\nProcessing page {page_num + 1}...")
            
            # Try different extraction methods
            methods = []
            
            # Method 1: Standard text
            try:
                text1 = page.get_text()
                methods.append(("standard", text1, len(text1)))
                print(f"  Standard method: {len(text1)} characters")
            except Exception as e:
                print(f"  Standard method failed: {e}")
            
            # Method 2: Dictionary-based
            try:
                text_dict = page.get_text("dict")
                dict_text = ""
                for block in text_dict.get("blocks", []):
                    if block.get("type") == 0:  # Text block
                        for line in block.get("lines", []):
                            line_text = ""
                            for span in line.get("spans", []):
                                span_text = span.get("text", "")
                                if span_text.strip():
                                    line_text += span_text + " "
                            if line_text.strip():
                                dict_text += line_text.strip() + "\n"
                methods.append(("dictionary", dict_text, len(dict_text)))
                print(f"  Dictionary method: {len(dict_text)} characters")
            except Exception as e:
                print(f"  Dictionary method failed: {e}")
            
            # Method 3: Blocks
            try:
                blocks = page.get_text("blocks")
                block_text = ""
                for block in blocks:
                    if len(block) >= 5:  # Valid text block
                        text = block[4].strip()
                        if text:
                            block_text += text + "\n"
                methods.append(("blocks", block_text, len(block_text)))
                print(f"  Blocks method: {len(block_text)} characters")
            except Exception as e:
                print(f"  Blocks method failed: {e}")
            
            # Use the method that extracted the most text
            if methods:
                best_method, best_text, best_len = max(methods, key=lambda x: x[2])
                print(f"  Best method: {best_method} with {best_len} characters")
                
                if best_text.strip():
                    all_text += f"\n=== PAGE {page_num + 1} ===\n{best_text.strip()}\n"
        
        doc.close()
        
        if all_text.strip():
            print(f"\nTotal extracted: {len(all_text)} characters")
            
            # Save to file
            with open("direct_test_result.txt", "w", encoding="utf-8") as f:
                f.write(all_text)
            print(f"Saved to: direct_test_result.txt")
            
            # Show preview
            print(f"\nPreview (first 500 characters):")
            print("-" * 50)
            print(all_text[:500])
            print("-" * 50)
            
            # Check for key content
            key_terms = ["ROHAN LUTE", "SUMMARY", "TECHNICAL SKILLS", "WORK EXPERIENCE", "PROJECTS", "EDUCATION"]
            found = [term for term in key_terms if term.upper() in all_text.upper()]
            print(f"\nKey terms found: {len(found)}/{len(key_terms)}")
            for term in found:
                print(f"  ✓ {term}")
            
            return True
        else:
            print("No text extracted")
            return False
            
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Direct PDF Text Extraction Test")
    print("=" * 40)
    
    success = test_direct_pdf_extraction()
    
    if success:
        print("\nDirect extraction successful!")
    else:
        print("\nDirect extraction failed.")