
import os
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from tqdm import tqdm
from typing import List

class CombinedAmharicTextExtractor:
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths
        print(f"Initialized extractor for {len(file_paths)} files")

        # Configure pytesseract
        try:
            # For Windows
            if os.name == 'nt':
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            # For Linux/Mac, it's usually in PATH
            else:
                pytesseract.pytesseract.tesseract_cmd = r"tesseract"
            print("Tesseract OCR initialized successfully.")
            self.tesseract_available = True
        except Exception as e:
            print(f"Error configuring Tesseract OCR: {e}. OCR fallback will be skipped.")
            self.tesseract_available = False

    def preprocess_image(self, img: Image.Image) -> Image.Image:
        """Enhanced preprocessing for better OCR on math books with Amharic text."""
        img = img.convert("L")  # Convert to grayscale
        img = ImageEnhance.Sharpness(img).enhance(2.5)  # Increased sharpness for Amharic
        img = ImageEnhance.Contrast(img).enhance(2.8)   # Increased contrast for better definition
        img = img.filter(ImageFilter.MedianFilter(size=3))  # Remove noise
        img = img.point(lambda x: 0 if x < 120 else 255, "1")  # Lower threshold for Amharic scripts
        return img

    def normalize_amharic_text(self, text: str) -> str:
        """Post-processing for Amharic text to improve quality."""
        if not text:
            return text
            
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Replace common OCR errors in Amharic text
        replacements = {
            '0': '፩',  # Example replacement, adjust based on actual OCR errors
            '1': '፩',
            '2': '፪',
            '3': '፫',
            # Add more replacements as needed
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF using multiple methods."""
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file '{pdf_path}' not found.")
            return ""

        all_text = f"\n\n==== {os.path.basename(pdf_path)} ====\n\n"
        
        # Try to extract text directly first (for selectable text)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"Extracting selectable text from page {page_num}/{len(pdf.pages)}")
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        normalized_text = self.normalize_amharic_text(page_text)
                        all_text += f"--- Page {page_num} ---\n{normalized_text}\n\n"
        except Exception as e:
            print(f"Warning: pdfplumber extraction failed: {e}")
        
        # Try PyMuPDF as another method for selectable text
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")
                # Only add text if pdfplumber didn't get it
                if page_text and page_text.strip() and f"--- Page {page_num+1} ---" not in all_text:
                    normalized_text = self.normalize_amharic_text(page_text)
                    all_text += f"--- Page {page_num+1} ---\n{normalized_text}\n\n"
            doc.close()
        except Exception as e:
            print(f"Warning: PyMuPDF extraction failed: {e}")
        
        # For non-selectable text, use OCR
        pages_with_text = set()
        for line in all_text.split("\n"):
            if line.startswith("--- Page "):
                try:
                    page_number = int(line.split("---")[1].strip().split(" ")[1])
                    pages_with_text.add(page_number)
                except (ValueError, IndexError):
                    pass
        
        # Only do OCR on pages that don't have text yet
        if self.tesseract_available:
            try:
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                
                for page_num in range(total_pages):
                    # Skip if we already have text for this page
                    if page_num + 1 in pages_with_text:
                        print(f"Skipping OCR for page {page_num+1} - already has text")
                        continue
                        
                    print(f"Performing OCR on page {page_num+1}/{total_pages}")
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=600)  # Higher DPI for Amharic
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img = self.preprocess_image(img)
                    
                    # Try different PSM modes
                    best_text = ""
                    for psm in [6, 3, 11]:
                        custom_config = f'--oem 3 --psm {psm} -l eng+amh+equ'
                        page_text = pytesseract.image_to_string(img, config=custom_config)
                        if len(page_text.strip()) > len(best_text.strip()):
                            best_text = page_text
                    
                    if best_text.strip():
                        normalized_text = self.normalize_amharic_text(best_text)
                        all_text += f"--- Page {page_num+1} (OCR) ---\n{normalized_text}\n\n"
                
                doc.close()
            except Exception as e:
                print(f"Warning: OCR processing failed: {e}")
        
        return all_text

    def extract_all_to_single_file(self, output_path: str):
        """Extract text from all PDFs and combine into one file."""
        print(f"Starting extraction of {len(self.file_paths)} files to single output: {output_path}")
        
        combined_text = "COMBINED AMHARIC TEXT EXTRACTION\n\n"
        
        for pdf_path in self.file_paths:
            print(f"\n=== Processing: {pdf_path} ===")
            pdf_text = self.extract_text_from_pdf(pdf_path)
            combined_text += pdf_text
            combined_text += "\n" + "=" * 80 + "\n\n"
        
        # Write all text to a single file
        try:
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(combined_text)
            print(f"\nSuccessfully saved all extracted text to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving combined text: {e}")
            return False

# Usage Example
if __name__ == "__main__":
    FILE_PATHS = [
            "1ኛ ክፍል አካባቢ ሳይንስ መጽሐፍ - ምዕራፍ 3.pdf"
    ]
    
    OUTPUT_FILE = "all_text.txt"
    
    extractor = CombinedAmharicTextExtractor(FILE_PATHS)
    extractor.extract_all_to_single_file(OUTPUT_FILE)
