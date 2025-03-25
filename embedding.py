# # import os
# # import numpy as np
# # import openai
# # import PyPDF2
# # import pytesseract
# # from pdf2image import convert_from_path
# # from tqdm import tqdm
# # from typing import Dict, List

# # class MathBookEmbedder:
# #     def __init__(self, file_paths: List[str], api_key: str, model_name="text-embedding-3-small"):
# #         """
# #         Initialize the MathBookEmbedder with multiple file paths and OpenAI API key.

# #         Args:
# #             file_paths (List[str]): List of paths to the math books.
# #             api_key (str): OpenAI API key.
# #             model_name (str): OpenAI embedding model.
# #         """
# #         self.file_paths = file_paths
# #         self.model_name = model_name
# #         self.client = openai.OpenAI(api_key=api_key)
# #         print(f"Initialized OpenAI client with model: {model_name}")

# #     def extract_text_from_pdf(self, pdf_path: str) -> str:
# #         """
# #         Extract text from a PDF file. If text cannot be extracted, perform OCR.

# #         Args:
# #             pdf_path (str): Path to the PDF file.

# #         Returns:
# #             str: Extracted text.
# #         """
# #         text = ""
# #         try:
# #             with open(pdf_path, "rb") as file:
# #                 reader = PyPDF2.PdfReader(file)
# #                 for page in reader.pages:
# #                     page_text = page.extract_text()
# #                     if page_text:
# #                         text += page_text + "\n"
# #         except:
# #             print(f"Failed text extraction from {pdf_path}, using OCR...")

# #         if not text.strip():
# #             images = convert_from_path(pdf_path)
# #             for img in tqdm(images, desc="Performing OCR"):
# #                 text += pytesseract.image_to_string(img, lang="eng") + "\n"

# #         return text.strip()

# #     def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
# #         """
# #         Generate OpenAI embeddings for given text chunks.

# #         Args:
# #             texts (List[str]): List of text chunks.

# #         Returns:
# #             List[np.ndarray]: List of vector embeddings.
# #         """
# #         embeddings = []
# #         for text in tqdm(texts, desc="Generating embeddings"):
# #             try:
# #                 response = self.client.embeddings.create(model=self.model_name, input=text)
# #                 embedding = np.array(response.data[0].embedding)
# #                 embeddings.append(embedding)
# #             except Exception as e:
# #                 print(f"Error embedding text: {e}")
# #                 embeddings.append(np.zeros(1536))  # Handle missing embeddings gracefully

# #         return embeddings

# #     def process_books(self, output_path: str):
# #         """
# #         Process all books: extract text, generate embeddings, and save.

# #         Args:
# #             output_path (str): File path to save embeddings.
# #         """
# #         all_texts = []
# #         page_numbers = []

# #         for pdf in self.file_paths:
# #             print(f"Processing: {pdf}")
# #             extracted_text = self.extract_text_from_pdf(pdf)
# #             text_chunks = [extracted_text[i:i+1000] for i in range(0, len(extracted_text), 1000)]
            
# #             all_texts.extend(text_chunks)
# #             page_numbers.extend([pdf] * len(text_chunks))  # Track source

# #         embeddings = self.generate_embeddings(all_texts)

# #         np.savez(output_path, page_numbers=page_numbers, embeddings=np.array(embeddings))
# #         print(f"Embeddings saved at: {output_path}")

# # # Usage Example
# # if __name__ == "__main__":
# #     FILE_PATHS = [
# #         "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 1.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 2.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 3.pdf",
# #         "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 4.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 5.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 6.pdf",
# #         "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 7.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 8.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 9.pdf",
# #         "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 10.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 12.pdf"  # Add paths to all 12 books
# #     ]
# #     API_KEY = "REMOVED"
# #     OUTPUT_FILE = "math_books_embeddings.npz"

# #     embedder = MathBookEmbedder(FILE_PATHS, API_KEY)
# #     embedder.process_books(OUTPUT_FILE)





# import os
# import numpy as np
# import openai
# import pdfplumber
# import fitz  # PyMuPDF
# from PIL import Image
# import pytesseract
# from tqdm import tqdm
# from typing import Dict, List

# class MathBookEmbedder:
#     def __init__(self, file_paths: List[str], api_key: str, model_name="text-embedding-3-small"):
#         """
#         Initialize the MathBookEmbedder with multiple file paths and OpenAI API key.

#         Args:
#             file_paths (List[str]): List of paths to the math books.
#             api_key (str): OpenAI API key.
#             model_name (str): OpenAI embedding model.
#         """
#         self.file_paths = file_paths
#         self.model_name = model_name
#         self.client = openai.OpenAI(api_key=api_key)
#         print(f"Initialized OpenAI client with model: {model_name}")

#         # Configure pytesseract (ensure Tesseract is installed)
#         try:
#             pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update path as needed
#             print("Tesseract OCR initialized successfully.")
#         except Exception as e:
#             print(f"Error configuring Tesseract OCR: {e}. OCR fallback will be skipped.")
#             self.tesseract_available = False
#         else:
#             self.tesseract_available = True

#     def extract_text_from_pdf(self, pdf_path: str) -> str:
#         """
#         Extract text from a PDF file using pdfplumber. If text cannot be extracted, use Tesseract OCR.

#         Args:
#             pdf_path (str): Path to the PDF file.

#         Returns:
#             str: Extracted text.
#         """
#         text = ""
#         try:
#             with pdfplumber.open(pdf_path) as pdf:
#                 for page in pdf.pages:
#                     page_text = page.extract_text()
#                     if page_text:
#                         text += page_text + "\n"
#         except Exception as e:
#             print(f"Failed text extraction from {pdf_path} using pdfplumber: {e}")

#         # Fallback to Tesseract OCR if text extraction fails and Tesseract is available
#         if not text.strip() and self.tesseract_available:
#             print(f"Using Tesseract OCR for {pdf_path}...")
#             text = self.ocr_with_tesseract(pdf_path)

#         return text.strip() if text else ""

#     def ocr_with_tesseract(self, pdf_path: str) -> str:
#         """
#         Perform OCR using Tesseract by converting PDF pages to images with PyMuPDF.

#         Args:
#             pdf_path (str): Path to the PDF file.

#         Returns:
#             str: Extracted text via OCR.
#         """
#         text = ""
#         try:
#             doc = fitz.open(pdf_path)
#             for page_num in tqdm(range(len(doc)), desc="Performing OCR with Tesseract"):
#                 page = doc.load_page(page_num)
#                 pix = page.get_pixmap(dpi=300)  # Higher DPI for better accuracy
#                 img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                 text += pytesseract.image_to_string(img, lang="eng+amh") + "\n"  # English and Amharic
#         except Exception as e:
#             print(f"Tesseract OCR failed for {pdf_path}: {e}")
#         return text

#     def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
#         """
#         Generate OpenAI embeddings for given text chunks.

#         Args:
#             texts (List[str]): List of text chunks.

#         Returns:
#             List[np.ndarray]: List of vector embeddings.
#         """
#         embeddings = []
#         for text in tqdm(texts, desc="Generating embeddings"):
#             if not text.strip():  # Skip empty chunks
#                 embeddings.append(np.zeros(1536))
#                 continue
#             try:
#                 response = self.client.embeddings.create(model=self.model_name, input=text)
#                 embedding = np.array(response.data[0].embedding)
#                 embeddings.append(embedding)
#             except Exception as e:
#                 print(f"Error embedding text: {e}")
#                 embeddings.append(np.zeros(1536))  # Handle missing embeddings

#         return embeddings

#     def process_books(self, output_path: str):
#         """
#         Process all books: extract text, generate embeddings, and save.

#         Args:
#             output_path (str): File path to save embeddings.
#         """
#         all_texts = []
#         page_numbers = []

#         for pdf in self.file_paths:
#             print(f"Processing: {pdf}")
#             extracted_text = self.extract_text_from_pdf(pdf)
#             if not extracted_text:
#                 print(f"Skipping {pdf}: No text extracted.")
#                 continue
#             text_chunks = [extracted_text[i:i+1000] for i in range(0, len(extracted_text), 1000)]
            
#             all_texts.extend(text_chunks)
#             page_numbers.extend([pdf] * len(text_chunks))  # Track source

#         embeddings = self.generate_embeddings(all_texts)

#         np.savez(output_path, page_numbers=page_numbers, embeddings=np.array(embeddings))
#         print(f"Embeddings saved at: {output_path}")

# # Usage Example
# if __name__ == "__main__":
#     FILE_PATHS = [
#         "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 1.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 2.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 3.pdf",
#         "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 4.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 5.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 6.pdf",
#         "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 7.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 8.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 9.pdf",
#         "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 10.pdf", "1ኛ ክፍል ሒሳብ መጽሐፍ - ምዕራፍ 12.pdf"  # Add paths to all 12 books
#     ]
#     API_KEY = "REMOVED"
#     OUTPUT_FILE = "math_books_embeddings.npz"

#     embedder = MathBookEmbedder(FILE_PATHS, API_KEY)
#     embedder.process_books(OUTPUT_FILE)



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
