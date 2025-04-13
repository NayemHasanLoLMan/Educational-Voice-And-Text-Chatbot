import os
import fitz  # PyMuPDF for PDF processing
import numpy as np
import json
import openai
import logging
import time
import unicodedata
from typing import Dict, Tuple
import dotenv
import pytesseract
from PIL import Image
import io
import fasttext
import re

dotenv.load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler("pdf_embedding.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Create stream handler with UTF-8 encoding
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add handlers to logger
logger.handlers = [file_handler, stream_handler]

# Load fasttext language detection model
fasttext_model = None
try:
    fasttext_model_path = "lid.176.bin"
    if not os.path.exists(fasttext_model_path):
        raise FileNotFoundError(f"FastText model file not found at {fasttext_model_path}")
    fasttext_model = fasttext.load_model(fasttext_model_path)
    logger.info("FastText model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load FastText model: {str(e)}. Using Ethiopic script fallback for Amharic detection.")

class BatchPDFVectorizerOpenAI:
    def __init__(self, folder_path: str):
        """Initialize the BatchPDFVectorizer with a folder path and OpenAI API key from environment."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found at: {folder_path}")
        
        self.folder_path = folder_path
        self.embedding_model = "text-embedding-3-small"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if not openai.api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")
        
        self.embedding_dimension = None
        logger.info(f"Initialized OpenAI client with embedding model: {self.embedding_model}")

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from each page of a PDF file, including OCR for images."""
        page_dict = {}
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Try native text extraction
                text = page.get_text().strip()
                
                # If little or no text, try OCR
                if not text or len(text) < 50:
                    logger.debug(f"Low text content on page {page_num+1} of {pdf_path}. Attempting OCR.")
                    ocr_text = self.ocr_page(page, page_num)
                    text = ocr_text if ocr_text else text
                
                if text.strip():
                    page_dict[page_num + 1] = text.strip()
                else:
                    logger.warning(f"No usable text extracted for page {page_num+1} of {pdf_path}")
            
            doc.close()
            logger.info(f"Extracted {len(page_dict)} pages from {pdf_path}.")
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return page_dict

    def ocr_page(self, page, page_num: int) -> str:
        """Perform OCR on a PDF page by converting it to an image."""
        try:
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72))  # Reduced DPI to save memory
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Perform OCR with Tesseract, specifying Amharic and English
            text = pytesseract.image_to_string(img, lang="amh+eng", config='--psm 6')
            cleaned_text = text.strip()
            
            if cleaned_text:
                logger.debug(f"OCR extracted {len(cleaned_text)} characters for page {page_num+1}")
                return cleaned_text
            else:
                logger.warning(f"OCR found no text for page {page_num+1}")
                return ""
        except Exception as e:
            logger.error(f"OCR error on page {page_num+1}: {str(e)}")
            return ""

    def detect_language_and_normalize(self, text: str) -> str:
        """Detect language with fasttext and normalize if Amharic."""
        if not text or len(text.strip()) < 5:
            logger.debug("Text too short for language detection. Returning original.")
            return text
        
        # Check for Ethiopic script as a heuristic for Amharic
        ethiopic_pattern = re.compile(r'[\u1200-\u137F]')
        has_ethiopic = bool(ethiopic_pattern.search(text))
        
        if has_ethiopic:
            logger.debug("Detected Ethiopic script. Assuming Amharic.")
            return unicodedata.normalize("NFKC", text)
        
        # Use fasttext for language detection
        if fasttext_model:
            try:
                # Clean text for fasttext (replace newlines, multiple spaces)
                clean_text = " ".join(text.split())
                if len(clean_text) < 5:
                    logger.debug("Cleaned text too short for fasttext. Returning original.")
                    return text
                
                predictions = fasttext_model.predict(clean_text, k=1)
                lang = predictions[0][0].replace('__label__', '')
                
                if lang == "am":
                    logger.debug("Fasttext detected Amharic. Applying Unicode normalization.")
                    return unicodedata.normalize("NFKC", text)
                logger.debug(f"Fasttext detected language: {lang}")
                return text
            except Exception as e:
                logger.warning(f"Fasttext detection failed: {str(e)}. Assuming English.")
                return text
        else:
            logger.debug("Fasttext unavailable. Assuming English for non-Ethiopic text.")
            return text

    def get_embedding_with_retry(self, text: str, max_retries=3, backoff_factor=2):
        """Get embedding with retry logic for API failures."""
        for attempt in range(max_retries):
            try:
                response = openai.Embedding.create(
                    model=self.embedding_model,
                    input=text
                )
                # Use np.asarray to avoid NumPy 2.0 copy warning
                embedding = np.asarray(response['data'][0]['embedding'])
                
                if self.embedding_dimension is None:
                    self.embedding_dimension = embedding.shape[0]
                    logger.info(f"Detected embedding dimension: {self.embedding_dimension}")
                
                return embedding
            except Exception as e:
                wait_time = backoff_factor ** attempt
                logger.warning(f"Embedding API error (attempt {attempt+1}/{max_retries}): {str(e)}")
                logger.warning(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        logger.error(f"Failed to get embedding after {max_retries} attempts")
        if self.embedding_dimension is None:
            self.embedding_dimension = 1536  # Default for text-embedding-3-small
        return np.zeros(self.embedding_dimension)

    def vectorize_pages(self, pdf_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
        """Generate vector embeddings for each page using OpenAI Embedding API."""
        page_dict = self.extract_text_from_pdf(pdf_path)
        page_embeddings = {}

        logger.info(f"Starting vectorization for {pdf_path} with {len(page_dict)} pages")
        for page_num, text in page_dict.items():
            if len(text) < 10:
                logger.warning(f"Skipping Page {page_num} in {pdf_path}: content too short ({len(text)} chars)")
                continue

            try:
                logger.debug(f"Page {page_num} text sample: {text[:50]}")
                normalized_text = self.detect_language_and_normalize(text)
                if not normalized_text.strip():
                    logger.warning(f"Empty normalized text for Page {page_num} in {pdf_path}")
                    continue
                truncated_text = normalized_text[:8000]  # Approx token limit safe zone
                logger.debug(f"Processing Page {page_num}: {len(truncated_text)} chars")
                embedding = self.get_embedding_with_retry(truncated_text)
                if embedding is None or not embedding.any():
                    logger.error(f"Invalid embedding for Page {page_num} in {pdf_path}")
                    continue
                page_embeddings[page_num] = embedding
                logger.info(f"Generated embedding for Page {page_num} in {pdf_path}, shape: {embedding.shape}")
            except Exception as e:
                logger.error(f"Failed to process Page {page_num} in {pdf_path}: {str(e)}")
                continue
        
        logger.info(f"Completed vectorization for {pdf_path}: {len(page_embeddings)} embeddings generated")
        return page_embeddings, page_dict

    def process_all_files(self, base_output_dir: str):
        """Process PDFs in all subfolders and save each PDF's embeddings separately."""
        logger.info(f"Starting processing of PDFs in {self.folder_path}")
        for root, _, files in os.walk(self.folder_path):
            for file_name in files:
                if not file_name.lower().endswith(".pdf"):
                    continue

                file_path = os.path.join(root, file_name)
                # Clean relative path to avoid '.\' prefix
                relative_dir = os.path.relpath(root, self.folder_path)
                if relative_dir == ".":
                    relative_dir = ""
                output_subdir = os.path.join(base_output_dir, relative_dir)
                os.makedirs(output_subdir, exist_ok=True)

                output_filename = f"{os.path.splitext(file_name)[0]}_embedding.npz"
                output_path = os.path.join(output_subdir, output_filename)

                logger.info(f"Processing: {file_path}")
                try:
                    page_embeddings, page_texts = self.vectorize_pages(file_path)

                    if not page_embeddings:
                        logger.warning(f"No embeddings generated for {file_name}")
                        continue

                    file_embeddings = {}
                    file_metadata = {}

                    for page_num, embedding in page_embeddings.items():
                        page_id = f"page_{page_num}"
                        file_embeddings[page_id] = embedding
                        file_metadata[page_id] = {
                            "file_name": file_name,
                            "page_number": page_num,
                            "text": page_texts.get(page_num, "")[:1000],
                            "char_count": len(page_texts.get(page_num, ""))
                        }

                    self.save_combined_embeddings(file_embeddings, file_metadata, output_path)

                except Exception as e:
                    logger.error(f"Error processing {file_name}: {str(e)}")
                    continue
        logger.info("Completed processing all PDFs")

    def save_combined_embeddings(self, embeddings: Dict[str, np.ndarray], 
                                metadata: Dict[str, Dict], output_path: str):
        """Save all embeddings and metadata to a single file."""
        if not embeddings:
            raise ValueError("No embeddings to save")
            
        page_ids = list(embeddings.keys())
        first_dim = embeddings[page_ids[0]].shape[0]

        for page_id in page_ids:
            if embeddings[page_id].shape[0] != first_dim:
                logger.error(f"Inconsistent embedding dimensions: {page_id} has {embeddings[page_id].shape[0]}, expected {first_dim}")
                embeddings[page_id] = np.resize(embeddings[page_id], (first_dim,))

        embedding_array = np.stack([embeddings[page_id] for page_id in page_ids], axis=0)

        try:
            np.savez(
                output_path,
                page_ids=page_ids,
                embeddings=embedding_array,
                metadata=json.dumps(metadata, ensure_ascii=False)
            )
            if os.path.exists(output_path):
                size = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"Saved {len(page_ids)} page embeddings to {output_path} ({size:.2f} MB)")
            else:
                logger.error(f"File not created at {output_path}")
        except Exception as e:
            logger.error(f"Failed saving .npz file: {e}")
            raise

def main():
    FOLDER_PATH = r"D:\\Len project PDF\\Grade 5"
    OUTPUT_DIR = r"D:\\Len project PDF\\Grade 5 Embedded"

    try:
        vectorizer = BatchPDFVectorizerOpenAI(FOLDER_PATH)
        vectorizer.process_all_files(OUTPUT_DIR)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Critical error during vectorization: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()