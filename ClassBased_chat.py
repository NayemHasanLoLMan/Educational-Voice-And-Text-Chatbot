import os
import numpy as np
import json
import openai
import logging
import unicodedata
import time
from typing import List, Dict
import dotenv
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import re

# Load environment variables
dotenv.load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler("chatbot.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Create stream handler
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

class ChatBot:
    def __init__(self, npz_dir: str, embedding_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        """Initialize the chatbot with the directory containing .npz files."""
        if not os.path.exists(npz_dir):
            raise FileNotFoundError(f"NPZ directory not found at: {npz_dir}")
        
        self.npz_dir = npz_dir
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai.api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")
        
        self.knowledge_base = self.load_knowledge_base()
        logger.info(f"ChatBot initialized with {len(self.knowledge_base)} entries from {npz_dir}")

    def load_knowledge_base(self) -> List[Dict]:
        """Load all .npz files into a knowledge base."""
        knowledge_base = []
        for root, _, files in os.walk(self.npz_dir):
            for file_name in files:
                if not file_name.lower().endswith(".npz"):
                    continue
                npz_path = os.path.join(root, file_name)
                try:
                    with np.load(npz_path, allow_pickle=True) as data:
                        page_ids = data['page_ids']
                        embeddings = data['embeddings']
                        # Convert NumPy array to string before parsing JSON
                        metadata_str = data['metadata'].item() if isinstance(data['metadata'], np.ndarray) else data['metadata']
                        metadata = json.loads(metadata_str)
                        for idx, page_id in enumerate(page_ids):
                            knowledge_base.append({
                                "page_id": page_id,
                                "embedding": embeddings[idx],
                                "metadata": metadata[page_id],
                                "npz_path": npz_path
                            })
                    logger.info(f"Loaded {len(page_ids)} entries from {npz_path}")
                except Exception as e:
                    logger.error(f"Error loading {npz_path}: {str(e)}")
        if not knowledge_base:
            raise ValueError("No valid .npz files found in the directory.")
        return knowledge_base

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

    def get_embedding_with_retry(self, text: str, max_retries: int = 3, backoff_factor: float = 2.0) -> np.ndarray:
        """Get embedding with retry logic for API failures."""
        for attempt in range(max_retries):
            try:
                response = openai.Embedding.create(
                    model=self.embedding_model,
                    input=text
                )
                return np.asarray(response['data'][0]['embedding'])
            except Exception as e:
                wait_time = backoff_factor ** attempt
                logger.warning(f"Embedding API error (attempt {attempt+1}/{max_retries}): {str(e)}")
                logger.warning(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        logger.error(f"Failed to get embedding after {max_retries} attempts")
        return np.zeros(1536)  # Fallback: assumes embedding dimension from text-embedding-3-small

    def query(self, user_input: str, top_k: int = 3) -> str:
        """Process a user query and return a response based on the knowledge base."""
        try:
            # Log input for debugging
            logger.debug(f"Processing query: {user_input[:50]}...")
            
            # Normalize and embed the query
            normalized_input = self.detect_language_and_normalize(user_input)
            query_embedding = self.get_embedding_with_retry(normalized_input)
            logger.info(f"Generated query embedding for: {user_input[:50]}...")

            # Compute cosine similarities
            embeddings = np.stack([entry["embedding"] for entry in self.knowledge_base])
            similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
            
            # Get top_k most similar entries
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_entries = [self.knowledge_base[idx] for idx in top_indices]
            top_scores = similarities[top_indices]

            # Build context from top entries
            context = ""
            for entry, score in zip(top_entries, top_scores):
                meta = entry["metadata"]
                context += (f"File: {meta['file_name']}, Page: {meta['page_number']}, "
                           f"Similarity: {score:.4f}\n{meta['text']}\n\n")
            logger.info(f"Retrieved {len(top_entries)} relevant pages for query")

            # Generate response using OpenAI chat model
            prompt = (
                "You are a helpful assistant with access to a knowledge base of PDF documents. "
                "Based on the following context, answer the user's query concisely and accurately. "
                "If the context doesn't provide enough information, say so and offer a general response. "
                f"User Query: {user_input}\n\nContext:\n{context}"
            )
            response = openai.ChatCompletion.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated response: {answer[:100]}...")

            return answer

        except Exception as e:
            logger.error(f"Error processing query '{user_input}': {str(e)}")
            return "Sorry, I encountered an error while processing your query. Please try again."

def main():
    OUTPUT_DIR = r"D:\\Len project PDF\\Grade 1 Embedded"

    try:
        # Initialize chatbot
        chatbot = ChatBot(OUTPUT_DIR)

        # Interactive chat loop
        print("Welcome to the PDF Knowledge Base ChatBot! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                print("Please enter a valid query.")
                continue
            
            response = chatbot.query(user_input)
            print(f"Bot: {response}")

    except Exception as e:
        logger.critical(f"Critical error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()