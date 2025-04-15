import openai
import requests
import json
import os
import tempfile
import sys
import time
import pygame
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import re
from textblob import TextBlob
from Levenshtein import distance
import fasttext
import nltk

# Initialize pygame mixer for audio playback
pygame.mixer.init()


# Download required corpora only if missing
try:
    nltk.data.find('corpora/brown')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('brown')
    nltk.download('punkt')

# Set your API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

openai.api_key = OPENAI_API_KEY

# Global variables
KNOWLEDGE_BASE = None
LAST_NPZ_DIR = None

# Ethiopian Education System Prompt
ETHIOPIAN_EDUCATION_PROMPT = """
You are an expert Ethiopian teacher for **{subject}** in **Grade {grade}**, following the national curriculum. Use the provided knowledge base file: **{knowledge_base_file}** to give accurate, detailed answers. Incorporate conversation history for context and explain concepts clearly for young learners.

🧠 Instructions:
1. **Use Knowledge Base**: Base answers on **{knowledge_base_file}**. Use multiple relevant pages for rich context. If insufficient, use general knowledge aligned with the Ethiopian curriculum for **{subject}**, **Grade {grade}**.
2. **Conversation History**: ALWAYS check previous messages to maintain context. If the user asks for more explanation, clarification, or uses words like "explain", "previous", or "details", provide more detailed information about the previous topic.
3. **Subject Restriction**: ONLY answer queries directly related to **{subject}**. For queries outside **{subject}** (e.g., math, science, or other subjects), respond with:
   - Amharic: "ይቅርታ፣ ይህ ጥያቄ ከ{subject} ውጪ ነው። እባክህ ስለ {subject} ጠይቅ።"
   - English: "Sorry, this question is outside {subject}. Please ask about {subject}."
4. **Response Structure**:
   - ALWAYS provide your response in two distinctly separate sections:
   - First section: Simple, friendly Amharic explanation, suitable for Grade {grade}.
   - Second section: Detailed English translation of the Amharic text (not just a direct translation but culturally appropriate).
   - ALWAYS format exactly as:
     አማርኛ መልስ፡ [Amharic explanation]
     
     English: [English explanation]
5. **Guidelines**:
   - Age-appropriate for Grade {grade}.
   - Simple Amharic vocabulary with Ethiopian cultural examples.
   - For vague queries like "explain" or "details", refer to the most recent topic and expand on it.
   - Stay within **{subject}**.
   - For greetings or general questions outside the curriculum, respond appropriately while maintaining your role as an Ethiopian teacher.
   - Avoid Western references and cultural examples.

💡 Follow-up Request Handling:
- When user asks "explain", "more details", or similar requests, always refer to your previous answer and elaborate with more information, examples, or context, but only if the topic is within **{subject}**.
- If the user mentions "previous answer" in any language, identify what was previously discussed and give a more detailed explanation, ensuring it remains within **{subject}**.

✅ Answers must:
- Be Grade {grade} appropriate
- Follow Ethiopian curriculum
- Use clear, simple examples
- Avoid Western references
- Provide detailed explanations when requested
- ALWAYS include both Amharic and English sections with the exact format specified above
- Use Markdown for formatting
"""

# Load fasttext language detection model if available
fasttext_model = None
try:
    fasttext_model_path = "lid.176.bin"
    if os.path.exists(fasttext_model_path):
        fasttext_model = fasttext.load_model(fasttext_model_path)
        print("FastText model loaded successfully.")
    else:
        print(f"FastText model not found at {fasttext_model_path}")
except Exception as e:
    print(f"Failed to load FastText model: {str(e)}. Using Ethiopic script fallback for Amharic detection.")

def correct_spelling(text):
    """Correct spelling only for non-Amharic text."""
    ethiopic_pattern = re.compile(r'[\u1200-\u137F]')
    if ethiopic_pattern.search(text):
        print("Amharic text detected. Skipping spelling correction.")
        return text
    
    educational_terms = ["amharic", "math", "mathematics", "science"]
    if len(text.split()) <= 2 or any(term in text.lower() for term in educational_terms):
        return text
    try:
        corrected = str(TextBlob(text).correct())
        if distance(text.lower(), corrected.lower()) > len(text) // 2:
            print(f"Warning: Significant correction from '{text}' to '{corrected}'. Using original.")
            return text
        return corrected
    except:
        return text

def detect_language_and_normalize(text: str) -> str:
    """Detect language with fasttext and normalize if Amharic."""
    if not text or len(text.strip()) < 5:
        print("Text too short for language detection. Returning original.")
        return text
    
    ethiopic_pattern = re.compile(r'[\u1200-\u137F]')
    has_ethiopic = bool(ethiopic_pattern.search(text))
    
    if has_ethiopic:
        print("Detected Ethiopic script. Assuming Amharic.")
        return unicodedata.normalize("NFKC", text)
    
    if fasttext_model:
        try:
            clean_text = " ".join(text.split())
            if len(clean_text) < 5:
                print("Cleaned text too short for fasttext. Returning original.")
                return text
            
            predictions = fasttext_model.predict(clean_text, k=1)
            lang = predictions[0][0].replace('__label__', '')
            
            if lang == "am":
                print("Fasttext detected Amharic. Applying Unicode normalization.")
                return unicodedata.normalize("NFKC", text)
            print(f"Fasttext detected language: {lang}")
            return text
        except Exception as e:
            print(f"Fasttext detection failed: {str(e)}. Assuming English.")
            return text
    else:
        print("Fasttext unavailable. Assuming English for non-Ethiopic text.")
        return text

def load_knowledge_base(npz_file: str) -> list:
    """Load a single .npz file into a knowledge base."""
    knowledge_base = []
    if not os.path.exists(npz_file):
        print(f"ERROR: NPZ file not found at: {npz_file}")
        return knowledge_base
    
    try:
        with np.load(npz_file, allow_pickle=True) as data:
            page_ids = data['page_ids']
            embeddings = data['embeddings']
            metadata_str = data['metadata'].item() if isinstance(data['metadata'], np.ndarray) else data['metadata']
            metadata = json.loads(metadata_str)
            for idx, page_id in enumerate(page_ids):
                knowledge_base.append({
                    "page_id": page_id,
                    "embedding": embeddings[idx],
                    "metadata": metadata[page_id],
                    "npz_path": npz_file
                })
        print(f"Loaded {len(page_ids)} entries from {npz_file}")
    except Exception as e:
        print(f"Error loading {npz_file}: {str(e)}")
    
    return knowledge_base

def get_embedding_with_retry(text: str, max_retries: int = 3, backoff_factor: float = 2.0) -> np.ndarray:
    """Get embedding with retry logic for API failures."""
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.asarray(response['data'][0]['embedding'])
        except Exception as e:
            wait_time = backoff_factor ** attempt
            print(f"Embedding API error (attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(wait_time)
    print(f"Failed to get embedding after {max_retries} attempts")
    return np.zeros(1536)

def extract_fidels(word: str) -> list:
    """Extract individual fidels from an Amharic word."""
    fidels = []
    for char in word:
        if '\u1200' <= char <= '\u137F':
            fidels.append(char)
    return fidels

def is_follow_up_question(user_input):
    """Determine if input is a follow-up question."""
    follow_up_indicators = ["explain", "detail", "previous", "privious", "tell me more", "elaborate", "expline"]
    return any(indicator in user_input.lower() for indicator in follow_up_indicators)

def handle_previous_reference(user_input, conversation_history):
    """Handle references to previous content in the conversation."""
    previous_terms = ["previous", "privious", "last", "before", "earlier", "ቀዳሚ", "ቀደም", "ያለፈ"]
    
    if any(term in user_input.lower() for term in previous_terms) and len(conversation_history) >= 4:
        # Get the last substantive QA pair
        last_question = None
        last_answer = None
        
        for i in range(len(conversation_history)-3, -1, -2):
            if conversation_history[i]["role"] == "user" and i+1 < len(conversation_history):
                last_question = conversation_history[i]["content"]
                last_answer = conversation_history[i+1]["content"]
                break
        
        if last_question and last_answer:
            return True, last_question, last_answer
    
    return False, None, None

def transcribe_audio(file_path, deepgram_api_key):
    """Transcribe audio using Deepgram API."""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print("Audio file missing or empty.")
            return ""

        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {deepgram_api_key}",
            "Content-Type": "audio/wav"
        }

        with open(file_path, "rb") as audio_file:
            response = requests.post(url, headers=headers, data=audio_file.read())

        if response.status_code == 200:
            result = response.json()
            transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            print(f"Transcribed: {transcript}")
            return transcript
        else:
            print("Deepgram Error:", response.text)
            return ""
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return ""

def text_to_speech(text, openai_api_key):
    """Generate speech from text using OpenAI TTS API."""
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": "alloy"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            return temp_path
        else:
            print("TTS API Error:", response.text)
            return None
    except Exception as e:
        print(f"TTS error: {str(e)}")
        return None

def process_query(user_input, conversation_history, subject, grade, npz_file, openai_api_key, top_k=3):
    """
    Process a user query and generate a response using the knowledge base.
    
    Args:
        user_input: String query from user
        conversation_history: List of conversation history
        subject: Subject being taught
        grade: Grade level
        npz_file: Path to NPZ knowledge base
        openai_api_key: OpenAI API key
        top_k: Number of relevant pages to use
    
    Returns:
        Tuple of (amharic_text, english_text)
    """
    global KNOWLEDGE_BASE, LAST_NPZ_DIR
    
    try:
        openai.api_key = openai_api_key
        
        # Reload knowledge base if npz_file changes
        if KNOWLEDGE_BASE is None or LAST_NPZ_DIR != npz_file:
            KNOWLEDGE_BASE = load_knowledge_base(npz_file)
            LAST_NPZ_DIR = npz_file
            
        # Format the prompt
        formatted_prompt = ETHIOPIAN_EDUCATION_PROMPT.format(
            subject=subject,
            grade=grade,
            knowledge_base_file=npz_file
        )

        # Check if this is a reference to previous content
        is_previous_reference, prev_question, prev_answer = handle_previous_reference(user_input, conversation_history)
        
        # Check if this is a follow-up question
        follow_up = is_follow_up_question(user_input)
        
        if is_previous_reference:
            # Make the context explicit to the model
            user_input = f"The user is asking for more details about their previous question: '{prev_question}' which you answered with: '{prev_answer}'. Please provide a more detailed explanation."
            print("Detected reference to previous content. Enhancing context.")
        elif follow_up and len(conversation_history) >= 2:
            # Find the last question to provide context
            last_question = None
            for entry in reversed(conversation_history):
                if entry["role"] == "user" and entry["content"] != user_input:
                    last_question = entry["content"]
                    break
            
            if last_question:
                user_input = f"The user asked '{user_input}' as a follow-up to their question: '{last_question}'. Please provide more detailed information about the previous topic."
                print("Detected follow-up question. Enhancing context.")

        # Handle specific tasks directly
        if "ፊደላት በቅደም ተከተል" in user_input:
            # Extract word from query (e.g., "ተማሪ")
            match = re.search(r'"([^"]+)"', user_input)
            if match:
                word = match.group(1)
                fidels = extract_fidels(word)
                fidel_str = ", ".join(fidels)
                amharic_text = f'ቃል "{word}" ፊደላት በቅደም ተከተል፡ {fidel_str} ናቸው።'
                english_text = f'The letters of the word "{word}" in order are: {fidel_str}.'
                return amharic_text, english_text

        # Normalize and embed the query
        normalized_input = detect_language_and_normalize(user_input)
        query_embedding = get_embedding_with_retry(normalized_input)
        print(f"Generated query embedding for: {user_input[:50]}...")

        # Compute cosine similarities if knowledge base is available
        context = ""
        if KNOWLEDGE_BASE:
            embeddings = np.stack([entry["embedding"] for entry in KNOWLEDGE_BASE])
            similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
            
            # Use top_k pages for broader context
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_entries = [KNOWLEDGE_BASE[idx] for idx in top_indices]
            top_scores = similarities[top_indices]

            # Build context from multiple pages
            for entry, score in zip(top_entries, top_scores):
                if score > 0.4:  # Threshold for relevance
                    meta = entry["metadata"]
                    context += f"Page {meta['page_number']}: {meta['text']}\n"
            print(f"Retrieved {len(top_entries)} relevant pages for query")
        else:
            print("No knowledge base available. Using general curriculum knowledge.")

        # Combine prompt with context
        combined_prompt = formatted_prompt
        if context:
            combined_prompt += "\n\nKnowledge Base Excerpt:\n" + context[:2500]  # Limit for token constraints
        else:
            combined_prompt += f"\n\nNote: Knowledge base unavailable. Use general knowledge of {subject} for Grade {grade}."

        # Add explicit instruction for greeting or general queries
        if user_input.lower().strip() in ["hello", "hi", "what do you know", "what do you know?"]:
            combined_prompt += f"\n\nUser has sent a greeting or general query. Respond appropriately as an Ethiopian Grade {grade} {subject} teacher while maintaining the required response format with both Amharic and English sections."

        # Prepare messages with conversation history
        messages = [{"role": "system", "content": combined_prompt}]
        
        # Include more conversation history for better context
        for entry in conversation_history[-6:]:  # Use last 6 entries for more context
            messages.append({"role": entry["role"], "content": entry["content"]})
        
        messages.append({"role": "user", "content": user_input})

        # Generate response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=2000,
            temperature=0.7
        )

        full_response = response["choices"][0]["message"]["content"].strip()

        # Improved parsing of Amharic and English parts
        amharic_text = ""
        english_text = ""
        
        # Check for the standard format first
        if "አማርኛ መልስ፡" in full_response and "English:" in full_response:
            parts = full_response.split("English:", 1)
            amharic_text = parts[0].replace("አማርኛ መልስ፡", "").strip()
            english_text = parts[1].strip()
        else:
            # Try alternative formats
            if "English:" in full_response:
                # Find where English starts
                english_start = full_response.find("English:")
                amharic_text = full_response[:english_start].strip()
                if "አማርኛ መልስ፡" in amharic_text:
                    amharic_text = amharic_text.replace("አማርኛ መልስ፡", "").strip()
                english_text = full_response[english_start+8:].strip()  # +8 for "English:"
            else:
                # Check for Ethiopic script
                ethiopic_pattern = re.compile(r'[\u1200-\u137F]')
                amharic_lines = []
                english_lines = []
                
                # Process line by line
                for line in full_response.split('\n'):
                    # If line contains Ethiopic characters
                    if ethiopic_pattern.search(line):
                        amharic_lines.append(line)
                    # If line is mostly Latin characters
                    elif re.search(r'[a-zA-Z]', line) and not ethiopic_pattern.search(line):
                        english_lines.append(line)
                    # If line has mixed content, determine based on character count
                    elif line.strip():
                        ethiopic_count = sum(1 for c in line if '\u1200' <= c <= '\u137F')
                        latin_count = sum(1 for c in line if c.isascii() and c.isalpha())
                        if ethiopic_count > latin_count:
                            amharic_lines.append(line)
                        else:
                            english_lines.append(line)
                
                amharic_text = ' '.join(amharic_lines).strip()
                english_text = ' '.join(english_lines).strip()
                
                # Fallback if nothing was found
                if not amharic_text:
                    amharic_text = "ይቅርታ፣ በአማርኛ መልስ መስጠት አልተቻለም።"
                if not english_text:
                    english_text = full_response if full_response else "Sorry, English translation not available."

        return amharic_text, english_text

    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        return (
            "ይቅርታ፣ መልሱን መስጠት አልተቻለም። ሌላ ጥያቄ ጠይቅ።",
            "Sorry, unable to provide an answer. Please ask another question."
        )

def process_text_chat(user_input, conversation_history, subject, grade, npz_file, openai_api_key):
    """
    Process text-based query for backend integration.
    
    Args:
        user_input: String query from user (Amharic or English)
        conversation_history: List of {"role": str, "content": str} for past conversation
        subject: Subject being taught
        grade: Grade level
        npz_file: Path to NPZ knowledge base
        openai_api_key: OpenAI API key
    
    Returns:
        Dict with:
            - amharic_response: Amharic answer
            - english_response: English answer
            - updated_history: Updated conversation history
            - error: Error message if any, else None
    """
    try:
        # Handle commands
        if user_input.lower().strip() == "/exit":
            return {
                "amharic_response": "ይቅርታ፣ ክፍለ ጊዜው ተጠናቋል።",
                "english_response": "Goodbye, session ended.",
                "updated_history": conversation_history,
                "error": None
            }
        
        if user_input.lower().strip() == "/switch":
            return {
                "amharic_response": "",
                "english_response": "",
                "updated_history": conversation_history,
                "error": "Switch command; no response generated"
            }
        
        # Correct spelling for non-Amharic text
        corrected_input = correct_spelling(user_input)
        
        # Process query
        amharic_text, english_text = process_query(
            corrected_input, 
            conversation_history, 
            subject, 
            grade, 
            npz_file, 
            openai_api_key
        )
        
        # Update conversation history
        updated_history = conversation_history.copy()
        updated_history.append({"role": "user", "content": corrected_input})
        updated_history.append({"role": "assistant", "content": f"አማርኛ መልስ፡ {amharic_text}\n\nEnglish: {english_text}"})
        
        # Keep history manageable
        max_history = 12
        while len(updated_history) > max_history:
            updated_history.pop(0)
        
        return {
            "amharic_response": amharic_text,
            "english_response": english_text,
            "updated_history": updated_history,
            "error": None
        }
    
    except Exception as e:
        print(f"Error in process_text_chat: {str(e)}")
        return {
            "amharic_response": "ይቅርታ፣ መልሱ አልተሰጠም።",
            "english_response": "Sorry, something went wrong.",
            "updated_history": conversation_history,
            "error": str(e)
        }

def process_voice_chat(audio_file_path, conversation_history, subject, grade, npz_file, openai_api_key, deepgram_api_key):
    """
    Process voice-based query for backend integration.
    
    Args:
        audio_file_path: Path to audio file (WAV format)
        conversation_history: List of {"role": str, "content": str} for past conversation
        subject: Subject being taught
        grade: Grade level
        npz_file: Path to NPZ knowledge base
        openai_api_key: OpenAI API key
        deepgram_api_key: Deepgram API key
    
    Returns:
        Dict with:
            - amharic_response: Amharic answer
            - english_response: English answer
            - updated_history: Updated conversation history
            - error: Error message if any, else None
            - transcription: Transcription of audio input
            - tts_audio_path: Path to TTS audio file or None
    """
    try:
        # Transcribe audio
        user_input = transcribe_audio(audio_file_path, deepgram_api_key)
        if not user_input:
            return {
                "amharic_response": "ይቅርታ፣ የድምፅ ግብዓት አልተገኘም።",
                "english_response": "Sorry, no audio input detected.",
                "updated_history": conversation_history,
                "error": "Empty transcription",
                "transcription": "",
                "tts_audio_path": None
            }
        
        # Handle commands
        if user_input.lower().strip() == "/exit":
            tts_path = text_to_speech("Goodbye! Keep learning!", openai_api_key)
            return {
                "amharic_response": "ይቅርታ፣ ክፍለ ጊዜው ተጠናቋል።",
                "english_response": "Goodbye, session ended.",
                "updated_history": conversation_history,
                "error": None,
                "transcription": user_input,
                "tts_audio_path": tts_path
            }
        
        if user_input.lower().strip() == "/switch":
            return {
                "amharic_response": "",
                "english_response": "",
                "updated_history": conversation_history,
                "error": "Switch command; no response generated",
                "transcription": user_input,
                "tts_audio_path": None
            }
        
        # Process query
        amharic_text, english_text = process_query(
            user_input, 
            conversation_history, 
            subject, 
            grade, 
            npz_file, 
            openai_api_key
        )
        
        # Update conversation history
        updated_history = conversation_history.copy()
        updated_history.append({"role": "user", "content": user_input})
        updated_history.append({"role": "assistant", "content": f"አማርኛ መልስ፡ {amharic_text}\n\nEnglish: {english_text}"})
        
        # Keep history manageable
        max_history = 12
        while len(updated_history) > max_history:
            updated_history.pop(0)
        
        # Generate TTS for English response
        tts_path = None
        if english_text:
            tts_path = text_to_speech(english_text, openai_api_key)
        
        return {
            "amharic_response": amharic_text,
            "english_response": english_text,
            "updated_history": updated_history,
            "error": None,
            "transcription": user_input,
            "tts_audio_path": tts_path
        }
    
    except Exception as e:
        print(f"Error in process_voice_chat: {str(e)}")
        return {
            "amharic_response": "ይቅርታ፣ መልሱ አልተሰጠም።",
            "english_response": "Sorry, something went wrong.",
            "updated_history": conversation_history,
            "error": str(e),
            "transcription": "",
            "tts_audio_path": None
        }

def main_api(request_data):
    """
    Main function for backend API integration.
    
    Args:
        request_data: Dict containing:
            - mode: "text" or "voice"
            - conversation_history: List of conversation history
            - user_input: Text input (for text mode)
            - audio_file_path: Path to audio file (for voice mode)
            - subject: Subject being taught
            - grade: Grade level
            - npz_file: Path to NPZ knowledge base
    
    Returns:
        Dict with response data
    """
    try:
        # Extract common parameters
        mode = request_data.get("mode", "text")
        conversation_history = request_data.get("conversation_history", [])
        subject = request_data.get("subject", "Amharic")
        grade = request_data.get("grade", "1")
        npz_file = request_data.get("npz_file", "")
        
        # Retrieve API keys from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return {"error": "OpenAI API key not found in environment variables"}
        
        if not npz_file or not os.path.exists(npz_file):
            return {
                "error": f"NPZ file not found at: {npz_file}"
            }
        
        # Process based on mode
        if mode == "text":
            user_input = request_data.get("user_input", "")
            if not user_input.strip():
                return {
                    "error": "User input is required for text mode"
                }
            
            return process_text_chat(
                user_input,
                conversation_history,
                subject,
                grade,
                npz_file,
                openai_api_key
            )
        
        elif mode == "voice":
            audio_file_path = request_data.get("audio_file_path", "")
            deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
            if not deepgram_api_key:
                return {"error": "Deepgram API key not found in environment variables"}
            
            if not audio_file_path or not os.path.exists(audio_file_path):
                return {
                    "error": f"Audio file not found at: {audio_file_path}"
                }
            
            return process_voice_chat(
                audio_file_path,
                conversation_history,
                subject,
                grade,
                npz_file,
                openai_api_key,
                deepgram_api_key
            )
        
        else:
            return {
                "error": f"Invalid mode: {mode}. Use 'text' or 'voice'."
            }
    
    except Exception as e:
        print(f"Error in main_api: {str(e)}")
        return {
            "error": f"Internal error: {str(e)}"
        }

# Example usage
if __name__ == "__main__":
    # Example request data for text mode
    text_request = {
        "mode": "text",
        "conversation_history": [],
        "user_input": "የትራፊክ መብራት አገልግሎቱ ምን ይመስላችኋል?",
        "subject": "Amharic",
        "grade": "1",
        "npz_file": r"D:\\Len project PDF\\Grade 1 Embedded\\grade 1-amharic_ethiofetenacom_8050_embedding.npz",
    }

    # Example request data for voice mode
    voice_request = {
        "mode": "voice",
        "conversation_history": [],
        "audio_file_path": "D:\\path\\to\\audio.wav",
        "subject": "Amharic",
        "grade": "1",
        "npz_file": r"D:\\Len project PDF\\Grade 1 Embedded\\grade 1-amharic_ethiofetenacom_8050_embedding.npz",
    }

    # Test text mode
    print("\nTesting text mode:")
    text_response = main_api(text_request)
    print("Text Response:", json.dumps(text_response, indent=2, ensure_ascii=False))

    # Test voice mode
    print("\nTesting voice mode:")
    voice_response = main_api(voice_request)
    print("Voice Response:", json.dumps(voice_response, indent=2, ensure_ascii=False))