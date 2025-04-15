import openai
import requests
import pyaudio
import wave
import json
import os
import pygame
import tempfile
import sys
import time
import keyboard
from textblob import TextBlob
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import fasttext
import re
from Levenshtein import distance

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

# Audio Recording Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512
AUDIO_FILENAME = "user_audio.wav"

# Memory for conversation
CONVERSATION_HISTORY = []
MAX_HISTORY = 12  # Increased for better context

# Initialize pygame mixer for console audio playback
pygame.mixer.init()

# Path to NPZ file
NPZ_FILE = r"D:\\Len project PDF\\Grade 1 Embedded\\grade 1-amharic_ethiofetenacom_8050_embedding.npz"

# Ethiopian Education System Prompt - Updated for better conversation handling
ETHIOPIAN_EDUCATION_PROMPT = """
You are an expert Ethiopian teacher for **{subject}** in **Grade {grade}**, following the national curriculum. Use the provided knowledge base file: **{knowledge_base_file}** to give accurate, detailed answers. Incorporate conversation history for context and explain concepts clearly for young learners.

🧠 Instructions:
1. **Use Knowledge Base**: Base answers on **{knowledge_base_file}**. Use multiple relevant pages for rich context. If insufficient, use general knowledge aligned with the Ethiopian curriculum for **{subject}**, **Grade {grade}**.
2. **Conversation History**: ALWAYS check previous messages to maintain context. If the user asks for more explanation, clarification, or uses words like "explain", "previous", or "details", provide more detailed information about the previous topic.
3. **Response Structure**:
   - ALWAYS provide your response in two distinctly separate sections:
   - First section: Simple, friendly Amharic explanation, suitable for Grade {grade}.
   - Second section: Detailed English translation of the Amharic text (not just a direct translation but culturally appropriate).
   - ALWAYS format exactly as:
     አማርኛ መልስ፡ [Amharic explanation]
     
     English: [English explanation]
4. **Guidelines**:
   - Age-appropriate for Grade {grade}.
   - Simple Amharic vocabulary with Ethiopian cultural examples.
   - For vague queries like "explain" or "details", refer to the most recent topic and expand on it.
   - When "explain" is requested, include more details about the previous answer.
   - Stay within **{subject}**.
   - For greetings or general questions outside the curriculum, respond appropriately while maintaining your role as an Ethiopian teacher.

💡 Follow-up Request Handling:
- When user asks "explain", "more details", or similar requests, always refer to your previous answer and elaborate with more information, examples, or context.
- If the user mentions "previous answer" in any language, identify what was previously discussed and give a more detailed explanation of it.

✅ Answers must:
- Be Grade {grade} appropriate
- Follow Ethiopian curriculum
- Use clear, simple examples
- Avoid Western references
- Avoid western culture references
- Provide detailed explanations when requested
- ALWAYS include both Amharic and English sections with the exact format specified above
"""

# Load fasttext language detection model
fasttext_model = None
try:
    fasttext_model_path = "lid.176.bin"
    if not os.path.exists(fasttext_model_path):
        raise FileNotFoundError(f"FastText model file not found at {fasttext_model_path}")
    fasttext_model = fasttext.load_model(fasttext_model_path)
    print("FastText model loaded successfully.")
except Exception as e:
    print(f"Failed to load FastText model: {str(e)}. Using Ethiopic script fallback for Amharic detection.")

class RecordingState:
    def __init__(self):
        self.recording = False
        self.stop_recording = False

recording_state = RecordingState()

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
    
    if not knowledge_base:
        print("No valid data found in the NPZ file.")
    else:
        print(f"Knowledge base loaded with {len(knowledge_base)} entries.")
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

def handle_previous_reference(user_input):
    """Handle references to previous content in the conversation."""
    previous_terms = ["previous", "privious", "last", "before", "earlier", "ቀዳሚ", "ቀደም", "ያለፈ"]
    
    if any(term in user_input.lower() for term in previous_terms) and len(CONVERSATION_HISTORY) >= 4:
        # Get the last substantive QA pair
        last_question = None
        last_answer = None
        
        for i in range(len(CONVERSATION_HISTORY)-3, -1, -2):
            if CONVERSATION_HISTORY[i]["role"] == "user" and i+1 < len(CONVERSATION_HISTORY):
                last_question = CONVERSATION_HISTORY[i]["content"]
                last_answer = CONVERSATION_HISTORY[i+1]["content"]
                break
        
        if last_question and last_answer:
            return True, last_question, last_answer
    
    return False, None, None

def is_follow_up_question(user_input):
    """Determine if input is a follow-up question."""
    follow_up_indicators = ["explain", "detail", "previous", "privious", "tell me more", "elaborate", "expline"]
    return any(indicator in user_input.lower() for indicator in follow_up_indicators)

def chat_with_ai(user_input: str, knowledge_base: list, subject: str, grade: str, knowledge_base_file: str, top_k: int = 3):
    """Process user input and generate a response using the NPZ knowledge base."""
    try:
        # Format the prompt
        formatted_prompt = ETHIOPIAN_EDUCATION_PROMPT.format(
            subject=subject,
            grade=grade,
            knowledge_base_file=knowledge_base_file
        )

        # Check if this is a reference to previous content
        is_previous_reference, prev_question, prev_answer = handle_previous_reference(user_input)
        
        # Check if this is a follow-up question
        follow_up = is_follow_up_question(user_input)
        
        if is_previous_reference:
            # Make the context explicit to the model
            user_input = f"The user is asking for more details about their previous question: '{prev_question}' which you answered with: '{prev_answer}'. Please provide a more detailed explanation."
            print("Detected reference to previous content. Enhancing context.")
        elif follow_up and len(CONVERSATION_HISTORY) >= 2:
            # Find the last question to provide context
            last_question = None
            for entry in reversed(CONVERSATION_HISTORY):
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
                
                # Update conversation history
                CONVERSATION_HISTORY.append({"role": "user", "content": user_input})
                CONVERSATION_HISTORY.append({
                    "role": "assistant", 
                    "content": f"አማርኛ መልስ፡ {amharic_text}\n\nEnglish: {english_text}"
                })
                
                return amharic_text, english_text

        # Normalize and embed the query
        normalized_input = detect_language_and_normalize(user_input)
        query_embedding = get_embedding_with_retry(normalized_input)
        print(f"Generated query embedding for: {user_input[:50]}...")

        # Compute cosine similarities if knowledge base is available
        context = ""
        if knowledge_base:
            embeddings = np.stack([entry["embedding"] for entry in knowledge_base])
            similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
            
            # Use top_k pages for broader context
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_entries = [knowledge_base[idx] for idx in top_indices]
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
            combined_prompt += "\n\nKnowledge Base Excerpt:\n" + context[:2500]  # Increased limit for multiple pages
        else:
            combined_prompt += f"\n\nNote: Knowledge base unavailable. Use general knowledge of {subject} for Grade {grade}."

        # Add explicit instruction for greeting or general queries
        if user_input.lower().strip() in ["hello", "hi", "what do you know", "what do you know?"]:
            combined_prompt += f"\n\nUser has sent a greeting or general query. Respond appropriately as an Ethiopian Grade {grade} {subject} teacher while maintaining the required response format with both Amharic and English sections."

        # Prepare messages with conversation history
        messages = [{"role": "system", "content": combined_prompt}]
        
        # Include more conversation history for better context
        for entry in CONVERSATION_HISTORY[-6:]:  # Use last 6 entries for more context
            messages.append({"role": entry["role"], "content": entry["content"]})
        
        messages.append({"role": "user", "content": user_input})

        # Generate response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=2000,  # Increased for detailed explanations
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

        # Update conversation history with the FULL response for better context
        CONVERSATION_HISTORY.append({"role": "user", "content": user_input})
        CONVERSATION_HISTORY.append({"role": "assistant", "content": full_response})

        while len(CONVERSATION_HISTORY) > MAX_HISTORY:
            CONVERSATION_HISTORY.pop(0)

        return amharic_text, english_text

    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        return (
            "ይቅርታ፣ መልሱን መስጠት አልተቻለም። ሌላ ጥያቄ ጠይቅ።",
            "Sorry, unable to provide an answer. Please ask another question."
        )

def record_audio_with_button():
    global recording_state
    print("Press SPACE to start recording, then press SPACE again to stop...")
    while not recording_state.recording:
        time.sleep(0.1)
    print("Recording... (Press SPACE to stop)")

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    while recording_state.recording and not recording_state.stop_recording:
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()

    recording_state.recording = False
    recording_state.stop_recording = False

    if len(frames) > 20:
        with wave.open(AUDIO_FILENAME, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
        return AUDIO_FILENAME
    else:
        print("Recording too short.")
        return None

def transcribe_audio(file_path):
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print("Audio file missing or empty.")
            return ""

        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        }

        with open(file_path, "rb") as audio_file:
            response = requests.post(url, headers=headers, data=audio_file.read())

        if response.status_code == 200:
            result = response.json()
            transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            print(f"You said: {transcript}")
            return transcript
        else:
            print("Deepgram Error:", response.text)
            return ""
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return ""

def text_to_speech(text):
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
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
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            os.unlink(temp_path)
        else:
            print("TTS API Error:", response.text)
    except Exception as e:
        print(f"TTS error: {str(e)}")

def key_press_handler(e):
    global recording_state
    if e.name == 'space':
        if not recording_state.recording:
            recording_state.recording = True
        else:
            recording_state.stop_recording = True

def chatbot():
    print("Welcome to the Ethiopian Education AI Teacher!")
    print("Choose a mode: (1) Voice Chat or (2) Text Chat")
    
    # Backend-provided subject, grade, and knowledge base file
    subject = "Amharic"
    grade = "1"
    knowledge_base_file = NPZ_FILE
    
    # Load NPZ knowledge base
    knowledge_base = load_knowledge_base(knowledge_base_file)
    
    keyboard.on_press(key_press_handler)
    mode = input("\nChoose: (1) Voice Chat, (2) Text Chat: ").strip()

    while True:
        if mode not in ["1", "2"]:
            print("Invalid choice.")
            mode = input("Choose: (1) Voice, (2) Text: ").strip()
            continue

        if mode == "1":
            print("\n(Say '/switch' or '/exit')")
            audio_file = record_audio_with_button()
            if audio_file is None:
                continue
            user_input = transcribe_audio(audio_file)
            if not user_input:
                continue
        else:
            user_input = input("\nYou: ")
            if not user_input.strip():
                continue
            print(f"Original input: {user_input}")
            corrected_input = correct_spelling(user_input)
            print(f"Corrected input: {corrected_input}")
            user_input = corrected_input

        if user_input.lower().strip() == "/exit":
            print("Goodbye!")
            if mode == "1":
                text_to_speech("Goodbye! Keep learning!")
            break

        if user_input.lower().strip() == "/switch":
            mode = "2" if mode == "1" else "1"
            recording_state.recording = False
            recording_state.stop_recording = False
            print(f"\nSwitching to {'Voice' if mode == '1' else 'Text'} mode!")
            time.sleep(0.5)
            continue

        print("Thinking like an Ethiopian teacher...")
        am_text, en_text = chat_with_ai(user_input, knowledge_base, subject, grade, knowledge_base_file)
        print(f"\U0001f469‍\U0001f3eb አማርኛ፡ {am_text}")
        print(f"(English: {en_text})")
        if mode == "1":
            text_to_speech(en_text)

if __name__ == "__main__": 
    try:
        chatbot()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
    finally:
        pygame.mixer.quit()