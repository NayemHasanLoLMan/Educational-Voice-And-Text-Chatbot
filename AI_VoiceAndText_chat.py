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

# Download required corpora only if missing
try:
    nltk.data.find('corpora/brown')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('brown')
    nltk.download('punkt')

# Set your API keys
OPENAI_API_KEY = "REMOVED"
DEEPGRAM_API_KEY = "REMOVED"

openai.api_key = OPENAI_API_KEY

# Audio Recording Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512
AUDIO_FILENAME = "user_audio.wav"

# Memory for conversation
CONVERSATION_HISTORY = []
MAX_HISTORY = 6

# Initialize pygame mixer for console audio playback
pygame.mixer.init()

# Path to external knowledge base file
KNOWLEDGE_BASE_PATH = "all_text.txt"

# Ethiopian Education System Prompt with format example
ETHIOPIAN_EDUCATION_PROMPT = """
You are now an expert Ethiopian teacher trained in the national curriculum. Your job is to help Ethiopian students from Grade 1 to Grade 12 understand concepts clearly, kindly, and in their context.

🧠 Structure your answers this way:
1. Start with a friendly Amharic explanation.
2. Then translate the same content into English.
3. Use the format below:

አማርኛ መልስ፡ [Amharic explanation]
English: [English explanation]

💡 Example:

Q: What is 2 x 3?

አማርኛ መልስ፡ 2 በ 3 መባዛት ማለት ሁለትን ሶስት ጊዜ መድገም ነው። 2 + 2 + 2 = 6። ስለዚህ 2 x 3 = 6 ይሆናል።

English: 2 multiplied by 3 means adding 2 three times. 2 + 2 + 2 = 6. So, 2 x 3 = 6.

✅ Make sure answers:
- Are age-appropriate
- Use simple vocabulary
- Follow the Ethiopian curriculum
- Include examples and clarity
"""

class RecordingState:
    def __init__(self):
        self.recording = False
        self.stop_recording = False

recording_state = RecordingState()

def correct_spelling(text):
    try:
        return str(TextBlob(text).correct())
    except:
        return text

def load_knowledge_base(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            print(f"Loaded knowledge base ({len(content)} characters)")
            return content
        else:
            print(f"Knowledge base not found at {file_path}")
            return ""
    except Exception as e:
        print(f"Error loading knowledge base: {str(e)}")
        return ""

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

def chat_with_ai(user_input, external_knowledge):
    try:
        combined_prompt = ETHIOPIAN_EDUCATION_PROMPT
        if external_knowledge:
            combined_prompt += "\n\nAdditional Knowledge Base:\n" + external_knowledge

        messages = [{"role": "system", "content": combined_prompt}]
        for entry in CONVERSATION_HISTORY:
            messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": user_input})

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        full_response = response["choices"][0]["message"]["content"].strip()

        if "English:" in full_response:
            parts = full_response.split("English:", 1)
            amharic_text = parts[0].replace("አማርኛ መልስ፡", "").strip()
            english_text = parts[1].strip()
        else:
            amharic_text = full_response
            english_text = full_response

        CONVERSATION_HISTORY.append({"role": "user", "content": user_input})
        CONVERSATION_HISTORY.append({"role": "assistant", "content": amharic_text + "\n(English: " + english_text + ")"})

        while len(CONVERSATION_HISTORY) > MAX_HISTORY:
            CONVERSATION_HISTORY.pop(0)

        return amharic_text, english_text
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        return "ይቅርታ፣ መልሱ አልተሰጠም።", "Sorry, something went wrong."

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
    external_knowledge = load_knowledge_base(KNOWLEDGE_BASE_PATH)
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
            user_input = correct_spelling(user_input)

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
        am_text, en_text = chat_with_ai(user_input, external_knowledge)
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
