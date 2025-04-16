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
import threading
from concurrent.futures import ThreadPoolExecutor
import librosa
import numpy as np
import webrtcvad
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

openai.api_key = OPENAI_API_KEY

# Audio Recording Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512
AUDIO_FILENAME = "user_audio.wav"

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Initialize thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=3)

# Global variables
CONVERSATION_HISTORY = []
MAX_HISTORY = 15
TTS_CACHE = {}

# Teacher personality and system prompt
ENGLISH_TEACHER_PROMPT = """
            You are Sarah, a warm and friendly English conversation partner. Your primary goal is to help learners improve their spoken English by engaging them in natural, flowing, and encouraging conversations that feel like talking to a thoughtful friend.

            As Sarah, you should:
            1. Lead relaxed, everyday conversations on familiar topics (e.g., hobbies, routines, food, travel, goals, culture).
            2. Respond in a personal and friendly tone, showing genuine interest in what the student shares.
            3. Adapt your language based on the student’s level, gradually introducing more advanced vocabulary and sentence structures as the conversation progresses.
            4. Subtly correct important language mistakes by naturally modeling the correct usage in your next response—never call out or highlight errors directly.
            5. Ask open-ended, engaging follow-up questions to encourage the student to expand on their thoughts and speak more.
            6. Keep the conversation flowing smoothly—prioritize natural dialogue over formal instruction.
            7. Never interrupt with grammar explanations or corrections like “that’s wrong” or “you should say.”
            8. Use gentle rephrasing or mirroring to guide learners toward more natural and accurate English.
            9. Always be encouraging, supportive, and positive—make the student feel confident and relaxed.
            10. After 5–6 exchanges, start using slightly more complex vocabulary, phrasal verbs, or idiomatic expressions, but in a way that’s still accessible.
            11. If the student seems to struggle, revert to simpler language and topics to ensure they feel comfortable and engaged.
            12. If the student asks for help with a specific topic, provide a brief, friendly explanation and then return to the conversation.
            13. Naturally teach vocabulary and phrases through context, examples, and conversation rather than direct instruction.
            14. Naturally teach grammar and pronunciation through conversation, modeling correct usage without explicit correction.
            15. If the student seems to be struggling, gently guide them back to simpler language and topics to ensure they feel comfortable and engaged.  

            Remember:
            - You are a supportive conversation partner, not a traditional teacher.
            - Avoid formal teaching techniques or grammar lectures.
            - Your role is to guide learners toward fluency through meaningful, engaging conversation.
            - Focus on connection, comfort, and consistent improvement through subtle modeling and encouragement.
"""

# Recording State Management
class RecordingState:
    def __init__(self):
        self.recording = False
        self.stop_recording = False

recording_state = RecordingState()

# Utility Functions
def show_progress(message, stop_event):
    spinner = ['⣾', '⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽']
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r  {message} {spinner[i % len(spinner)]}   ")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write(f"\r{' ' * (len(message) + 10)}\r")
    sys.stdout.flush()

def record_audio_with_button():
    global recording_state
    print("  Press SPACE to start recording. Press SPACE again to stop.")
    keyboard.wait('space')
    print("  Recording started... Press SPACE to stop.")
    
    recording_state.recording = True
    recording_state.stop_recording = False
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    
    def stop_on_space():
        keyboard.wait('space')
        recording_state.stop_recording = True
    
    threading.Thread(target=stop_on_space, daemon=True).start()
    
    try:
        while recording_state.recording and not recording_state.stop_recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        recording_state.recording = False
    
    if len(frames) > 0:
        with wave.open(AUDIO_FILENAME, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
        print("  Recording stopped. Processing...")
        return AUDIO_FILENAME
    print("  No audio recorded.")
    return None

def transcribe_audio(file_path):
    url = "https://api.deepgram.com/v1/listen"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"}
    params = {
        "model": "nova-2",
        "language": "en",
        "punctuate": "true",
        "diarize": "false",
        "detect_language": "true"
    }
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print("  Audio file is missing or empty.")
            return "", None, []
        with open(file_path, "rb") as audio_file:
            response = requests.post(url, headers=headers, params=params, data=audio_file.read())
        if response.status_code == 200:
            result = response.json()
            transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            confidence = result["results"]["channels"][0]["alternatives"][0]["confidence"]
            words = result["results"]["channels"][0]["alternatives"][0].get("words", [])
            return transcript, confidence, words
        print(f"  Speech Recognition API Error: {response.status_code}")
        return "", None, []
    except Exception as e:
        print(f"  Error during transcription: {str(e)}")
        return "", None, []

def get_tts_audio(text):
    if text in TTS_CACHE:
        return TTS_CACHE[text]
    url = "https://api.openai.com/v1/audio/speech"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "tts-1", "input": text, "voice": "alloy", "response_format": "mp3", "speed": 1.0}
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            TTS_CACHE[text] = response.content
            return response.content
        print(f"  Text-to-speech API Error: {response.status_code}")
        return None
    except Exception as e:
        print(f"  Error in text-to-speech: {str(e)}")
        return None

def play_audio(audio_data):
    if not audio_data:
        return
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        try:
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    except Exception as e:
        print(f"  Error playing audio: {str(e)}")

def text_to_speech_async(text):
    future = executor.submit(get_tts_audio, text)
    def done_callback(future):
        audio_data = future.result()
        if audio_data:
            play_audio(audio_data)
    future.add_done_callback(done_callback)
    return future

# Analysis Functions
def analyze_speech(spoken_text, confidence, words):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""
                    You are analyzing English speech to help an AI conversation partner. 
                    Identify: 
                    1. Up to 3 key grammar or vocabulary errors (if any)
                    2. Any pronunciation issues based on word confidence scores below 0.6
                    3. Topic interests of the speaker
                    4. General fluency level
                    
                    Return only JSON:
                    {{
                        "errors": [{{"error": "", "correction": ""}}],
                        "pronunciation": [{{word: "", confidence: 0.0}}],
                        "interests": ["topic1", "topic2"],
                        "fluency_impression": ""
                    }}
                """},
                {"role": "user", "content": f"""
                    Spoken text: "{spoken_text}"
                    Transcription confidence: {confidence}
                    Word-level data: {json.dumps(words)}
                """}
            ],
            temperature=0.3
        )
        
        raw_content = response.choices[0].message.content.strip()
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            return {
                "errors": [],
                "pronunciation": [],
                "interests": ["general conversation"],
                "fluency_impression": "moderate"
            }
    except Exception as e:
        print(f"  Speech analysis error: {str(e)}")
        return {
            "errors": [],
            "pronunciation": [],
            "interests": ["general conversation"],
            "fluency_impression": "moderate"
        }

def analyze_audio_features(audio_file_path):
    try:
        y, sr = librosa.load(audio_file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        speech_rate = len(librosa.onset.onset_detect(y=y, sr=sr)) / duration if duration > 0 else 0
        
        # Use WebRTC VAD to check voice activity
        vad = webrtcvad.Vad(3)  # Aggressiveness level 3
        frame_duration = 30  # ms
        frame_size = int(sr * frame_duration / 1000)
        vad_frames = []
        
        for i in range(0, len(y) - frame_size, frame_size):
            frame = y[i:i+frame_size]
            # Convert to PCM16
            frame_pcm = (frame * 32768).astype(np.int16).tobytes()
            if len(frame_pcm) == frame_size * 2:  # 2 bytes per sample
                try:
                    vad_frames.append(vad.is_speech(frame_pcm, sr))
                except:
                    vad_frames.append(False)
        
        speech_percentage = sum(vad_frames) / len(vad_frames) if vad_frames else 0
        
        return {
            "duration": duration,
            "speech_rate": speech_rate,
            "speech_percentage": speech_percentage
        }
    except Exception as e:
        print(f"  Audio feature analysis error: {str(e)}")
        return {"duration": 0, "speech_rate": 0, "speech_percentage": 0}

def generate_conversation_context(audio_analysis, audio_features, exchanges):
    """Creates context for the AI to generate appropriate responses"""
    complexity_level = min(1 + (exchanges // 5), 5)  # Gradually increase complexity
    
    errors = audio_analysis.get("errors", [])
    pronunciation = audio_analysis.get("pronunciation", [])
    interests = audio_analysis.get("interests", ["general topics"])
    fluency = audio_analysis.get("fluency_impression", "moderate")
    
    context = {
        "complexity_level": complexity_level,  # 1-5 scale
        "errors_to_address": errors[:2],  # Limit to addressing just 2 errors
        "pronunciation_issues": pronunciation[:2],
        "interests": interests,
        "fluency_impression": fluency,
        "speech_rate": audio_features.get("speech_rate", 0),
        "speech_percentage": audio_features.get("speech_percentage", 0),
        "exchanges": exchanges
    }
    
    return context

def chat_with_english_partner(user_input, audio_file, confidence, words, exchanges):
    """Main function to generate AI responses"""
    # Analyze speech and audio
    audio_analysis = analyze_speech(user_input, confidence, words)
    audio_features = analyze_audio_features(audio_file)
    
    # Generate context for the AI
    context = generate_conversation_context(audio_analysis, audio_features, exchanges)
    
    # Build messages for the language model
    messages = [{"role": "system", "content": ENGLISH_TEACHER_PROMPT}]
    
    # Add context info for the AI
    context_message = f"""
    Conversation info (invisible to user):
    - Exchanges: {context['exchanges']}
    - Complexity level: {context['complexity_level']}/5
    - User errors: {json.dumps(context['errors_to_address'])}
    - Pronunciation issues: {json.dumps(context['pronunciation_issues'])}
    - User interests: {', '.join(context['interests'])}
    - Fluency impression: {context['fluency_impression']}
    
    Guidance:
    - Respond naturally to what they said
    - If they made grammar errors, subtly model the correct form in your response
    - Aim for natural correction through modeling, not explicit teaching
    - Ask engaging follow-up questions
    - Complexity should match level {context['complexity_level']}
    """
    messages.append({"role": "system", "content": context_message})
    
    # Add conversation history
    for entry in CONVERSATION_HISTORY:
        messages.append({"role": entry["role"], "content": entry["content"]})
    
    # Add user's latest input
    messages.append({"role": "user", "content": user_input})
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        assistant_response = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  API Error: {str(e)}")
        assistant_response = "I'm sorry, I didn't catch that. Could you say it again, please?"
    
    # Update conversation history
    CONVERSATION_HISTORY.append({"role": "user", "content": user_input})
    CONVERSATION_HISTORY.append({"role": "assistant", "content": assistant_response})
    
    # Keep history size manageable
    if len(CONVERSATION_HISTORY) > MAX_HISTORY:
        CONVERSATION_HISTORY.pop(0)
        CONVERSATION_HISTORY.pop(0)
    
    return assistant_response

def start_english_conversation():
    """Main function to run the conversation agent"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "="*70)
    print("  English Conversation Practice")
    print("  Press ESC at any time to exit")
    print("="*70)
    print("\n  Let's start a conversation!\n")
    
    exchanges = 0
    
    # Initial greeting to start conversation
    initial_greeting = "Hi there! I'm Sarah. It's nice to meet you! How are you doing today?"
    print(f"  Sarah: {initial_greeting}")
    initial_audio = text_to_speech_async(initial_greeting)
    
    try:
        while True:
            if keyboard.is_pressed('esc'):
                print("\n  Sarah: It was nice talking with you! Hope to chat again soon!")
                end_audio = text_to_speech_async("It was nice talking with you! Hope to chat again soon!")
                end_audio.result()
                break
            
            audio_file = record_audio_with_button()
            if not audio_file:
                print("  Sarah: I didn't hear anything. Let's try again!")
                text_to_speech_async("I didn't hear anything. Let's try again!")
                continue
            
            stop_event = threading.Event()
            progress_thread = threading.Thread(target=show_progress, args=("Processing speech", stop_event))
            progress_thread.start()
            user_input, confidence, words = transcribe_audio(audio_file)
            stop_event.set()
            progress_thread.join()
            
            if not user_input:
                print("  Sarah: Sorry, I didn't catch that. Could you say it again?")
                text_to_speech_async("Sorry, I didn't catch that. Could you say it again?")
                continue
            
            print(f"  You: {user_input}")
            
            stop_event = threading.Event()
            progress_thread = threading.Thread(target=show_progress, args=("Sarah is thinking", stop_event))
            progress_thread.start()
            response = chat_with_english_partner(user_input, audio_file, confidence, words, exchanges)
            stop_event.set()
            progress_thread.join()
            
            print(f"  Sarah: {response}")
            text_to_speech_async(response)
            
            exchanges += 1
            
            # Clean up audio file
            if os.path.exists(audio_file):
                os.remove(audio_file)
    
    except KeyboardInterrupt:
        print("\n  Conversation ended. Thanks for practicing!")
        farewell = "Thanks for practicing English with me! Goodbye!"
        print(f"  Sarah: {farewell}")
        farewell_audio = text_to_speech_async(farewell)
        farewell_audio.result()
    
    except Exception as e:
        print(f"  Unexpected error: {str(e)}")
    
    finally:
        initial_audio.result()
        executor.shutdown(wait=True)
        pygame.mixer.quit()

if __name__ == "__main__":
    try:
        start_english_conversation()
    except Exception as e:
        print(f"  Fatal error: {str(e)}")