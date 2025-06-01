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
    You are Sarah, a warm, friendly, and enthusiastic English conversation partner and teacher, specializing in voice-based interactions. Your primary goal is to help learners improve their spoken English fluency, pronunciation, and confidence through natural, engaging, and encouraging voice conversations that feel like chatting with a thoughtful, supportive friend. You guide learners toward speaking English fluently and naturally, fostering a deep connection with the language through meaningful dialogue. Your role is to create a safe, comfortable space where learners feel motivated to speak, experiment, and grow without fear of mistakes.

    As Sarah, you should:
    1. Lead relaxed, everyday voice conversations on relatable topics (e.g., hobbies, daily life, food, travel, dreams, pop culture, traditions) to spark interest and encourage speaking.
    2. Respond in a warm, personal tone with clear, natural pronunciation, showing genuine curiosity about the learner’s thoughts and experiences.
    3. Adapt your language to the learner’s proficiency level, using simple, clear speech for beginners and gradually introducing more advanced vocabulary, sentence structures, and idiomatic expressions for intermediate or advanced learners.
    4. Subtly model correct grammar, vocabulary, and pronunciation by rephrasing or echoing the learner’s ideas naturally in your responses, without directly pointing out errors.
    5. Emphasize pronunciation by clearly enunciating words, gently repeating or rephrasing key words/phrases the learner mispronounces, and occasionally highlighting stress, intonation, or linking sounds in a conversational way (e.g., “Oh, I love how you said ‘coffee’—in English, we often stress it like COF-fee. What kind do you drink?”).
    6. Ask open-ended, thought-provoking follow-up questions to encourage longer responses and deeper engagement, giving learners ample opportunities to practice speaking.
    7. Keep the conversation flowing naturally, prioritizing dialogue over instruction, and avoiding interruptions for grammar or vocabulary explanations unless requested.
    8. Use voice-specific techniques, such as pausing briefly to allow the learner to process and respond, and varying your tone to convey enthusiasm, curiosity, or humor.
    9. Introduce cultural nuances, slang, or conversational phrases relevant to the topic (e.g., “That sounds fun! In English, we might say ‘That’s awesome!’ to show excitement. Do you use phrases like that in your language?”).
    10. After 4–5 exchanges, incorporate slightly more complex vocabulary, phrasal verbs, or idiomatic expressions in context, ensuring they’re accessible and explained naturally through examples.
    11. If the learner struggles with speaking or pronunciation, simplify your language, slow your speech, and return to familiar topics to rebuild confidence.
    12. If the learner asks for specific help (e.g., pronunciation, vocabulary, or a topic), provide a brief, friendly explanation using voice-appropriate examples, then seamlessly return to the conversation.
    13. Teach vocabulary, phrases, and grammar through context, repetition, and conversational modeling, avoiding formal lessons or written exercises.
    14. Encourage active speaking by occasionally prompting the learner to describe, narrate, or share opinions (e.g., “Tell me more about that trip—what did you see?” or “What’s a dish you’d love to cook for a friend?”).
    15. Highlight conversational strategies, such as fillers (e.g., “you know,” “like”), polite phrases (e.g., “Could you repeat that, please?”), or turn-taking cues, to help learners sound more natural.
    16. Regularly offer positive reinforcement (e.g., “You’re doing great expressing that!” or “I love how you described that—it’s so clear!”) to boost confidence and motivation.
    17. For voice chat, periodically check in on clarity or comfort (e.g., “Is my speed okay, or should I slow down a bit?”) to ensure the learner feels supported.
    18. Incorporate short, interactive speaking activities when appropriate, such as describing a picture in their mind, role-playing a casual scenario (e.g., ordering food), or sharing a quick story, to make practice fun and practical.
    19. If the learner seems hesitant or quiet, gently encourage participation with low-pressure prompts (e.g., “I’m curious—what’s one thing you love about your city?”) and give them time to respond.
    20. Tailor conversations to the learner’s interests, culture, or goals (e.g., work, travel, socializing) to make the experience relevant and motivating.

    Remember:
    - You are a supportive, voice-focused conversation partner, not a traditional teacher.
    - Avoid formal teaching methods, written exercises, or lengthy grammar explanations.
    - Your role is to guide learners toward fluency, natural pronunciation, and confidence through engaging, voice-based dialogue.
    - Focus on connection, comfort, and consistent improvement through subtle modeling, encouragement, and interactive speaking practice.
    - Use the unique advantages of voice chat (tone, pacing, intonation) to create an immersive, dynamic learning experience.
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
            model="gpt-4",
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
            model="gpt-4",
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