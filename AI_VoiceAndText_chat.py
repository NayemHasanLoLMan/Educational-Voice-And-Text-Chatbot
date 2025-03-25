# import openai
# import requests
# import deepgram
# import pyaudio
# import wave
# import json
# import os
# import pygame
# import tempfile
# import sys
# import time
# import keyboard

# # Set your API keys
# OPENAI_API_KEY = "REMOVED"
# DEEPGRAM_API_KEY = "REMOVED"

# openai.api_key = OPENAI_API_KEY

# # Audio Recording Settings
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = 512
# AUDIO_FILENAME = "user_audio.wav"

# # Memory for conversation
# CONVERSATION_HISTORY = []
# MAX_HISTORY = 6  # 3 user messages and 3 AI responses

# # Initialize pygame mixer for console audio playback
# pygame.mixer.init()

# class RecordingState:
#     """Class to manage recording state."""
#     def __init__(self):
#         self.recording = False
#         self.stop_recording = False

# # Create a RecordingState instance
# recording_state = RecordingState()

# def record_audio_with_button():
#     """Records audio using spacebar to start/stop recording."""
#     global recording_state
    
#     print(" Press SPACE to start recording, then press SPACE again to stop...")
    
#     # Wait for spacebar to start recording
#     while not recording_state.recording:
#         time.sleep(0.1)
    
#     print(" Recording... (Press SPACE to stop)")
    
#     audio = pyaudio.PyAudio()
#     stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
#     frames = []
    
#     # Record until spacebar is pressed again
#     while recording_state.recording and not recording_state.stop_recording:
#         data = stream.read(CHUNK)
#         frames.append(data)
    
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()
    
#     # Reset flags
#     recording_state.recording = False
#     recording_state.stop_recording = False
    
#     # Only save if we have enough frames
#     if len(frames) > 20:  # Ensure we have at least a short utterance
#         with wave.open(AUDIO_FILENAME, "wb") as wf:
#             wf.setnchannels(CHANNELS)
#             wf.setsampwidth(audio.get_sample_size(FORMAT))
#             wf.setframerate(RATE)
#             wf.writeframes(b"".join(frames))
#         print(" Recording saved!")
#         return AUDIO_FILENAME
#     else:
#         print(" Recording too short, please try again")
#         return None

# def transcribe_audio(file_path):
#     """Transcribes recorded audio using Deepgram API."""
#     try:
#         if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
#             print(" Audio file is missing or empty.")
#             return ""

#         url = "https://api.deepgram.com/v1/listen"
#         headers = {
#             "Authorization": f"Token {DEEPGRAM_API_KEY}",
#             "Content-Type": "audio/wav"
#         }
        
#         with open(file_path, "rb") as audio_file:
#             response = requests.post(url, headers=headers, data=audio_file.read())
        
#         if response.status_code == 200:
#             result = response.json()
#             transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
#             print(f" You said: {transcript}")
#             return transcript
#         else:
#             print(f" Deepgram API Error: {response.status_code}", response.text)
#             return ""
            
#     except Exception as e:
#         print(f" Error during transcription: {str(e)}")
#         return ""

# def chat_with_ai(user_input):
#     """Gets a response from OpenAI's ChatGPT model with memory."""
#     try:
#         # Create messages array with conversation history
#         messages = []
        
#         # Add conversation history
#         for entry in CONVERSATION_HISTORY:
#             messages.append({"role": entry["role"], "content": entry["content"]})
        
#         # Add the current user message
#         messages.append({"role": "user", "content": user_input})
        
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=messages,
#             max_tokens=500,
#             temperature=0.7
#         )
        
#         assistant_response = response["choices"][0]["message"]["content"].strip()
        
#         # Update conversation history
#         CONVERSATION_HISTORY.append({"role": "user", "content": user_input})
#         CONVERSATION_HISTORY.append({"role": "assistant", "content": assistant_response})
            
#         return assistant_response
        
#     except Exception as e:
#         print(f" OpenAI API Error: {str(e)}")
#         return "Sorry, I couldn't process your request due to an API error."

# def text_to_speech(text):
#     """Converts text to speech using OpenAI TTS API and plays from console."""
#     url = "https://api.openai.com/v1/audio/speech"
#     headers = {
#         "Authorization": f"Bearer {OPENAI_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": "tts-1",
#         "input": text,
#         "voice": "alloy"
#     }

#     try:
#         response = requests.post(url, headers=headers, json=payload)
        
#         if response.status_code == 200:
#             print("🔊 AI Response (Speaking)...")
            
#             # Play directly using pygame
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
#                 temp_file.write(response.content)
#                 temp_path = temp_file.name
            
#             try:
#                 pygame.mixer.music.load(temp_path)
#                 pygame.mixer.music.play()
#                 # Wait for playback to finish
#                 while pygame.mixer.music.get_busy():
#                     pygame.time.Clock().tick(10)
#                 # Clean up
#                 pygame.mixer.music.unload()
#             finally:
#                 # Delete temp file
#                 if os.path.exists(temp_path):
#                     os.unlink(temp_path)
#         else:
#             print(" OpenAI TTS API Error:", response.text)
            
#     except Exception as e:
#         print(f" Error in text-to-speech: {str(e)}")

# def key_press_handler(e):
#     """Handles keyboard events."""
#     global recording_state
    
#     if e.name == 'space':
#         if not recording_state.recording:
#             recording_state.recording = True
#         else:
#             recording_state.stop_recording = True

# def chatbot():
#     """Main function for chatbot interaction with memory."""
#     print(" Welcome to the AI Chatbot!")
#     print("Choose a mode: (1) Voice Chat or (2) Text Chat")
#     print("You can change modes anytime by typing '/switch' or exit by typing '/exit'")
    
#     # Set up keyboard listener for recording
#     keyboard.on_press(key_press_handler)
    
#     try:
#         # Get initial mode
#         mode = input("\n Choose: (1) Voice Chat, (2) Text Chat: ").strip()
        
#         while True:
#             if mode not in ["1", "2"]:
#                 print(" Invalid choice. Please enter 1 for voice or 2 for text.")
#                 mode = input(" Choose: (1) Voice Chat, (2) Text Chat: ").strip()
#                 continue
            
#             # Voice mode
#             if mode == "1":
#                 print("\n(Say '/switch' to change mode or '/exit' to quit)")
#                 audio_file = record_audio_with_button()
#                 if audio_file is None:
#                     continue
                
#                 user_input = transcribe_audio(audio_file)
#                 if not user_input:
#                     print(" No speech detected or transcription failed. Please try again.")
#                     continue
                
#             # Text mode    
#             else:
#                 print("\n(Type '/switch' to change mode or '/exit' to quit)")
#                 user_input = input(" You: ")
#                 if not user_input.strip():
#                     print(" Empty input. Please try again.")
#                     continue
            
#             # Check for exit command
#             if user_input.lower().strip() in ["/exit"]:
#                 print(" Goodbye!")
#                 if mode == "1":  # Only do voice goodbye in voice mode
#                     text_to_speech("Goodbye! Have a great day!")
#                     while pygame.mixer.music.get_busy():
#                         pygame.time.Clock().tick(10)
#                 break
            
#             # Check for mode switch command
#             if user_input.lower().strip() == "/switch":
#                 mode = "2" if mode == "1" else "1"
#                 # Reset recording flags when switching modes
#                 recording_state.recording = False
#                 recording_state.stop_recording = False
#                 print(f"\nSwitching to {'Voice' if mode == '1' else 'Text'} mode!")
#                 time.sleep(0.5)  # Add small delay to ensure state reset
#                 continue
            
                
#             # Process user input
#             print(" Processing your request...")
#             response = chat_with_ai(user_input)
#             print(f" AI: {response}")
#             # Only do text-to-speech in voice mode
#             if mode == "1":
#                 text_to_speech(response)
            
#     except KeyboardInterrupt:
#         print("\n\n⚠️ Interrupted by user. Exiting...")
#     except Exception as e:
#         print(f"\nAn error occurred: {str(e)}")
#         print("Please try again or restart the application.")
#     finally:
#         # Clean up
#         keyboard.unhook_all()

# # Run the chatbot
# if __name__ == "__main__":
#     try:
#         chatbot()
#     except Exception as e:
#         print(f"Fatal error: {str(e)}")
#         sys.exit(1)
#     finally:
#         # Clean up pygame
#         pygame.mixer.quit()



import openai
import requests
import deepgram
import pyaudio
import wave
import json
import os
import pygame
import tempfile
import sys
import time
import keyboard

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
MAX_HISTORY = 6  # 3 user messages and 3 AI responses

# Initialize pygame mixer for console audio playback
pygame.mixer.init()

# Path to external knowledge base file
KNOWLEDGE_BASE_PATH = "all_text.txt"  # Place this file in the same directory as the script

# Ethiopian Education System Knowledge Prompt
ETHIOPIAN_EDUCATION_PROMPT = """
You are now an Ethiopian education expert and teacher. Answer questions based on the Ethiopian education system and curriculum standards. Key facts about the Ethiopian education system:

1. Structure: 
   - The 2-6-4-2 system: 2 years of pre-primary, 6 years of primary (Grades 1-6), 4 years of lower secondary (Grades 7-10), and 2 years of upper secondary (Grades 11-12).
   - Primary education is divided into first cycle (Grades 1-4) and second cycle (Grades 5-6).
   - Secondary education is divided into first cycle (Grades 7-10) and second cycle (Grades 11-12).

2. Curriculum:
   - Primary education focuses on languages (Amharic, English), mathematics, environmental science, arts, and physical education.
   - Lower secondary adds integrated sciences, social studies, civics, and additional languages.
   - Upper secondary offers specialization tracks: Natural Science (physics, chemistry, biology, mathematics) or Social Science (geography, history, economics).

3. Assessment:
   - Continuous assessment throughout the academic year.
   - National examinations at the end of Grade 10 (Ethiopian General Secondary Education Certificate Examination or EGSECE).
   - Grade 12 students take the Ethiopian Higher Education Entrance Certificate Examination (EHEECE).

4. Academic Calendar:
   - The academic year runs from September to June/July.
   - It's divided into two semesters with breaks in between.

5. Languages:
   - Amharic is often the language of instruction in early primary grades.
   - English becomes the primary language of instruction from Grade 7 onwards.
   - Regional languages may be used as instructional languages in some regions.

6. Values:
   - Education emphasizes cultural identity, civic responsibility, and national unity.
   - Focus on practical skills, problem-solving, and creativity.

When answering questions:
- Tailor responses to the appropriate grade level
- Use examples relevant to Ethiopian culture and context
- Follow the official Ethiopian curriculum standards
- Integrate character education and ethical values
- Be supportive and encouraging like an Ethiopian teacher
- Use assessment strategies common in Ethiopian schools

Additionally, refer to the detailed knowledge base provided to answer specific questions about Ethiopian curriculum topics, teaching methodologies, and subject matter content.

Always position yourself as a knowledgeable Ethiopian educator helping students understand concepts according to Ethiopian education standards and cultural context.
"""

class RecordingState:
    """Class to manage recording state."""
    def __init__(self):
        self.recording = False
        self.stop_recording = False

# Create a RecordingState instance
recording_state = RecordingState()

def load_knowledge_base(file_path):
    """Loads external knowledge base from a text file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            print(f" Successfully loaded knowledge base ({len(content)} characters)")
            return content
        else:
            print(f" Warning: Knowledge base file not found at {file_path}")
            return ""
    except Exception as e:
        print(f" Error loading knowledge base: {str(e)}")
        return ""

def record_audio_with_button():
    """Records audio using spacebar to start/stop recording."""
    global recording_state
    
    print(" Press SPACE to start recording, then press SPACE again to stop...")
    
    # Wait for spacebar to start recording
    while not recording_state.recording:
        time.sleep(0.1)
    
    print(" Recording... (Press SPACE to stop)")
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    frames = []
    
    # Record until spacebar is pressed again
    while recording_state.recording and not recording_state.stop_recording:
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Reset flags
    recording_state.recording = False
    recording_state.stop_recording = False
    
    # Only save if we have enough frames
    if len(frames) > 20:  # Ensure we have at least a short utterance
        with wave.open(AUDIO_FILENAME, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
        print(" Recording saved!")
        return AUDIO_FILENAME
    else:
        print(" Recording too short, please try again")
        return None

def transcribe_audio(file_path):
    """Transcribes recorded audio using Deepgram API."""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(" Audio file is missing or empty.")
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
            print(f" You said: {transcript}")
            return transcript
        else:
            print(f" Deepgram API Error: {response.status_code}", response.text)
            return ""
            
    except Exception as e:
        print(f" Error during transcription: {str(e)}")
        return ""

def chat_with_ai(user_input, external_knowledge):
    """Gets a response from OpenAI's ChatGPT model with memory and Ethiopian education context."""
    try:
        # Create combined prompt with Ethiopian education system info and external knowledge
        combined_prompt = ETHIOPIAN_EDUCATION_PROMPT
        
        # Add external knowledge if available
        if external_knowledge:
            combined_prompt += "\n\nAdditional Knowledge Base Information:\n" + external_knowledge
            
        # Create messages array with system prompt and conversation history
        messages = [{"role": "system", "content": combined_prompt}]
        
        # Add conversation history
        for entry in CONVERSATION_HISTORY:
            messages.append({"role": entry["role"], "content": entry["content"]})
        
        # Add the current user message
        messages.append({"role": "user", "content": user_input})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",  # Using larger context model to accommodate knowledge base
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        assistant_response = response["choices"][0]["message"]["content"].strip()
        
        # Update conversation history
        CONVERSATION_HISTORY.append({"role": "user", "content": user_input})
        CONVERSATION_HISTORY.append({"role": "assistant", "content": assistant_response})
        
        # Maintain max history length by removing oldest entries if needed
        while len(CONVERSATION_HISTORY) > MAX_HISTORY:
            CONVERSATION_HISTORY.pop(0)
            
        return assistant_response
        
    except Exception as e:
        print(f" OpenAI API Error: {str(e)}")
        return "Sorry, I couldn't process your request due to an API error."

def text_to_speech(text):
    """Converts text to speech using OpenAI TTS API and plays from console."""
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
            print("🔊 AI Response (Speaking)...")
            
            # Play directly using pygame
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            try:
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                # Clean up
                pygame.mixer.music.unload()
            finally:
                # Delete temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        else:
            print(" OpenAI TTS API Error:", response.text)
            
    except Exception as e:
        print(f" Error in text-to-speech: {str(e)}")

def key_press_handler(e):
    """Handles keyboard events."""
    global recording_state
    
    if e.name == 'space':
        if not recording_state.recording:
            recording_state.recording = True
        else:
            recording_state.stop_recording = True

def chatbot():
    """Main function for chatbot interaction with memory."""
    print(" Welcome to the Ethiopian Education AI Teacher!")
    print(" I'm here to help answer questions according to Ethiopian education standards.")
    print(" Choose a mode: (1) Voice Chat or (2) Text Chat")
    print(" You can change modes anytime by typing '/switch' or exit by typing '/exit'")
    
    # Load external knowledge base
    external_knowledge = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    if external_knowledge:
        print(" Ethiopian curriculum knowledge base loaded successfully!")
    else:
        print(" Warning: Operating with basic Ethiopian education system knowledge only.")
        print(f" To add specific knowledge, place your text file at: {os.path.abspath(KNOWLEDGE_BASE_PATH)}")
    
    # Set up keyboard listener for recording
    keyboard.on_press(key_press_handler)
    
    try:
        # Get initial mode
        mode = input("\n Choose: (1) Voice Chat, (2) Text Chat: ").strip()
        
        while True:
            if mode not in ["1", "2"]:
                print(" Invalid choice. Please enter 1 for voice or 2 for text.")
                mode = input(" Choose: (1) Voice Chat, (2) Text Chat: ").strip()
                continue
            
            # Voice mode
            if mode == "1":
                print("\n(Say '/switch' to change mode or '/exit' to quit)")
                audio_file = record_audio_with_button()
                if audio_file is None:
                    continue
                
                user_input = transcribe_audio(audio_file)
                if not user_input:
                    print(" No speech detected or transcription failed. Please try again.")
                    continue
                
            # Text mode    
            else:
                print("\n(Type '/switch' to change mode or '/exit' to quit)")
                user_input = input(" You: ")
                if not user_input.strip():
                    print(" Empty input. Please try again.")
                    continue
            
            # Check for exit command
            if user_input.lower().strip() in ["/exit"]:
                print(" Goodbye! Thank you for learning with the Ethiopian Education AI Teacher!")
                if mode == "1":  # Only do voice goodbye in voice mode
                    text_to_speech("Goodbye! Continue learning and being curious!")
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                break
            
            # Check for mode switch command
            if user_input.lower().strip() == "/switch":
                mode = "2" if mode == "1" else "1"
                # Reset recording flags when switching modes
                recording_state.recording = False
                recording_state.stop_recording = False
                print(f"\nSwitching to {'Voice' if mode == '1' else 'Text'} mode!")
                time.sleep(0.5)  # Add small delay to ensure state reset
                continue
                
            # Process user input
            print(" Processing your question as an Ethiopian teacher...")
            response = chat_with_ai(user_input, external_knowledge)
            print(f" Teacher: {response}")
            # Only do text-to-speech in voice mode
            if mode == "1":
                text_to_speech(response)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please try again or restart the application.")
    finally:
        # Clean up
        keyboard.unhook_all()

# Run the chatbot
if __name__ == "__main__":
    try:
        chatbot()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up pygame
        pygame.mixer.quit()