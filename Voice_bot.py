# import openai
# import requests
# import pyaudio
# import wave
# import json
# import os
# import pygame
# import tempfile
# import sys
# import time
# import keyboard
# import threading
# from concurrent.futures import ThreadPoolExecutor
# import librosa
# import numpy as np
# import webrtcvad

# # API Keys - Replace with your actual keys
# OPENAI_API_KEY = "REMOVED"
# SPEECH_RECOGNITION_API_KEY = "REMOVED"  # Deepgram API key

# # Audio Recording Settings
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = 512
# AUDIO_FILENAME = "user_audio.wav"

# # Initialize pygame mixer for audio playback
# pygame.mixer.init()

# # Initialize thread pool for concurrent processing
# executor = ThreadPoolExecutor(max_workers=3)

# # Global variables
# CONVERSATION_HISTORY = []
# MAX_HISTORY = 15
# TTS_CACHE = {}
# session_start_time = None
# current_level = None

# # Learning data storage
# LEARNING_DATA = {
#     'errors_made': [],
#     'vocabulary_introduced': [],
#     'lesson_topics': [],
#     'assessment_results': [],
#     'speaking_samples': [],
#     'improvement_areas': [],
#     'strengths': []
# }

# # English teacher system prompt
# ENGLISH_TEACHER_PROMPT = """
# You are Emma, a human-like English language teacher with a warm, encouraging personality. Your goal is to have natural conversations with the student to help them improve their English speaking skills. Act exactly like a real English teacher would in a one-on-one conversation session.

# As Emma, you should:
# 1. Lead the conversation naturally by asking engaging questions about topics interesting to the student
# 2. Listen to the student's responses and continue the conversation naturally
# 3. Provide subtle, constructive feedback on errors by modeling the correct language
# 4. Gauge the student's level through their responses and adjust your language accordingly
# 5. Introduce new vocabulary or expressions naturally within the conversation
# 6. Ask follow-up questions to encourage longer responses and deeper thinking
# 7. Provide positive reinforcement when the student communicates effectively
# 8. Keep the conversation flowing like a real human teacher would - avoid sounding like an AI
# 9. Incorporate audio feedback on pronunciation, fluency, and rhythm when provided

# IMPORTANT NEW FEATURES:
# 1. Mini-Lessons: After every 3-4 exchanges, introduce a short (1-2 minute) focused mini-lesson based on:
#    - Errors in text or audio (grammar, pronunciation)
#    - Relevant grammar or vocabulary for their level
#    - The mini-lesson should feel natural
# 2. Proficiency Testing: After every 5-6 exchanges, include a small test or challenge such as:
#    - Rephrasing with specific grammar
#    - Using recent vocabulary or correct pronunciation
#    - Always give feedback on performance
# 3. Progress Tracking: Track:
#    - Recurring errors (text and audio)
#    - Vocabulary introduced and used
#    - Strengths and improvement areas (including fluency, pronunciation)
# """

# # Recording State Management
# class RecordingState:
#     def __init__(self):
#         self.recording = False
#         self.stop_recording = False

# recording_state = RecordingState()

# # Utility Functions
# def show_progress(message, stop_event):
#     spinner = ['⣾', '⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽']
#     i = 0
#     while not stop_event.is_set():
#         sys.stdout.write(f"\r  {message} {spinner[i % len(spinner)]}   ")
#         sys.stdout.flush()
#         i += 1
#         time.sleep(0.1)
#     sys.stdout.write(f"\r{' ' * (len(message) + 10)}\r")
#     sys.stdout.flush()

# def record_audio_with_button():
#     global recording_state
#     print("  Press SPACE to start recording. Press SPACE again to stop.")
#     keyboard.wait('space')
#     print("  Recording started... Press SPACE to stop.")
    
#     recording_state.recording = True
#     recording_state.stop_recording = False
#     audio = pyaudio.PyAudio()
#     stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
#     frames = []
    
#     def stop_on_space():
#         keyboard.wait('space')
#         recording_state.stop_recording = True
    
#     threading.Thread(target=stop_on_space, daemon=True).start()
    
#     try:
#         while recording_state.recording and not recording_state.stop_recording:
#             data = stream.read(CHUNK, exception_on_overflow=False)
#             frames.append(data)
#     finally:
#         stream.stop_stream()
#         stream.close()
#         audio.terminate()
#         recording_state.recording = False
    
#     if len(frames) > 0:
#         with wave.open(AUDIO_FILENAME, "wb") as wf:
#             wf.setnchannels(CHANNELS)
#             wf.setsampwidth(audio.get_sample_size(FORMAT))
#             wf.setframerate(RATE)
#             wf.writeframes(b"".join(frames))
#         print("  Recording stopped. Processing...")
#         return AUDIO_FILENAME
#     print("  No audio recorded.")
#     return None

# def transcribe_audio(file_path):
#     url = "https://api.deepgram.com/v1/listen"
#     headers = {"Authorization": f"Token {SPEECH_RECOGNITION_API_KEY}", "Content-Type": "audio/wav"}
#     params = {
#         "model": "nova-2",
#         "language": "en",
#         "punctuate": "true",
#         "diarize": "false",
#         "detect_language": "true",
#         "utterances": "true",
#         "detect_topics": "true",
#         "summarize": "v2"
#     }
#     try:
#         if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
#             print("  Audio file is missing or empty.")
#             return "", None, []
#         with open(file_path, "rb") as audio_file:
#             response = requests.post(url, headers=headers, params=params, data=audio_file.read())
#         if response.status_code == 200:
#             result = response.json()
#             transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
#             confidence = result["results"]["channels"][0]["alternatives"][0]["confidence"]
#             words = result["results"]["channels"][0]["alternatives"][0].get("words", [])
#             return transcript, confidence, words
#         print(f"  Speech Recognition API Error: {response.status_code}")
#         return "", None, []
#     except Exception as e:
#         print(f"  Error during transcription: {str(e)}")
#         return "", None, []

# def get_tts_audio(text):
#     if text in TTS_CACHE:
#         return TTS_CACHE[text]
#     url = "https://api.openai.com/v1/audio/speech"
#     headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
#     payload = {"model": "tts-1", "input": text, "voice": "nova", "response_format": "mp3", "speed": 1.0}
#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         if response.status_code == 200:
#             TTS_CACHE[text] = response.content
#             return response.content
#         print(f"  Text-to-speech API Error: {response.status_code}")
#         return None
#     except Exception as e:
#         print(f"  Error in text-to-speech: {str(e)}")
#         return None

# def play_audio(audio_data):
#     if not audio_data:
#         return
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
#             temp_file.write(audio_data)
#             temp_path = temp_file.name
#         try:
#             pygame.mixer.music.load(temp_path)
#             pygame.mixer.music.play()
#             while pygame.mixer.music.get_busy():
#                 pygame.time.Clock().tick(10)
#             pygame.mixer.music.unload()
#         finally:
#             if os.path.exists(temp_path):
#                 os.unlink(temp_path)
#     except Exception as e:
#         print(f"  Error playing audio: {str(e)}")

# def text_to_speech_async(text):
#     future = executor.submit(get_tts_audio, text)
#     def done_callback(future):
#         audio_data = future.result()
#         if audio_data:
#             play_audio(audio_data)
#     future.add_done_callback(done_callback)
#     return future

# # Language and Audio Analysis Functions
# def analyze_language_level(text):
#     try:
#         messages = [
#             {"role": "system", "content": "Analyze the text and determine the English proficiency level (beginner, intermediate, advanced) based on vocabulary, grammar, and complexity. Respond with ONLY the level word."},
#             {"role": "user", "content": text}
#         ]
#         response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=10, temperature=0.3)
#         level = response.choices[0].message.content.strip().lower()
#         if "beginner" in level:
#             return "beginner"
#         elif "advanced" in level:
#             return "advanced"
#         return "intermediate"
#     except Exception as e:
#         print(f"  Error in language level analysis: {str(e)}")
#         return "intermediate"

# def analyze_language_errors(text, level):
#     try:
#         messages = [
#             {"role": "system", "content": f"Analyze the text from a {level} level English learner and identify up to 3 grammar or vocabulary errors. Reply in JSON format with array of objects with 'error_type', 'error', and 'correction' fields."},
#             {"role": "user", "content": text}
#         ]
#         response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=250, temperature=0.3)
#         return json.loads(response.choices[0].message.content.strip())
#     except Exception as e:
#         print(f"  Error in language error analysis: {str(e)}")
#         return []

# def analyze_speech(spoken_text, confidence, audio_file_path, words):
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": f"""
#                     You are an English teacher analyzing speech. Focus on the student's level: {current_level}.
#                     Use word-level confidence scores to identify pronunciation issues (below 0.9).
#                     Use transcription confidence ({confidence}) for overall clarity if below 0.95.
#                     Provide feedback on pronunciation, fluency, and accent.
#                     All scores between 0.0 and 1.0.
#                 """},
#                 {"role": "user", "content": f"""
#                     Spoken text: "{spoken_text}"
#                     Transcription confidence: {confidence}
#                     Audio file: {audio_file_path}
#                     Word-level data: {json.dumps(words)}
#                     Learner level: {current_level}
#                     Provide analysis in JSON:
#                     {{
#                         "pronunciation": {{"mispronounced_words": [{{"word": "", "suggestion": ""}}], "score": 0.0, "feedback": ""}},
#                         "fluency": {{"issues": [], "score": 0.0, "feedback": ""}},
#                         "accent": {{"patterns": [], "feedback": ""}},
#                         "overall": {{"score": 0.0, "strengths": [], "areas_for_improvement": []}}
#                     }}
#                 """}
#             ],
#             temperature=0.7
#         )
#         analysis = json.loads(response.choices[0].message.content.strip())
#         print(f"  Debug: audio_analysis = {json.dumps(analysis)}")
#         return analysis
#     except Exception as e:
#         print(f"  Speech analysis error: {str(e)}")
#         return {
#             "pronunciation": {"mispronounced_words": [], "score": 0.5, "feedback": "Keep practicing pronunciation!"},
#             "fluency": {"issues": [], "score": 0.5, "feedback": "You're doing fine, keep going!"},
#             "accent": {"patterns": [], "feedback": "Your accent is understandable!"},
#             "overall": {"score": 0.5, "strengths": ["Effort"], "areas_for_improvement": ["Clarity"]}
#         }

# def analyze_audio_features(audio_file_path):
#     try:
#         y, sr = librosa.load(audio_file_path, sr=None)
#         duration = librosa.get_duration(y=y, sr=sr)
#         f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
#         f0_cleaned = f0[~np.isnan(f0)]
#         mean_f0 = np.mean(f0_cleaned) if len(f0_cleaned) > 0 else 0
#         std_f0 = np.std(f0_cleaned) if len(f0_cleaned) > 0 else 0
#         rms = librosa.feature.rms(y=y)[0]
#         mean_rms = np.mean(rms)
#         speech_rate = len(librosa.onset.onset_detect(y=y, sr=sr)) / duration if duration > 0 else 0
#         return {
#             "pitch": {"mean": mean_f0, "variation": std_f0},
#             "volume": {"mean": mean_rms},
#             "speech_rate": speech_rate,
#             "duration": duration
#         }
#     except Exception as e:
#         print(f"  Audio feature analysis error: {str(e)}")
#         return {"pitch": {"mean": 0, "variation": 0}, "volume": {"mean": 0}, "speech_rate": 0, "duration": 0}

# def generate_lesson_plan(session_data, recent_exchanges, audio_analysis):
#     recent_text = " ".join([entry["content"] for entry in recent_exchanges])
#     prompt = f"""
#     Create a 1-2 min mini-lesson for a {session_data['level']} student based on:
#     - Recent conversation: {recent_text}
#     - Audio analysis: {json.dumps(audio_analysis)}
#     Focus on grammar, vocab, or pronunciation errors if present. Format as JSON:
#     {{'topic': '', 'explanation': '', 'examples': '', 'practice': ''}}
#     """
#     try:
#         response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": prompt}], max_tokens=350)
#         lesson = json.loads(response.choices[0].message.content.strip())
#         LEARNING_DATA['lesson_topics'].append(lesson['topic'])
#         return lesson
#     except Exception as e:
#         print(f"  Error in lesson plan generation: {str(e)}")
#         return {"topic": "Conversation practice", "explanation": "Let's keep practicing.", "examples": "We've been chatting!", "practice": "Tell me more!"}

# def create_proficiency_test(session_data):
#     recent_lessons = LEARNING_DATA['lesson_topics'][-3:] if LEARNING_DATA['lesson_topics'] else ["general conversation"]
#     prompt = f"Recent lessons: {', '.join(recent_lessons)}\nStudent level: {session_data['level']}"
#     try:
#         messages = [
#             {"role": "system", "content": "Create a brief (1-2 question) conversational test. Format as JSON with 'test_topic', 'questions', 'expected_responses'."},
#             {"role": "user", "content": prompt}
#         ]
#         response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=300, temperature=0.5)
#         return json.loads(response.choices[0].message.content.strip())
#     except Exception as e:
#         print(f"  Error in proficiency test creation: {str(e)}")
#         return {"test_topic": "Basic conversation", "questions": ["Tell me about your day."], "expected_responses": ["Any coherent response"]}

# def evaluate_test_response(test, user_response):
#     try:
#         evaluation_prompt = f"Test topic: {test['test_topic']}\nQuestion: {test['questions'][0]}\nExpected: {test['expected_responses'][0]}\nUser response: {user_response}"
#         messages = [
#             {"role": "system", "content": "Evaluate the response. Provide score (1-5) and feedback in JSON: {'score': 0, 'feedback': '', 'strengths': '', 'areas_to_improve': ''}"},
#             {"role": "user", "content": evaluation_prompt}
#         ]
#         response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=250, temperature=0.3)
#         evaluation = json.loads(response.choices[0].message.content.strip())
#         LEARNING_DATA['assessment_results'].append({'topic': test['test_topic'], 'score': evaluation['score'], 'feedback': evaluation['feedback']})
#         return evaluation
#     except Exception as e:
#         print(f"  Error in test response evaluation: {str(e)}")
#         return {"score": 3, "feedback": "Good effort!", "strengths": "Participation", "areas_to_improve": "Details"}

# def generate_learning_report():
#     session_data = {
#         'duration': int((time.time() - session_start_time) / 60),
#         'exchanges': len(CONVERSATION_HISTORY) // 2,
#         'level': current_level,
#         'topics': LEARNING_DATA['lesson_topics'],
#         'errors': LEARNING_DATA['errors_made'],
#         'speaking_samples': LEARNING_DATA['speaking_samples'],
#         'strengths': LEARNING_DATA['strengths'],
#         'improvement_areas': LEARNING_DATA['improvement_areas']
#     }
    
#     avg_pronunciation = np.mean([s['audio_analysis']['pronunciation']['score'] for s in session_data['speaking_samples']]) if session_data['speaking_samples'] else 0
#     avg_fluency = np.mean([s['audio_analysis']['fluency']['score'] for s in session_data['speaking_samples']]) if session_data['speaking_samples'] else 0
#     avg_speech_rate = np.mean([s['audio_features']['speech_rate'] for s in session_data['speaking_samples']]) if session_data['speaking_samples'] else 0
    
#     prompt = f"""
#     Generate a concise report:
#     - Duration: {session_data['duration']} min
#     - Exchanges: {session_data['exchanges']}
#     - Level: {session_data['level']}
#     - Topics: {', '.join(session_data['topics'])}
#     - Errors: {len(session_data['errors'])} (e.g., {session_data['errors'][:2] if session_data['errors'] else 'None'})
#     - Pronunciation Score: {avg_pronunciation:.2f}
#     - Fluency Score: {avg_fluency:.2f}
#     - Avg Speech Rate: {avg_speech_rate:.1f} words/sec
#     - Strengths: {', '.join(session_data['strengths'])}
#     - Improvement Areas: {', '.join(session_data['improvement_areas'])}
#     Include:
#     1. Session Summary
#     2. Key Analytics (errors, pronunciation, fluency)
#     3. Improvement Areas
#     4. Progress Made
#     """
#     try:
#         response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "system", "content": prompt}], max_tokens=500)
#         report = response.choices[0].message.content.strip()
#         timestamp = time.strftime("%Y%m%d-%H%M%S")
#         report_filename = f"english_learning_report_{timestamp}.txt"
#         with open(report_filename, "w") as f:
#             f.write(report)
#         print(f"\n  Learning report saved as: {report_filename}")
#         return report
#     except Exception as e:
#         print(f"  Error generating report: {str(e)}")
#         return "Report generation failed."

# def chat_with_english_teacher(user_input, session_data, audio_file, confidence, words):
#     global LEARNING_DATA
    
#     errors = analyze_language_errors(user_input, session_data['level'])
#     if errors:
#         LEARNING_DATA['errors_made'].extend([e for e in errors if e not in LEARNING_DATA['errors_made']])
    
#     audio_analysis = analyze_speech(user_input, confidence, audio_file, words)
#     audio_features = analyze_audio_features(audio_file)
#     LEARNING_DATA['speaking_samples'].append({
#         'text': user_input[:150],
#         'exchange_num': session_data['exchanges'],
#         'audio_analysis': audio_analysis,
#         'audio_features': audio_features
#     })
    
#     insert_lesson = session_data['exchanges'] % 4 == 0 and session_data['exchanges'] > 0
#     insert_test = session_data['exchanges'] % 6 == 0 and session_data['exchanges'] > 0 and not insert_lesson
    
#     messages = [{"role": "system", "content": ENGLISH_TEACHER_PROMPT}]
#     context_message = f"""
#     Session info:
#     - Level: {session_data['level']}
#     - Duration: {session_data['duration']} min
#     - Topics: {', '.join(session_data['topics'])}
#     - Exchanges: {session_data['exchanges']}
#     - Errors: {json.dumps(LEARNING_DATA['errors_made'][-3:])}
#     - Audio: Pronunciation {audio_analysis['pronunciation']['score']}, Fluency {audio_analysis['fluency']['score']}, Rate {audio_features['speech_rate']:.1f}
#     {f'INSERT MINI-LESSON: Yes' if insert_lesson else ''}
#     {f'INSERT TEST: Yes' if insert_test else ''}
#     """
#     messages.append({"role": "system", "content": context_message})
    
#     if insert_lesson:
#         recent_exchanges = CONVERSATION_HISTORY[-6:]
#         lesson_plan = generate_lesson_plan(session_data, recent_exchanges, audio_analysis)
#         messages.append({"role": "system", "content": f"INSERT MINI-LESSON: {json.dumps(lesson_plan)}"})
    
#     if insert_test:
#         test = create_proficiency_test(session_data)
#         messages.append({"role": "system", "content": f"INSERT TEST: {json.dumps(test)}"})
#         session_data['current_test'] = test
    
#     for entry in CONVERSATION_HISTORY:
#         messages.append({"role": entry["role"], "content": entry["content"]})
#     messages.append({"role": "user", "content": user_input})
    
#     try:
#         response = openai.ChatCompletion.create(model="gpt-4", messages=messages, max_tokens=400, temperature=0.7)
#         assistant_response = response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"  API Error: {str(e)}")
#         assistant_response = "Sorry, I had trouble processing that. Let's keep going!"
    
#     mispronounced = audio_analysis["pronunciation"].get("mispronounced_words", [])
#     if isinstance(mispronounced, list) and mispronounced and all(isinstance(w, dict) and 'word' in w and 'suggestion' in w for w in mispronounced):
#         pronunciation_feedback = "I noticed pronunciation points: " + ", ".join(
#             [f"'{w['word']}' - try {w['suggestion']}" for w in mispronounced]
#         )
#         assistant_response += " " + pronunciation_feedback
    
#     CONVERSATION_HISTORY.append({"role": "user", "content": user_input})
#     CONVERSATION_HISTORY.append({"role": "assistant", "content": assistant_response})
#     session_data['exchanges'] += 1
#     return assistant_response

# def voice_english_teacher():
#     print("Debug: Entering voice_english_teacher")
#     global session_start_time, current_level
#     session_data = {'level': 'intermediate', 'duration': 0, 'topics': [], 'exchanges': 0, 'start_time': time.time(), 'current_test': None}
#     session_start_time = time.time()
#     current_level = 'intermediate'
    
#     os.system('cls' if os.name == 'nt' else 'clear')
#     print("Debug: After clearing screen")
#     print("\n" + "="*70)
#     print("  Welcome to Your Enhanced Voice English Conversation Practice!")
#     print("  I'm Emma, your English teacher for today.")
#     print("="*70)
#     print("\n  Let's start!\n")
    
#     initial_greeting = "Hi! I'm Emma, here to help you with your English. We'll chat, and I'll give you tips on speaking, including pronunciation and fluency. Tell me about yourself and what you'd like to practice!"
#     print(f"  Emma: {initial_greeting}")
#     initial_audio = text_to_speech_async(initial_greeting)
    
#     try:
#         while True:
#             if keyboard.is_pressed('esc'):
#                 print("\n  Emma: Thanks for practicing with me!")
#                 end_audio = text_to_speech_async("Thanks for practicing! Here's your report.")
#                 end_audio.result()
                
#                 stop_event = threading.Event()
#                 progress_thread = threading.Thread(target=show_progress, args=("Generating report", stop_event))
#                 progress_thread.start()
#                 report = generate_learning_report()
#                 stop_event.set()
#                 progress_thread.join()
                
#                 print("\n" + "="*70)
#                 print("  YOUR ENGLISH LEARNING REPORT")
#                 print("="*70)
#                 print(report)
#                 print("="*70)
#                 break
            
#             audio_file = record_audio_with_button()
#             if not audio_file:
#                 print("  Emma: I didn’t hear anything. Try again!")
#                 text_to_speech_async("I didn’t hear anything. Try again!")
#                 continue
            
#             stop_event = threading.Event()
#             progress_thread = threading.Thread(target=show_progress, args=("Processing speech", stop_event))
#             progress_thread.start()
#             user_input, confidence, words = transcribe_audio(audio_file)
#             stop_event.set()
#             progress_thread.join()
            
#             if not user_input:
#                 print("  Emma: I didn’t catch that. Try again?")
#                 text_to_speech_async("I didn’t catch that. Try again?")
#                 continue
            
#             print(f"  You: {user_input}")
#             session_data['duration'] = int((time.time() - session_data['start_time']) / 60)
            
#             if session_data['exchanges'] == 0:
#                 current_level = analyze_language_level(user_input)
#                 session_data['level'] = current_level
#                 print(f"  (Detected level: {current_level})")
            
#             stop_event = threading.Event()
#             progress_thread = threading.Thread(target=show_progress, args=("Emma is thinking", stop_event))
#             progress_thread.start()
#             response = chat_with_english_teacher(user_input, session_data, audio_file, confidence, words)
#             stop_event.set()
#             progress_thread.join()
            
#             print(f"  Emma: {response}")
#             if session_data['current_test']:
#                 evaluation = evaluate_test_response(session_data['current_test'], user_input)
#                 feedback = f"Feedback: {evaluation['feedback']} Strengths: {evaluation['strengths']}. Work on: {evaluation['areas_to_improve']}."
#                 print(f"  Emma (Feedback): {feedback}")
#                 response += " " + feedback
#                 session_data['current_test'] = None
            
#             text_to_speech_async(response)
#             if os.path.exists(audio_file):
#                 os.remove(audio_file)
    
#     except KeyboardInterrupt:
#         print("\n  Session interrupted by user. Generating your learning report...")
#         interrupt_audio = text_to_speech_async("Session interrupted. I'll prepare your report now.")
#         interrupt_audio.result()
        
#         stop_event = threading.Event()
#         progress_thread = threading.Thread(target=show_progress, args=("Generating report", stop_event))
#         progress_thread.start()
#         report = generate_learning_report()
#         stop_event.set()
#         progress_thread.join()
        
#         print("\n" + "="*70)
#         print("  YOUR ENGLISH LEARNING REPORT")
#         print("="*70)
#         print(report)
#         print("="*70)
        
#         farewell = "Thanks for practicing with me! Keep up your English studies!"
#         print(f"  Emma: {farewell}")
#         farewell_audio = text_to_speech_async(farewell)
#         farewell_audio.result()
    
#     except Exception as e:
#         print(f"  Unexpected error in voice_english_teacher: {str(e)}")
    
#     finally:
#         initial_audio.result()
#         executor.shutdown(wait=True)
#         pygame.mixer.quit()

# if __name__ == "__main__":
#     print("Debug: Starting main execution")
#     try:
#         voice_english_teacher()
#     except Exception as e:
#         print(f"  Fatal error: {str(e)}")













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

# API Keys - Replace with your actual keys
OPENAI_API_KEY = "REMOVED"
SPEECH_RECOGNITION_API_KEY = "REMOVED"  # Deepgram API key

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
session_start_time = None
current_level = None

SESSION_FILE = "saved_session.json"

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def save_progress(session_data, learning_data):
    data = {
        "session_data": session_data,
        "learning_data": learning_data
    }
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f, default=convert_numpy)



def load_progress():
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r") as f:
                try:
                    data = json.load(f)
                    return data.get("session_data", {}), data.get("learning_data", {})
                except json.JSONDecodeError as e:
                    print(f"  ⚠️ Error reading saved session: {str(e)}")
                    # If file is corrupted, delete it
                    os.remove(SESSION_FILE)
                    print("  ℹ️ Corrupted session file removed. Starting fresh session.")
                    return {}, {}
        return {}, {}
    except Exception as e:
        print(f"  ⚠️ Error accessing session file: {str(e)}")
        return {}, {}


# Learning data storage
LEARNING_DATA = {
    'errors_made': [],
    'vocabulary_introduced': [],
    'lesson_topics': [],
    'assessment_results': [],
    'speaking_samples': [],
    'improvement_areas': [],
    'strengths': []
}

# English teacher system prompt
ENGLISH_TEACHER_PROMPT = """
You are Emma, a human-like English language teacher with a warm, encouraging personality. Your goal is to have natural conversations with the student to help them improve their English speaking skills. Act exactly like a real English teacher would in a one-on-one conversation session.

As Emma, you should:
1. Lead the conversation naturally by asking engaging questions about topics interesting to the student
2. Listen to the student's responses and continue the conversation naturally
3. Provide subtle, constructive feedback on errors by modeling the correct language
4. Gauge the student's level through their responses and adjust your language accordingly
5. Introduce new vocabulary or expressions naturally within the conversation
6. Ask follow-up questions to encourage longer responses and deeper thinking
7. Provide positive reinforcement when the student communicates effectively
8. Keep the conversation flowing like a real human teacher would - avoid sounding like an AI
9. Incorporate audio feedback on pronunciation, fluency, and rhythm when provided


🧠 Your core behaviors:
1. Engage the student in natural conversation using thoughtful questions.
2. Speak clearly and encourage longer responses.
3. Keep the tone warm, human, and real – not robotic.

🔍 Student Proficiency:
- English level is rated on a scale of 1 to 10 (1 = beginner, 10 = fluent).
- Adapt your speaking complexity to match the student's level.
- Avoid using labels like "beginner" – instead think in terms of numerical level (e.g., Level 3 or Level 7).

📚 Mini-Lessons:
- Insert short (1–2 minute) lessons **every 4 exchanges**.
- Focus on issues you've observed in speech or grammar.
- Format: Explain > Give 2–3 examples > Invite a response.

🧪 Proficiency Testing:
- Insert a **multi-part spoken test every 6 exchanges**, dynamically created via AI.
- Test includes grammar, fill-in-the-blank, phrase repetition, and paragraph reading.
- Do **not** evaluate immediately after each user message — save full feedback for after the test.

✅ Feedback Strategy:
- Provide detailed feedback only after:
  - Every **mini-lesson**, or
  - After a **full proficiency test**
- Avoid interrupting the flow with small corrections every time.

🗂️ Session Context:
- Track and reference recent errors, topics, test scores, and improvements.
- Encourage progress by noticing improvements over time.

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
    headers = {"Authorization": f"Token {SPEECH_RECOGNITION_API_KEY}", "Content-Type": "audio/wav"}
    params = {
        "model": "nova-2",
        "language": "en",
        "punctuate": "true",
        "diarize": "false",
        "detect_language": "true",
        "utterances": "true",
        "detect_topics": "true",
        "summarize": "v2"
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
    payload = {"model": "tts-1", "input": text, "voice": "nova", "response_format": "mp3", "speed": 1.0}
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

# Language and Audio Analysis Functions
def analyze_language_level(text):
    try:
        messages = [
            {"role": "system", "content": "You are an English examiner. Based on the user's spoken text, assess their English proficiency on a scale of 1 to 10 (1 = very basic, 10 = near native). Consider grammar, vocabulary, fluency, and sentence complexity. Reply with ONLY the number."},
            {"role": "user", "content": text}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=5,
            temperature=0.3
        )
        level = response.choices[0].message.content.strip()
        if level.isdigit():
            level_num = int(level)
            return min(max(level_num, 1), 10)
    except Exception as e:
        print(f"  Error in language level analysis: {str(e)}")
    return 5  # Default fallback


def analyze_language_errors(text, level):
    try:
        messages = [
            {
                "role": "system",
                "content": f"""Analyze the text from a level {level}/10 English learner and identify up to 3 grammar or vocabulary errors.
Return ONLY a JSON array, like:
[
  {{"error_type": "grammar", "error": "I don't like it.", "correction": "I don't like that."}},
  ...
]
"""},
            {"role": "user", "content": text}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=250,
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()

        # Improved JSON parsing with error handling
        try:
            # Check if content looks like JSON before attempting to parse
            if content.startswith('[') or content.startswith('{'):
                return json.loads(content)
            else:
                print(f"⚠️ Response not in expected JSON format: {content[:100]}...")
                # Fallback: Return empty array instead of failing
                return []
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON parsing error: {str(json_err)}")
            print(f"⚠️ Problematic content: {content[:200]}...")
            # Return empty list as fallback
            return []

    except Exception as e:
        print(f"❌ Error in language error analysis: {str(e)}")
        return []



def analyze_speech(spoken_text, confidence, audio_file_path, words):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"""
                    You are an English teacher analyzing speech. Focus on the student's level: {current_level}.
                    Use word-level confidence scores to identify pronunciation issues (below 0.9).
                    Use transcription confidence ({confidence}) for overall clarity if below 0.95.
                    Provide feedback on pronunciation, fluency, and accent.
                    All scores between 0.0 and 1.0.
                """},
                {"role": "user", "content": f"""
                    Spoken text: "{spoken_text}"
                    Transcription confidence: {confidence}
                    Audio file: {audio_file_path}
                    Word-level data: {json.dumps(words)}
                    Learner level: {current_level}
                    Provide analysis in JSON:
                    {{
                        "pronunciation": {{"mispronounced_words": [{{"word": "", "suggestion": ""}}], "score": 0.0, "feedback": ""}},
                        "fluency": {{"issues": [], "score": 0.0, "feedback": ""}},
                        "accent": {{"patterns": [], "feedback": ""}},
                        "overall": {{"score": 0.0, "strengths": [], "areas_for_improvement": []}}
                    }}
                """}
            ],
            temperature=0.7
        )
        
        raw_content = response.choices[0].message.content.strip()
        try:
            analysis = json.loads(raw_content)
            print(f"  Debug: audio_analysis = {json.dumps(analysis)}")
            return analysis
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON parsing error in speech analysis: {str(json_err)}")
            print(f"⚠️ Raw content: {raw_content[:200]}...")
            # Return fallback analysis data
            return {
                "pronunciation": {"mispronounced_words": [], "score": 0.5, "feedback": "Keep practicing pronunciation!"},
                "fluency": {"issues": [], "score": 0.5, "feedback": "You're doing fine, keep going!"},
                "accent": {"patterns": [], "feedback": "Your accent is understandable!"},
                "overall": {"score": 0.5, "strengths": ["Effort"], "areas_for_improvement": ["Clarity"]}
            }
    except Exception as e:
        print(f"  Speech analysis error: {str(e)}")
        return {
            "pronunciation": {"mispronounced_words": [], "score": 0.5, "feedback": "Keep practicing pronunciation!"},
            "fluency": {"issues": [], "score": 0.5, "feedback": "You're doing fine, keep going!"},
            "accent": {"patterns": [], "feedback": "Your accent is understandable!"},
            "overall": {"score": 0.5, "strengths": ["Effort"], "areas_for_improvement": ["Clarity"]}
        }


def detect_user_intent(text):
    try:
        messages = [
            {"role": "system", "content": "You are an assistant identifying user intent from spoken English. Possible intents: 'take_test', 'grammar_practice', 'vocab_practice', 'just_talk'. Respond with only the intent."},
            {"role": "user", "content": text}
        ]
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=10)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Intent detection error: {str(e)}")
        return "just_talk"



def analyze_audio_features(audio_file_path):
    try:
        y, sr = librosa.load(audio_file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_cleaned = f0[~np.isnan(f0)]
        mean_f0 = np.mean(f0_cleaned) if len(f0_cleaned) > 0 else 0
        std_f0 = np.std(f0_cleaned) if len(f0_cleaned) > 0 else 0
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = np.mean(rms)
        speech_rate = len(librosa.onset.onset_detect(y=y, sr=sr)) / duration if duration > 0 else 0
        return {
            "pitch": {"mean": mean_f0, "variation": std_f0},
            "volume": {"mean": mean_rms},
            "speech_rate": speech_rate,
            "duration": duration
        }
    except Exception as e:
        print(f"  Audio feature analysis error: {str(e)}")
        return {"pitch": {"mean": 0, "variation": 0}, "volume": {"mean": 0}, "speech_rate": 0, "duration": 0}

def generate_lesson_plan(session_data, recent_exchanges, audio_analysis):
    recent_text = " ".join([entry["content"] for entry in recent_exchanges])
    prompt = f"""
    Create a 1-2 min mini-lesson for a {session_data['level']} student based on:
    - Recent conversation: {recent_text}
    - Audio analysis: {json.dumps(audio_analysis)}
    Focus on grammar, vocab, or pronunciation errors if present. Format as JSON:
    {{'topic': '', 'explanation': '', 'examples': '', 'practice': ''}}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=350
        )
        raw = response.choices[0].message.content.strip()
        print(f"\n[DEBUG] Lesson plan raw content:\n{raw[:500]}")

        # Improved JSON parsing
        try:
            if raw.startswith('{') or raw.startswith('['):
                return json.loads(raw)
            else:
                print("⚠️ Lesson plan response not valid JSON. Content:")
                print(raw[:200])
                return {
                    "topic": "Conversation Practice",
                    "explanation": "Let's just keep practicing natural conversation.",
                    "examples": "We've been talking well!",
                    "practice": "Try saying something about your day."
                }
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON parsing error in lesson plan: {str(json_err)}")
            return {
                "topic": "Conversation Practice",
                "explanation": "Let's just keep practicing natural conversation.",
                "examples": "We've been talking well!",
                "practice": "Try saying something about your day."
            }

    except Exception as e:
        print(f"❌ Error in lesson plan generation: {e}")
        return {
            "topic": "Conversation Practice",
            "explanation": "Let's just keep practicing natural conversation.",
            "examples": "We've been talking well!",
            "practice": "Try saying something about your day."
        }

def create_proficiency_test(session_data):
    prompt = f"""
    The student is at level {session_data['level']} out of 10.
    Design a spoken English proficiency test with 4 tasks:
    1. Grammar question (multiple choice, one correct answer)
    2. Fill in the blank (simple sentence)
    3. Repeat this phrase (to test pronunciation)
    4. Read aloud a short paragraph (to test fluency and intonation)

    Respond in this JSON format:
    {{
        "test_topic": "Comprehensive Voice Proficiency Test",
        "tasks": [
            {{"type": "grammar", "question": "", "choices": ["", "", ""], "answer": ""}},
            {{"type": "fill_blank", "sentence": "", "answer": ""}},
            {{"type": "repeat_phrase", "phrase": ""}},
            {{"type": "read_paragraph", "text": ""}}
        ]
    }}
    """
    try:
        messages = [{"role": "system", "content": prompt}]
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages, max_tokens=500, temperature=0.5)
        raw = response.choices[0].message.content.strip()
        print(f"\n[DEBUG] Test creation raw content:\n{raw[:500]}")

        try:
            if raw.startswith('{') or raw.startswith('['):
                return json.loads(raw)
            else:
                print("⚠️ Test response not valid JSON. Content:")
                print(raw[:200])
                return fallback_test()
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON parsing error in test creation: {str(json_err)}")
            return fallback_test()

    except Exception as e:
        print(f"❌ Error creating test: {e}")
        return fallback_test()

def fallback_test():
    return {
        "test_topic": "Fallback Voice Test",
        "tasks": [
            {"type": "grammar", "question": "Which is correct?", "choices": ["He go", "He goes", "He gone"], "answer": "He goes"},
            {"type": "fill_blank", "sentence": "She ___ to work every day.", "answer": "goes"},
            {"type": "repeat_phrase", "phrase": "Practice makes perfect."},
            {"type": "read_paragraph", "text": "Learning a language takes time, effort, and consistency."}
        ]
    }



def run_proficiency_test_via_voice(test):
    print("\n🎤 Starting Proficiency Test...\n")
    user_responses = []

    for task in test["tasks"]:
        if task["type"] == "grammar":
            question_text = f"{task['question']} Choices: {', '.join(task['choices'])}"
        elif task["type"] == "fill_blank":
            question_text = f"Fill in the blank: {task['sentence']}"
        elif task["type"] == "repeat_phrase":
            question_text = f"Please repeat: {task['phrase']}"
        elif task["type"] == "read_paragraph":
            question_text = f"Please read this paragraph: {task['text']}"
        else:
            continue

        print(f"  Emma: {question_text}")
        text_to_speech_async(question_text).result()

        audio_file = record_audio_with_button()
        if not audio_file:
            user_responses.append("")
            continue
        transcript, confidence, words = transcribe_audio(audio_file)
        print(f"  You said: {transcript}")
        user_responses.append(transcript)
        if os.path.exists(audio_file):
            os.remove(audio_file)

    return user_responses



def evaluate_test_response(test, user_responses, session_data):
    try:
        evaluation_prompt = {
            "test_topic": test["test_topic"],
            "tasks": test["tasks"],
            "user_responses": user_responses
        }

        messages = [
            {"role": "system", "content": """
            You are an English teacher evaluating a student's voice test.
            Assess each response (grammar, fill-in, repeat, read) for correctness, pronunciation, and fluency.
            Give a score from 1 to 5 for each part, then provide an overall score (average), strengths, and areas to improve.
            Reply in this JSON format:
            {
            "scores": [1, 4, 5, 3],
            "overall_score": 3.5,
            "feedback": "",
            "strengths": "",
            "areas_to_improve": ""
            }
            """},
            {"role": "user", "content": json.dumps(evaluation_prompt)}
        ]

        response = openai.ChatCompletion.create(model="gpt-4", messages=messages, max_tokens=400)
        raw = response.choices[0].message.content.strip()
        print(f"\n[DEBUG] Test eval raw content:\n{raw[:500]}")

        try:
            if raw.startswith('{') or raw.startswith('['):
                evaluation = json.loads(raw)
            else:
                print("⚠️ Evaluation response not JSON. Content:")
                print(raw[:200])
                return {
                    "scores": [3, 3, 3, 3],
                    "overall_score": 3.0,
                    "feedback": "Fair performance overall.",
                    "strengths": "Clear speech",
                    "areas_to_improve": "Grammar accuracy, fluency"
                }
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON parsing error in test evaluation: {str(json_err)}")
            return {
                "scores": [3, 3, 3, 3],
                "overall_score": 3.0,
                "feedback": "Fair performance overall.",
                "strengths": "Clear speech",
                "areas_to_improve": "Grammar accuracy, fluency"
            }

        LEARNING_DATA['assessment_results'].append({
            'topic': test['test_topic'],
            'score': evaluation['overall_score'],
            'feedback': evaluation['feedback']
        })

        session_data['current_test_completed'] = True
        return evaluation

    except Exception as e:
        print(f"❌ Error evaluating test: {e}")
        return {
            "scores": [3, 3, 3, 3],
            "overall_score": 3.0,
            "feedback": "Fair performance overall.",
            "strengths": "Clear speech",
            "areas_to_improve": "Grammar accuracy, fluency"
        }



        
def generate_learning_report():
    session_data = {
        'duration': int((time.time() - session_start_time) / 60),
        'exchanges': len(CONVERSATION_HISTORY) // 2,
        'level': current_level,
        'topics': LEARNING_DATA['lesson_topics'],
        'errors': LEARNING_DATA['errors_made'],
        'speaking_samples': LEARNING_DATA['speaking_samples'],
        'strengths': LEARNING_DATA['strengths'],
        'improvement_areas': LEARNING_DATA['improvement_areas'],
        'assessments': LEARNING_DATA['assessment_results']
    }

    speaking_samples = session_data['speaking_samples']
    if speaking_samples:
        avg_pronunciation = float(np.mean([s['audio_analysis']['pronunciation']['score'] for s in speaking_samples]))
        avg_fluency = float(np.mean([s['audio_analysis']['fluency']['score'] for s in speaking_samples]))
        avg_speech_rate = float(np.mean([s['audio_features']['speech_rate'] for s in speaking_samples]))
    else:
        avg_pronunciation = avg_fluency = avg_speech_rate = 0.0

    latest_test = session_data['assessments'][-1] if session_data['assessments'] else None
    recent_errors = session_data['errors'][-3:] if session_data['errors'] else []

    prompt = f"""
You are an English language tutor summarizing a short voice-based learning session.

🧾 Session Summary:
- Duration: {session_data['duration']} min
- Exchanges: {session_data['exchanges']}
- Detected Level: {session_data['level']}/10

🎯 Analytics:
- Avg Pronunciation Score: {avg_pronunciation:.2f}
- Avg Fluency Score: {avg_fluency:.2f}
- Avg Speech Rate: {avg_speech_rate:.1f} words/sec

🧪 Most Recent Test:
{f"- Score: {latest_test.get('score', 'N/A')}/5" if latest_test else "- No test taken"}
{f"- Feedback: {latest_test.get('feedback', '')}" if latest_test else ""}

✅ Strengths:
{', '.join(session_data['strengths']) if session_data['strengths'] else "N/A"}

🔧 Needs Improvement:
{', '.join(session_data['improvement_areas']) if session_data['improvement_areas'] else "N/A"}

📝 Common Issues:
{json.dumps(recent_errors, indent=2) if recent_errors else "No significant recurring errors."}

🎁 Final Tip:
Give the student 1 sentence of encouragement or next steps.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=400
        )
        report = response.choices[0].message.content.strip()

        # Save as .txt
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_filename = f"english_learning_report_{timestamp}.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n  ✅ Learning report saved as: {report_filename}")
        return report

    except Exception as e:
        print(f"  ❌ Error generating report: {str(e)}")
        return "Report generation failed."




def collect_feedback_for_delayed_delivery(audio_analysis, errors):
    feedback = {}

    # Grammar/Language issues
    if errors:
        feedback["grammar"] = errors[-3:]

    # Pronunciation feedback
    mispronounced = audio_analysis["pronunciation"].get("mispronounced_words", [])
    if isinstance(mispronounced, list) and mispronounced:
        pronunciation_feedback = ", ".join(
            [f"'{w['word']}' - try {w['suggestion']}" for w in mispronounced]
        )
        feedback["pronunciation"] = pronunciation_feedback

    return feedback



def chat_with_english_teacher(user_input, session_data, audio_file, confidence, words):
    global LEARNING_DATA
    
    errors = analyze_language_errors(user_input, session_data['level'])

    # Store errors for later feedback only every 4 exchanges or after test
    if session_data['exchanges'] > 0 and (session_data['exchanges'] % 4 == 0 or session_data.get('current_test_completed')):
        if errors:
            LEARNING_DATA['errors_made'].extend([e for e in errors if e not in LEARNING_DATA['errors_made']])
        session_data['current_test_completed'] = False  # reset

    
    audio_analysis = analyze_speech(user_input, confidence, audio_file, words)
    audio_features = analyze_audio_features(audio_file)
    LEARNING_DATA['speaking_samples'].append({
        'text': user_input[:150],
        'exchange_num': session_data['exchanges'],
        'audio_analysis': audio_analysis,
        'audio_features': audio_features
    })
    
    insert_lesson = session_data['exchanges'] % 4 == 0 and session_data['exchanges'] > 0
    insert_test = session_data['exchanges'] % 6 == 0 and session_data['exchanges'] > 0 and not insert_lesson

    
    messages = [{"role": "system", "content": ENGLISH_TEACHER_PROMPT}]
    context_message = f"""
    Session info:
    - Level: {session_data['level']}
    - Duration: {session_data['duration']} min
    - Topics: {', '.join(session_data['topics'])}
    - Exchanges: {session_data['exchanges']}
    - Errors: {json.dumps(LEARNING_DATA['errors_made'][-3:])}
    - Audio: Pronunciation {audio_analysis['pronunciation']['score']}, Fluency {audio_analysis['fluency']['score']}, Rate {audio_features['speech_rate']:.1f}
    {f'INSERT MINI-LESSON: Yes' if insert_lesson else ''}
    {f'INSERT TEST: Yes' if insert_test else ''}
    """
    messages.append({"role": "system", "content": context_message})
    
    if insert_lesson:
        recent_exchanges = CONVERSATION_HISTORY[-6:]
        lesson_plan = generate_lesson_plan(session_data, recent_exchanges, audio_analysis)
        messages.append({"role": "system", "content": f"INSERT MINI-LESSON: {json.dumps(lesson_plan)}"})
    
    if insert_test:
        test = create_proficiency_test(session_data)
        messages.append({"role": "system", "content": f"INSERT TEST: {json.dumps(test)}"})
        session_data['current_test'] = test
    
    for entry in CONVERSATION_HISTORY:
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": user_input})
    
    try:
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages, max_tokens=400, temperature=0.7)
        assistant_response = response.choices[0].message.content.strip()

        # Show delayed feedback only every 4 exchanges or after a test
        if session_data['exchanges'] % 4 == 0 or session_data.get('current_test_completed'):
            if 'pending_feedback' in session_data:
                if session_data['pending_feedback'].get("grammar"):
                    grammar_fb = session_data['pending_feedback']['grammar']
                    assistant_response += "\n\n🔍 Language Note:\n"
                    for g in grammar_fb:
                        assistant_response += f"- {g['error']} → {g['correction']}\n"
                if session_data['pending_feedback'].get("pronunciation"):
                    assistant_response += "\n🔊 Pronunciation Tip:\n" + session_data['pending_feedback']['pronunciation']
                session_data['pending_feedback'] = {}  # clear after showing

    except Exception as e:
        print(f"  API Error: {str(e)}")
        assistant_response = "Sorry, I had trouble processing that. Let's keep going!"
    
    if session_data['exchanges'] > 0 and (session_data['exchanges'] % 4 == 0 or session_data.get('current_test_completed')):
        delayed_feedback = collect_feedback_for_delayed_delivery(audio_analysis, errors)
        session_data['pending_feedback'] = delayed_feedback
    
    CONVERSATION_HISTORY.append({"role": "user", "content": user_input})
    CONVERSATION_HISTORY.append({"role": "assistant", "content": assistant_response})
    session_data['exchanges'] += 1
    return assistant_response

def voice_english_teacher():
    print("Debug: Entering voice_english_teacher")
    global session_start_time, current_level
    loaded_session, loaded_learning = load_progress()

    session_data = loaded_session or {'level': 5, 'duration': 0, 'topics': [], 'exchanges': 0, 'start_time': time.time(), 'current_test': None}
    LEARNING_DATA.update(loaded_learning or {})
    session_data['start_time'] = time.time()
    session_start_time = time.time()
    current_level = session_data['level']

    if loaded_session:
        print("  Emma: Welcome back! Let’s continue from where we left off.")
        text_to_speech_async("Welcome back! Let’s continue from where we left off.")
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Debug: After clearing screen")
    print("\n" + "="*70)
    print("  Welcome to Your Enhanced Voice English Conversation Practice!")
    print("  I'm Emma, your English teacher for today.")
    print("="*70)
    print("\n  Let's start!\n")
    
    initial_greeting = "Hi! I'm Emma, here to help you with your English. We'll chat, and I'll give you tips on speaking, including pronunciation and fluency. Tell me about yourself and what you'd like to practice!"
    print(f"  Emma: {initial_greeting}")
    initial_audio = text_to_speech_async(initial_greeting)
    
    try:
        while True:
            if keyboard.is_pressed('esc'):
                print("\n  Emma: Thanks for practicing with me!")
                end_audio = text_to_speech_async("Thanks for practicing! Here's your report.")
                end_audio.result()
                
                stop_event = threading.Event()
                progress_thread = threading.Thread(target=show_progress, args=("Generating report", stop_event))
                progress_thread.start()
                report = generate_learning_report()
                stop_event.set()
                progress_thread.join()
                
                print("\n" + "="*70)
                print("  YOUR ENGLISH LEARNING REPORT")
                print("="*70)
                print(report)
                print("="*70)
                break
            
            audio_file = record_audio_with_button()
            if not audio_file:
                print("  Emma: I didn’t hear anything. Try again!")
                text_to_speech_async("I didn’t hear anything. Try again!")
                continue
            
            stop_event = threading.Event()
            progress_thread = threading.Thread(target=show_progress, args=("Processing speech", stop_event))
            progress_thread.start()
            user_input, confidence, words = transcribe_audio(audio_file)
            stop_event.set()
            progress_thread.join()
            
            if not user_input:
                print("  Emma: I didn’t catch that. Try again?")
                text_to_speech_async("I didn’t catch that. Try again?")
                continue
            
            print(f"  You: {user_input}")
            session_data['duration'] = int((time.time() - session_data['start_time']) / 60)
            
            if session_data['exchanges'] == 0:
                current_level = analyze_language_level(user_input)
                session_data['level'] = current_level
                print(f"  (Detected level: {current_level}/10)")
            
            stop_event = threading.Event()
            progress_thread = threading.Thread(target=show_progress, args=("Emma is thinking", stop_event))
            progress_thread.start()
            user_intent = detect_user_intent(user_input)
            print(f"  (Detected intent: {user_intent})")

            if user_intent == "take_test":
                session_data['current_test'] = create_proficiency_test(session_data)
                print("  Emma: Starting your test!")
                text_to_speech_async("Okay! Let's begin the test.").result()
                user_responses = run_proficiency_test_via_voice(session_data['current_test'])
                evaluation = evaluate_test_response(session_data['current_test'], user_responses, session_data)

                # [Same feedback code block here...]
                session_data['current_test'] = None
                session_data['current_test_completed'] = True
                response = "Thanks for completing the test! Here's your feedback."

            else:
                response = chat_with_english_teacher(user_input, session_data, audio_file, confidence, words)

            stop_event.set()
            progress_thread.join()
            
            print(f"  Emma: {response}")
            if session_data['current_test']:
                user_responses = run_proficiency_test_via_voice(session_data['current_test'])
                evaluation = evaluate_test_response(session_data['current_test'], user_responses, session_data)

                feedback = (
                    f"Test complete!\n"
                    f"  ✅ Scores: {evaluation['scores']}\n"
                    f"  ⭐ Overall Score: {evaluation['overall_score']}/5\n"
                    f"  💬 Feedback: {evaluation['feedback']}\n"
                    f"  💡 Strengths: {evaluation['strengths']}\n"
                    f"  🔧 Areas to Improve: {evaluation['areas_to_improve']}"
                )
                print(f"  Emma (Feedback):\n{feedback}")
                text_to_speech_async("Here's your feedback from the test.").result()
                text_to_speech_async(evaluation['feedback'])

                # Save to learning data
                LEARNING_DATA['strengths'].append(evaluation['strengths'])
                LEARNING_DATA['improvement_areas'].append(evaluation['areas_to_improve'])

                session_data['current_test'] = None

            
            text_to_speech_async(response)
            if os.path.exists(audio_file):
                os.remove(audio_file)
    
    except KeyboardInterrupt:
        print("\n  Session interrupted by user. Generating your learning report...")
        interrupt_audio = text_to_speech_async("Session interrupted. I'll prepare your report now.")
        interrupt_audio.result()
        
        stop_event = threading.Event()
        progress_thread = threading.Thread(target=show_progress, args=("Generating report", stop_event))
        progress_thread.start()
        report = generate_learning_report()
        stop_event.set()
        progress_thread.join()
        
        print("\n" + "="*70)
        print("  YOUR ENGLISH LEARNING REPORT")
        print("="*70)
        print(report)
        print("="*70)
        
        farewell = "Thanks for practicing with me! Keep up your English studies!"
        print(f"  Emma: {farewell}")
        farewell_audio = text_to_speech_async(farewell)
        farewell_audio.result()
    
    except Exception as e:
        print(f"  Unexpected error in voice_english_teacher: {str(e)}")
    
    finally:
        initial_audio.result()
        save_progress(session_data, LEARNING_DATA)
        executor.shutdown(wait=True)
        pygame.mixer.quit()

if __name__ == "__main__":
    print("Debug: Starting main execution")
    try:
        voice_english_teacher()
    except Exception as e:
        print(f"  Fatal error: {str(e)}")

