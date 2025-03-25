# import openai
# import sounddevice as sd
# import numpy as np
# import wave
# import requests
# import pyttsx3
# import tempfile
# import os
# import json
# import time
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import webrtcvad
# import collections
# from datetime import datetime
# from dotenv import load_dotenv

# # Predefined test sentences and paragraphs
# TEST_TEXTS = {
#     "1": {"title": "Simple Sentence", "text": "The quick brown fox jumps over the lazy dog."},
#     "2": {"title": "Intermediate Sentence", "text": "She walks to the park every morning with her friendly golden retriever."},
#     "3": {"title": "Short Paragraph", "text": "The sun sets behind the mountains, painting the sky with shades of orange and pink. Birds fly back to their nests, and a cool breeze begins to blow through the trees."},
#     "4": {"title": "Longer Paragraph", "text": "Learning a new language can be both challenging and rewarding. It requires dedication, practice, and patience, but the ability to communicate with others opens up a world of opportunities. Many people find that immersion is the best way to improve their skills quickly."}
# }

# # Function to initialize text-to-speech engine
# def init_tts_engine():
#     engine = pyttsx3.init()
#     engine.setProperty('rate', 150)
#     engine.setProperty('volume', 0.9)
#     return engine

# # Function to speak text using TTS engine
# def speak_text(text, engine):
#     print(f"🔊 AI: {text}")
#     engine.say(text)
#     engine.runAndWait()

# # Function to record and save audio
# def record_audio(duration=5, sample_rate=16000):
#     print("🎤 Speak now...")
#     recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
#     sd.wait()
#     recording = (recording * 32767).astype(np.int16)
#     # Save with a timestamped filename
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"recording_{timestamp}.wav"
#     with wave.open(filename, 'wb') as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)
#         wf.setframerate(sample_rate)
#         wf.writeframes(recording.tobytes())
#     print(f"Audio saved as: {filename}")
#     return filename

# # Function to transcribe audio using Deepgram with confidence scores
# def transcribe_audio(audio_file_path, deepgram_api_key):
#     url = "https://api.deepgram.com/v1/listen"
#     headers = {"Authorization": f"Token {deepgram_api_key}"}
#     params = {
#         "model": "nova-2",
#         "language": "en",
#         "punctuate": "true",
#         "diarize": "false",  # Changed from False to "false"
#         "detect_language": "true",
#         "utterances": "true",
#         "detect_topics": "true",
#         "summarize": "v2"
#     }
    
#     with open(audio_file_path, 'rb') as audio:
#         response = requests.post(url, headers=headers, params=params, data=audio)
#     if response.status_code == 200:
#         result = response.json()
#         transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
#         confidence = result["results"]["channels"][0]["alternatives"][0]["confidence"]
#         return transcript, confidence, audio_file_path
    
#     print(f"Deepgram error: {response.status_code} - {response.text}")
#     os.unlink(audio_file_path)
#     return None, None, audio_file_path

# # Function to analyze speech for language proficiency with harsher critique
# def analyze_speech(reference_text, spoken_text, confidence, audio_file_path, openai_api_key):
#     openai.api_key = openai_api_key
    
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": """
#                     You are a strict, highly critical English language teacher with expertise in pronunciation, grammar, fluency, and accent analysis (both British and American English). 
#                     Provide a harsh, detailed critique of the spoken text compared to the reference text. 
#                     Identify every possible error or deviation, even minor ones, and suggest specific improvements. 
#                     Use the transcription confidence score ({confidence}) to infer pronunciation accuracy if below 0.95. 
#                     Evaluate accent based on transcription clues and note that an audio file is available at {audio_file_path} for further manual review if needed. 
#                     Scores must be between 0.0 and 1.0 (0% to 100%). Be relentless in pointing out flaws and providing actionable feedback.
#                 """},
#                 {"role": "user", "content": f"""
#                     Reference text: "{reference_text}"
#                     Spoken text: "{spoken_text}"
#                     Transcription confidence: {confidence}
#                     Audio file: {audio_file_path}
                    
#                     Provide analysis in strict JSON format:
#                     {{
#                         "grammar": {{"errors": [{{"error": "", "correction": ""}}], "score": 0.0, "feedback": ""}},
#                         "pronunciation": {{"mispronounced_words": [{{"word": "", "issue": "", "suggestion": ""}}], "score": 0.0, "feedback": ""}},
#                         "accent": {{"patterns": [], "british_similarity": 0.0, "american_similarity": 0.0, "feedback": ""}},
#                         "fluency": {{"issues": [], "score": 0.0, "feedback": ""}},
#                         "punctuation": {{"issues": [], "score": 0.0, "feedback": ""}},
#                         "overall": {{"score": 0.0, "strengths": [], "areas_for_improvement": [], "summary": ""}}
#                     }}
#                 """}
#             ],
#             temperature=0.7
#         )
        
#         content = response.choices[0].message.content.strip()
#         try:
#             return json.loads(content)
#         except json.JSONDecodeError as e:
#             print(f"JSON parsing error: {str(e)} - Response content: {content}")
#             return {
#                 "grammar": {"errors": [], "score": 0.5, "feedback": "Analysis unavailable"},
#                 "pronunciation": {"mispronounced_words": [], "score": 0.5, "feedback": "Analysis unavailable"},
#                 "accent": {"patterns": [], "british_similarity": 0.5, "american_similarity": 0.5, "feedback": "Analysis unavailable"},
#                 "fluency": {"issues": [], "score": 0.5, "feedback": "Analysis unavailable"},
#                 "punctuation": {"issues": [], "score": 0.5, "feedback": "Analysis unavailable"},
#                 "overall": {"score": 0.5, "strengths": ["Unable to analyze"], "areas_for_improvement": ["System error"], "summary": "Unable to provide detailed analysis"}
#             }
#     except Exception as e:
#         print(f"API error: {str(e)}")
#         return {"error": "Failed to analyze speech", "message": str(e)}

# # Main simplified function (original)
# def simple_speech_analysis(openai_api_key, deepgram_api_key, reference_text):
#     print("\n===== 🗣️ English Speech Analysis =====")
#     tts_engine = init_tts_engine()
    
#     print(f"\nPlease read the following text:")
#     print(f"\"{reference_text}\"")
#     speak_text(reference_text, tts_engine)
    
#     words_count = len(reference_text.split())
#     suggested_duration = max(5, min(15, words_count * 0.5))
    
#     duration_input = input(f"\nRecording duration in seconds (suggested: {suggested_duration:.1f}s, press Enter to use suggestion): ")
#     try:
#         duration = float(duration_input) if duration_input else suggested_duration
#     except ValueError:
#         duration = suggested_duration
#         print(f"Invalid input. Using suggested duration: {suggested_duration:.1f}s")
    
#     print("\nGet ready to speak...")
#     time.sleep(1)
#     audio_file = record_audio(duration=duration)
#     spoken_text, confidence, audio_file_path = transcribe_audio(audio_file, deepgram_api_key)
    
#     if not spoken_text:
#         print("Sorry, I couldn't understand what you said. Please try again.")
#         return
        
#     print(f"\nYou said: \"{spoken_text}\"")
#     print(f"Transcription confidence: {confidence:.2%}")
    
#     print("\nAnalyzing your speech...")
#     analysis = analyze_speech(reference_text, spoken_text, confidence, audio_file_path, openai_api_key)
    
#     if "error" in analysis:
#         print(f"Error: {analysis['error']}: {analysis['message']}")
#         return
    
#     # Display analysis results with capped percentages
#     print("\n===== 📊 Speech Analysis Results =====")
    
#     overall = analysis["overall"]
#     print(f"\n🌟 OVERALL SCORE: {min(int(overall['score']*100), 100)}%")
#     print(f"\n📝 Summary: {overall['summary']}")
    
#     print("\n✅ Strengths:")
#     for strength in overall["strengths"]:
#         print(f"• {strength}")
    
#     print("\n🔍 Areas for improvement:")
#     for area in overall["areas_for_improvement"]:
#         print(f"• {area}")
    
#     print("\n===== Detailed Analysis =====")
    
#     grammar = analysis["grammar"]
#     print(f"\n📐 GRAMMAR SCORE: {min(int(grammar['score']*100), 100)}%")
#     print(f"Feedback: {grammar['feedback']}")
#     if grammar["errors"]:
#         print("Errors identified:")
#         for error in grammar["errors"]:
#             print(f"• Error: {error['error']}")
#             print(f"  Correction: {error['correction']}")
    
#     pronunciation = analysis["pronunciation"]
#     print(f"\n🗣️ PRONUNCIATION SCORE: {min(int(pronunciation['score']*100), 100)}%")
#     print(f"Feedback: {pronunciation['feedback']}")
#     if pronunciation["mispronounced_words"]:
#         print("Mispronounced words:")
#         for word in pronunciation["mispronounced_words"]:
#             print(f"• Word: {word['word']}")
#             print(f"  Issue: {word['issue']}")
#             print(f"  Suggestion: {word['suggestion']}")
    
#     accent = analysis["accent"]
#     print(f"\n🌐 ACCENT SCORE - British: {min(int(accent['british_similarity']*100), 100)}%, American: {min(int(accent['american_similarity']*100), 100)}%")
#     print(f"Feedback: {accent['feedback']}")
#     if accent["patterns"]:
#         print("Accent patterns:")
#         for pattern in accent["patterns"]:
#             print(f"• {pattern}")
    
#     fluency = analysis["fluency"]
#     print(f"\n🌊 FLUENCY SCORE: {min(int(fluency['score']*100), 100)}%")
#     print(f"Feedback: {fluency['feedback']}")
#     if fluency["issues"]:
#         print("Fluency issues:")
#         for issue in fluency["issues"]:
#             print(f"• {issue}")
    
#     punctuation = analysis["punctuation"]
#     print(f"\n📝 PUNCTUATION SCORE: {min(int(punctuation['score']*100), 100)}%")
#     print(f"Feedback: {punctuation['feedback']}")
#     if punctuation["issues"]:
#         print("Punctuation issues:")
#         for issue in punctuation["issues"]:
#             print(f"• {issue}")

# # ====================== NEW VOICE ANALYSIS FUNCTIONS ======================

# def analyze_audio_features(audio_file_path):
#     """Extract prosodic and acoustic features from audio file"""
#     # Load the audio file
#     y, sr = librosa.load(audio_file_path, sr=None)
    
#     # Duration
#     duration = librosa.get_duration(y=y, sr=sr)
    
#     # Extract pitch (fundamental frequency) using PYIN algorithm
#     f0, voiced_flag, voiced_probs = librosa.pyin(y, 
#                                                 fmin=librosa.note_to_hz('C2'), 
#                                                 fmax=librosa.note_to_hz('C7'),
#                                                 sr=sr)
#     # Filter out NaN values for calculations
#     f0_cleaned = f0[~np.isnan(f0)]
#     if len(f0_cleaned) > 0:
#         mean_f0 = np.mean(f0_cleaned)
#         std_f0 = np.std(f0_cleaned)
#         min_f0 = np.min(f0_cleaned)
#         max_f0 = np.max(f0_cleaned)
#     else:
#         mean_f0, std_f0, min_f0, max_f0 = 0, 0, 0, 0
    
#     # Rhythm features - tempo and beats
#     onset_env = librosa.onset.onset_strength(y=y, sr=sr)
#     tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
#     # Energy/volume
#     rms = librosa.feature.rms(y=y)[0]
#     mean_rms = np.mean(rms)
#     std_rms = np.std(rms)
    
#     # Speech rate estimation (syllables per second approximation)
#     # Using onsets as proxy for syllables
#     onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
#     if len(onsets) > 0:
#         speech_rate = len(onsets) / duration  # onsets per second
#     else:
#         speech_rate = 0
    
#     # Spectral features
#     spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
#     spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
#     spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)[0])
    
#     # Zero crossing rate (related to consonant pronunciation)
#     zcr = np.mean(librosa.feature.zero_crossing_rate(y=y)[0])
    
#     # MFCC features (timbre, voice quality)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     mean_mfccs = np.mean(mfccs, axis=1)
    
#     # Pauses detection (silence periods)
#     non_silent_intervals = librosa.effects.split(y, top_db=30)
#     silent_intervals = []
    
#     if len(non_silent_intervals) > 0:
#         # Convert frame indices to time
#         for i in range(len(non_silent_intervals) - 1):
#             current_end = non_silent_intervals[i][1] / sr
#             next_start = non_silent_intervals[i+1][0] / sr
#             if next_start - current_end > 0.2:  # Silence longer than 200ms
#                 silent_intervals.append((current_end, next_start))
    
#         # Add initial silence if exists
#         if non_silent_intervals[0][0] > 0:
#             initial_silence = non_silent_intervals[0][0] / sr
#             if initial_silence > 0.2:
#                 silent_intervals.insert(0, (0, initial_silence))
                
#         # Add final silence if exists
#         last_end = non_silent_intervals[-1][1] / sr
#         if duration - last_end > 0.2:
#             silent_intervals.append((last_end, duration))
            
#     total_silence_duration = sum(end - start for start, end in silent_intervals)
#     speaking_rate = (len(onsets) / (duration - total_silence_duration)) if duration > total_silence_duration else 0
    
#     return {
#         "duration": duration,
#         "pitch": {
#             "mean": mean_f0,
#             "std": std_f0,
#             "min": min_f0,
#             "max": max_f0,
#             "range": max_f0 - min_f0 if max_f0 > 0 and min_f0 > 0 else 0
#         },
#         "tempo": tempo,
#         "volume": {
#             "mean": mean_rms,
#             "std": std_rms,
#             "range": np.max(rms) - np.min(rms)
#         },
#         "speech_rate": speech_rate,
#         "speaking_rate": speaking_rate,  # Speech rate excluding pauses
#         "spectral_features": {
#             "centroid": spectral_centroid,
#             "bandwidth": spectral_bandwidth,
#             "contrast": spectral_contrast,
#             "zero_crossing_rate": zcr
#         },
#         "mfccs": mean_mfccs.tolist(),
#         "pauses": {
#             "count": len(silent_intervals),
#             "total_duration": total_silence_duration,
#             "percentage": (total_silence_duration / duration) * 100 if duration > 0 else 0,
#             "intervals": silent_intervals
#         }
#     }

# def visualize_audio_features(audio_file_path, save_path=None):
#     """Generate visualizations of audio features"""
#     y, sr = librosa.load(audio_file_path, sr=None)
    
#     # Create figure with subplots
#     plt.figure(figsize=(14, 10))
    
#     # Waveform
#     plt.subplot(3, 1, 1)
#     librosa.display.waveshow(y, sr=sr)
#     plt.title('Waveform')
    
#     # Spectrogram
#     plt.subplot(3, 1, 2)
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
#     librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogram')
    
#     # Pitch contour
#     plt.subplot(3, 1, 3)
#     f0, voiced_flag, voiced_probs = librosa.pyin(y, 
#                                               fmin=librosa.note_to_hz('C2'), 
#                                               fmax=librosa.note_to_hz('C7'),
#                                               sr=sr)
#     times = librosa.times_like(f0, sr=sr)
#     plt.plot(times, f0, label='F0', color='b', linewidth=1)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Frequency (Hz)')
#     plt.title('Pitch Contour (F0)')
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path)
#         return save_path
#     else:
#         plt.show()
#         return None

# def analyze_voice_characteristics(audio_features):
#     """Analyze voice characteristics based on extracted audio features"""
#     analysis = {}
    
#     # Pitch analysis
#     pitch = audio_features["pitch"]
#     # Pitch reference ranges (approximate Hz)
#     male_pitch_range = (85, 180)
#     female_pitch_range = (165, 255)
#     child_pitch_range = (250, 400)
    
#     # Determine voice type based on mean pitch
#     mean_pitch = pitch["mean"]
#     if mean_pitch == 0:
#         analysis["voice_type"] = "Unable to determine (no clear pitch detected)"
#     elif male_pitch_range[0] <= mean_pitch <= male_pitch_range[1]:
#         analysis["voice_type"] = "Adult male"
#     elif female_pitch_range[0] <= mean_pitch <= female_pitch_range[1]:
#         analysis["voice_type"] = "Adult female"
#     elif child_pitch_range[0] <= mean_pitch <= child_pitch_range[1]:
#         analysis["voice_type"] = "Child"
#     elif mean_pitch > child_pitch_range[1]:
#         analysis["voice_type"] = "Very high-pitched voice"
#     elif mean_pitch < male_pitch_range[0]:
#         analysis["voice_type"] = "Very low-pitched voice"
#     else:
#         analysis["voice_type"] = "Intermediate/ambiguous pitch range"
    
#     # Pitch variation analysis (vocal expressiveness/monotony)
#     pitch_std = pitch["std"]
#     pitch_range = pitch["range"]
    
#     if pitch_std < 10:
#         analysis["pitch_variation"] = "Monotonous speech with little pitch variation"
#     elif 10 <= pitch_std < 30:
#         analysis["pitch_variation"] = "Moderate pitch variation, average expressiveness"
#     else:
#         analysis["pitch_variation"] = "Highly expressive speech with significant pitch variation"
    
#     # Speech rate analysis
#     speech_rate = audio_features["speech_rate"]
    
#     if speech_rate < 2:
#         analysis["speech_rate"] = "Slow speech"
#     elif 2 <= speech_rate < 4:
#         analysis["speech_rate"] = "Average speech rate"
#     else:
#         analysis["speech_rate"] = "Fast speech"
        
#     # Speaking fluency based on pauses
#     pause_percentage = audio_features["pauses"]["percentage"]
#     pause_count = audio_features["pauses"]["count"]
    
#     if pause_percentage > 30 and pause_count > 5:
#         analysis["fluency"] = "Hesitant speech with frequent pauses"
#     elif 15 <= pause_percentage <= 30:
#         analysis["fluency"] = "Natural speech with typical pausing"
#     elif pause_percentage < 15:
#         analysis["fluency"] = "Fluid, continuous speech with few pauses"
    
#     # Voice quality based on spectral features and volume
#     zcr = audio_features["spectral_features"]["zero_crossing_rate"]
#     spectral_centroid = audio_features["spectral_features"]["centroid"]
#     volume_mean = audio_features["volume"]["mean"]
#     volume_std = audio_features["volume"]["std"]
    
#     # Breathiness
#     if zcr > 0.1:
#         analysis["breathiness"] = "Breathy voice quality detected"
#     else:
#         analysis["breathiness"] = "Low breathiness in voice"
    
#     # Voice brightness/darkness
#     if spectral_centroid > 2000:
#         analysis["brightness"] = "Bright voice timbre"
#     elif spectral_centroid < 1000:
#         analysis["brightness"] = "Dark/warm voice timbre"
#     else:
#         analysis["brightness"] = "Balanced voice timbre"
    
#     # Volume characteristics
#     if volume_mean > 0.1:
#         analysis["volume"] = "Loud speech"
#     elif volume_mean < 0.05:
#         analysis["volume"] = "Quiet/soft speech"
#     else:
#         analysis["volume"] = "Average volume"
    
#     if volume_std > 0.05:
#         analysis["volume_consistency"] = "Highly variable volume (dynamic speech)"
#     else:
#         analysis["volume_consistency"] = "Consistent volume throughout speech"

#     # Overall voice characteristics summary
#     analysis["summary"] = f"The speaker has a {analysis['voice_type'].lower()} voice with {analysis['pitch_variation'].lower()}. "
#     analysis["summary"] += f"They speak at a {analysis['speech_rate'].lower()} with {analysis['fluency'].lower()}. "
#     analysis["summary"] += f"The voice has a {analysis['brightness'].lower()} with {analysis['volume'].lower()}."
    
#     return analysis

# def read_wave(path):
#     """Reads a .wav file."""
#     with wave.open(path, 'rb') as wf:
#         num_channels = wf.getnchannels()
#         sample_width = wf.getsampwidth()
#         sample_rate = wf.getframerate()
#         pcm_data = wf.readframes(wf.getnframes())
#         return pcm_data, sample_rate

# def frame_generator(frame_duration_ms, audio, sample_rate):
#     """Generates audio frames from PCM audio data."""
#     n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
#     offset = 0
#     while offset + n < len(audio):
#         yield audio[offset:offset + n]
#         offset += n

# def analyze_speech_activity(file_path, aggressiveness=3):
#     """
#     Analyzes speech activity in an audio file.
    
#     Args:
#         file_path: Path to audio file
#         aggressiveness: VAD aggressiveness (1-3)
    
#     Returns:
#         Dictionary with speech activity metrics
#     """
#     audio, sample_rate = read_wave(file_path)
#     vad = webrtcvad.Vad(aggressiveness)
    
#     # Check if sample rate is one of the supported rates
#     if sample_rate not in [8000, 16000, 32000, 48000]:
#         print(f"Sample rate {sample_rate} not supported, trying to convert to 16000Hz")
#         # If necessary, could add sample rate conversion here
#         # For now we'll just return placeholder data
#         return {
#             "speech_percentage": 0,
#             "total_duration": 0,
#             "speech_duration": 0,
#             "silence_duration": 0,
#             "avg_speech_segment": 0,
#             "max_speech_segment": 0,
#             "speech_segments_count": 0,
#             "error": f"Sample rate {sample_rate} not supported by webrtcvad"
#         }
    
#     frame_duration_ms = 30  # 30ms frames
#     frames = list(frame_generator(frame_duration_ms, audio, sample_rate))
    
#     # Process frames
#     voiced_frames = []
#     for frame in frames:
#         if len(frame) < 640:  # Skip frames that are too short
#             continue
#         try:
#             is_speech = vad.is_speech(frame, sample_rate)
#             voiced_frames.append(is_speech)
#         except Exception as e:
#             print(f"Error processing frame: {e}")
#             continue
    
#     # Calculate speech activity metrics
#     total_frames = len(voiced_frames)
#     if total_frames == 0:
#         return {
#             "speech_percentage": 0,
#             "total_duration": 0,
#             "speech_duration": 0,
#             "silence_duration": 0,
#             "avg_speech_segment": 0,
#             "max_speech_segment": 0,
#             "speech_segments_count": 0
#         }
        
#     speech_frames = sum(voiced_frames)
#     silence_frames = total_frames - speech_frames
    
#     speech_percentage = (speech_frames / total_frames) * 100 if total_frames > 0 else 0
    
#     # Analyze speech runs (continuous speech segments)
#     speech_runs = []
#     current_run = 0
    
#     for is_speech in voiced_frames:
#         if is_speech:
#             current_run += 1
#         elif current_run > 0:
#             speech_runs.append(current_run)
#             current_run = 0
    
#     # Don't forget the last run if it's speech
#     if current_run > 0:
#         speech_runs.append(current_run)
    
#     # Convert frame counts to durations
#     frame_duration_sec = frame_duration_ms / 1000
    
#     if speech_runs:
#         avg_speech_run = np.mean(speech_runs) * frame_duration_sec
#         max_speech_run = max(speech_runs) * frame_duration_sec
#     else:
#         avg_speech_run = 0
#         max_speech_run = 0
    
#     # Calculate total speech and silence durations
#     total_duration = total_frames * frame_duration_sec
#     speech_duration = speech_frames * frame_duration_sec
#     silence_duration = silence_frames * frame_duration_sec
    
#     return {
#         "speech_percentage": speech_percentage,
#         "total_duration": total_duration,
#         "speech_duration": speech_duration,
#         "silence_duration": silence_duration,
#         "avg_speech_segment": avg_speech_run,
#         "max_speech_segment": max_speech_run,
#         "speech_segments_count": len(speech_runs) if speech_runs else 0
#     }

# # Enhanced speech analysis function
# def enhanced_speech_analysis(openai_api_key, deepgram_api_key, reference_text):
#     print("\n===== 🗣️ Enhanced Speech Analysis =====")
#     tts_engine = init_tts_engine()
    
#     print(f"\nPlease read the following text:")
#     print(f"\"{reference_text}\"")
#     speak_text(reference_text, tts_engine)
    
#     words_count = len(reference_text.split())
#     suggested_duration = max(5, min(15, words_count * 0.5))
    
#     duration_input = input(f"\nRecording duration in seconds (suggested: {suggested_duration:.1f}s, press Enter to use suggestion): ")
#     try:
#         duration = float(duration_input) if duration_input else suggested_duration
#     except ValueError:
#         duration = suggested_duration
#         print(f"Invalid input. Using suggested duration: {suggested_duration:.1f}s")
    
#     print("\nGet ready to speak...")
#     time.sleep(1)
#     audio_file = record_audio(duration=duration)
#     spoken_text, confidence, audio_file_path = transcribe_audio(audio_file, deepgram_api_key)
    
#     if not spoken_text:
#         print("Sorry, I couldn't understand what you said. Please try again.")
#         return
        
#     print(f"\nYou said: \"{spoken_text}\"")
#     print(f"Transcription confidence: {confidence:.2%}")
    
#     # ------------------ New Voice Analysis Section ------------------
#     print("\nAnalyzing your speech (audio features)...")
    
#     # 1. Process voice characteristics
#     audio_features = analyze_audio_features(audio_file_path)
    
#     # 2. Analyze voice characteristics
#     voice_characteristics = analyze_voice_characteristics(audio_features)
    
#     # 3. Detect speech activity
#     speech_activity = analyze_speech_activity(audio_file_path)
    
#     # 4. Create visualization of speech features
#     visualization_path = os.path.join(tempfile.gettempdir(), f"speech_analysis_{os.path.basename(audio_file_path)}.png")
#     visualize_audio_features(audio_file_path, save_path=visualization_path)
    
#     # ------------------ Original Text Analysis Section ------------------
#     print("\nAnalyzing your speech (text content)...")
#     text_analysis = analyze_speech(reference_text, spoken_text, confidence, audio_file_path, openai_api_key)
    
#     if "error" in text_analysis:
#         print(f"Error in text analysis: {text_analysis['error']}: {text_analysis['message']}")
#         text_analysis = None
    
#     # ------------------ Display Combined Results ------------------
#     print("\n===== 📊 Speech Analysis Results =====")
    
#     # Display voice characteristics
#     print("\n===== 🔊 Voice Characteristics =====")
#     print(f"Voice Type: {voice_characteristics['voice_type']}")
#     print(f"Pitch Variation: {voice_characteristics['pitch_variation']}")
#     print(f"Speech Rate: {voice_characteristics['speech_rate']}")
#     print(f"Fluency: {voice_characteristics['fluency']}")
#     print(f"Voice Brightness: {voice_characteristics['brightness']}")
#     print(f"Volume: {voice_characteristics['volume']}")
#     print(f"Volume Consistency: {voice_characteristics['volume_consistency']}")
#     print(f"\nVoice Summary: {voice_characteristics['summary']}")
    
#     # Display speech activity metrics
#     print("\n===== 📈 Speech Activity =====")
#     print(f"Speech Percentage: {speech_activity['speech_percentage']:.1f}%")
#     print(f"Total Duration: {speech_activity['total_duration']:.2f} seconds")
#     print(f"Speech Duration: {speech_activity['speech_duration']:.2f} seconds")
#     print(f"Silence Duration: {speech_activity['silence_duration']:.2f} seconds")
#     print(f"Average Speech Segment: {speech_activity['avg_speech_segment']:.2f} seconds")
#     print(f"Maximum Speech Segment: {speech_activity['max_speech_segment']:.2f} seconds")
#     print(f"Speech Segments Count: {speech_activity['speech_segments_count']}")
    
#     # Display audio metrics
#     print("\n===== 🎵 Audio Metrics =====")
#     print(f"Mean Pitch: {audio_features['pitch']['mean']:.1f} Hz")
#     print(f"Pitch Range: {audio_features['pitch']['range']:.1f} Hz")
#     print(f"Speech Rate: {audio_features['speech_rate']:.2f} syllables/second")
#     print(f"Speaking Rate (excluding pauses): {audio_features['speaking_rate']:.2f} syllables/second")
#     print(f"Pauses: {audio_features['pauses']['count']} (total {audio_features['pauses']['total_duration']:.2f}s)")
    
#     # Display text analysis if available
#     if text_analysis:
#         overall = text_analysis["overall"]
#         print("\n===== 📝 Text Analysis =====")
#         print(f"Overall Score: {min(int(overall['score']*100), 100)}%")
#         print(f"Summary: {overall['summary']}")
        
#         print("\nStrengths:")
#         for strength in overall["strengths"]:
#             print(f"• {strength}")
        
#         print("\nAreas for improvement:")
#         for area in overall["areas_for_improvement"]:
#             print(f"• {area}")
    
#     print(f"\nSpeech visualization saved to: {visualization_path}")
#     return audio_features, voice_characteristics, speech_activity, text_analysis



# def generate_proficiency_report(audio_features, voice_characteristics, speech_activity, text_analysis):
#     """Generate a detailed English proficiency report based on analysis results."""
#     report = {
#         "overall": {},
#         "grammar": {},
#         "pronunciation": {},
#         "accent": {},
#         "fluency": {},
#         "voice_characteristics": {}
#     }

#     # Overall Proficiency
#     overall_score = text_analysis["overall"]["score"] * 100 if text_analysis else 88  # Default to 88% if no text analysis
#     report["overall"]["score"] = min(int(overall_score), 100)
#     report["overall"]["summary"] = text_analysis["overall"]["summary"] if text_analysis else "Your spoken English is quite good, with strengths in pronunciation and fluency, but grammar needs work."

#     # Grammar
#     report["grammar"]["strengths"] = text_analysis["grammar"]["feedback"] if text_analysis else "Clear sentence structure overall."
#     report["grammar"]["issues"] = []
#     report["grammar"]["improvements"] = []
#     if text_analysis and text_analysis["grammar"]["errors"]:
#         report["grammar"]["issues"] = [f"{err['error']} (e.g., '{err['correction']}')" for err in text_analysis["grammar"]["errors"]]
#         report["grammar"]["improvements"] = ["Practice including subjects in every sentence (e.g., 'She walks' instead of 'Walks')."]
#     else:
#         report["grammar"]["issues"] = ["Likely omitting subjects in sentences."]
#         report["grammar"]["improvements"] = ["Use drills like 'Subject + Verb + Object' to reinforce complete sentences."]

#     # Pronunciation
#     report["pronunciation"]["strengths"] = "Clear enunciation and intelligibility." if text_analysis and "Pronunciation" in text_analysis["overall"]["strengths"] else "Good pronunciation based on fluid speech."
#     report["pronunciation"]["issues"] = []
#     report["pronunciation"]["improvements"] = []
#     speech_rate = audio_features["speech_rate"]
#     if speech_rate > 5:
#         report["pronunciation"]["issues"].append(f"Fast speech rate ({speech_rate:.2f} syllables/second) may lead to slurring.")
#         report["pronunciation"]["improvements"].append("Slow down to 4-5 syllables/second for clearer articulation.")
#     if text_analysis and text_analysis["pronunciation"]["mispronounced_words"]:
#         report["pronunciation"]["issues"].extend([f"Mispronounced '{word['word']}': {word['issue']}" for word in text_analysis["pronunciation"]["mispronounced_words"]])
#         report["pronunciation"]["improvements"].extend([word["suggestion"] for word in text_analysis["pronunciation"]["mispronounced_words"]])

#     # Accent
#     report["accent"]["strengths"] = f"Consistent delivery with {voice_characteristics['brightness']}."
#     report["accent"]["issues"] = []
#     report["accent"]["improvements"] = []
#     pitch_range = audio_features["pitch"]["range"]
#     if pitch_range < 90:
#         report["accent"]["issues"].append(f"Moderate pitch variation ({pitch_range:.1f} Hz) may sound monotonous.")
#         report["accent"]["improvements"].append("Increase pitch range (aim for 100 Hz) by emphasizing key words.")
#     if text_analysis:
#         report["accent"]["british_similarity"] = min(int(text_analysis["accent"]["british_similarity"] * 100), 100)
#         report["accent"]["american_similarity"] = min(int(text_analysis["accent"]["american_similarity"] * 100), 100)

#     # Fluency
#     report["fluency"]["strengths"] = f"Fluid speech with {speech_activity['speech_segments_count']} segments and no pauses."
#     report["fluency"]["issues"] = []
#     report["fluency"]["improvements"] = []
#     if speech_rate > 5:
#         report["fluency"]["issues"].append(f"Overly fast delivery ({speech_rate:.2f} syllables/second) may reduce comprehension.")
#         report["fluency"]["improvements"].append("Insert natural pauses (1-2 per 5 seconds) to improve listener understanding.")

#     # Voice Characteristics
#     report["voice_characteristics"]["summary"] = voice_characteristics["summary"]
#     report["voice_characteristics"]["issues"] = []
#     report["voice_characteristics"]["improvements"] = []
#     if "Quiet/soft speech" in voice_characteristics["volume"]:
#         report["voice_characteristics"]["issues"].append("Quiet volume may be hard to hear in noisy settings.")
#         report["voice_characteristics"]["improvements"].append("Practice projecting voice louder (aim for RMS 0.05-0.1).")
#     if "Moderate pitch variation" in voice_characteristics["pitch_variation"]:
#         report["voice_characteristics"]["improvements"].append("Exaggerate intonation for more dynamic speech.")

#     # Print the report
#     print("\n===== 📋 English Proficiency Report =====")
#     print(f"\n1. Overall Proficiency")
#     print(f"   Score: {report['overall']['score']}%")
#     print(f"   Summary: {report['overall']['summary']}")

#     print(f"\n2. Grammar")
#     print(f"   Strengths: {report['grammar']['strengths']}")
#     print(f"   Issues: {', '.join(report['grammar']['issues']) if report['grammar']['issues'] else 'None detected'}")
#     print(f"   Improvements: {', '.join(report['grammar']['improvements'])}")

#     print(f"\n3. Pronunciation")
#     print(f"   Strengths: {report['pronunciation']['strengths']}")
#     print(f"   Issues: {', '.join(report['pronunciation']['issues']) if report['pronunciation']['issues'] else 'None detected'}")
#     print(f"   Improvements: {', '.join(report['pronunciation']['improvements'])}")

#     print(f"\n4. Accent")
#     print(f"   Strengths: {report['accent']['strengths']}")
#     if "british_similarity" in report["accent"]:
#         print(f"   British Similarity: {report['accent']['british_similarity']}%")
#         print(f"   American Similarity: {report['accent']['american_similarity']}%")
#     print(f"   Issues: {', '.join(report['accent']['issues']) if report['accent']['issues'] else 'None detected'}")
#     print(f"   Improvements: {', '.join(report['accent']['improvements'])}")

#     print(f"\n5. Fluency")
#     print(f"   Strengths: {report['fluency']['strengths']}")
#     print(f"   Issues: {', '.join(report['fluency']['issues']) if report['fluency']['issues'] else 'None detected'}")
#     print(f"   Improvements: {', '.join(report['fluency']['improvements'])}")

#     print(f"\n6. Voice Characteristics")
#     print(f"   Summary: {report['voice_characteristics']['summary']}")
#     print(f"   Issues: {', '.join(report['voice_characteristics']['issues']) if report['voice_characteristics']['issues'] else 'None detected'}")
#     print(f"   Improvements: {', '.join(report['voice_characteristics']['improvements'])}")

#     return report

# def enhanced_speech_analysis(openai_api_key, deepgram_api_key, reference_text):
#     print("\n===== 🗣️ Enhanced Speech Analysis =====")
#     tts_engine = init_tts_engine()
    
#     print(f"\nPlease read the following text:")
#     print(f"\"{reference_text}\"")
#     speak_text(reference_text, tts_engine)
    
#     words_count = len(reference_text.split())
#     suggested_duration = max(5, min(15, words_count * 0.5))
    
#     duration_input = input(f"\nRecording duration in seconds (suggested: {suggested_duration:.1f}s, press Enter to use suggestion): ")
#     try:
#         duration = float(duration_input) if duration_input else suggested_duration
#     except ValueError:
#         duration = suggested_duration
#         print(f"Invalid input. Using suggested duration: {suggested_duration:.1f}s")
    
#     print("\nGet ready to speak...")
#     time.sleep(1)
#     audio_file = record_audio(duration=duration)
#     spoken_text, confidence, audio_file_path = transcribe_audio(audio_file, deepgram_api_key)
    
#     if not spoken_text:
#         print("Sorry, I couldn't understand what you said. Please try again.")
#         return
        
#     print(f"\nYou said: \"{spoken_text}\"")
#     print(f"Transcription confidence: {confidence:.2%}")
    
#     # Process voice characteristics
#     print("\nAnalyzing your speech (audio features)...")
#     audio_features = analyze_audio_features(audio_file_path)
#     voice_characteristics = analyze_voice_characteristics(audio_features)
#     speech_activity = analyze_speech_activity(audio_file_path)
    
#     # Create visualization
#     visualization_path = os.path.join(tempfile.gettempdir(), f"speech_analysis_{os.path.basename(audio_file_path)}.png")
#     visualize_audio_features(audio_file_path, save_path=visualization_path)
    
#     # Text analysis
#     print("\nAnalyzing your speech (text content)...")
#     text_analysis = analyze_speech(reference_text, spoken_text, confidence, audio_file_path, openai_api_key)
#     if "error" in text_analysis:
#         print(f"Error in text analysis: {text_analysis['error']}: {text_analysis['message']}")
#         text_analysis = None
    
#     # Generate and display proficiency report
#     report = generate_proficiency_report(audio_features, voice_characteristics, speech_activity, text_analysis)
    
#     print(f"\nSpeech visualization saved to: {visualization_path}")
#     return audio_features, voice_characteristics, speech_activity, text_analysis, report




# def main():
#     print("=== 🎓 Advanced Speech Analysis Tool ===")
    
#     load_dotenv()
#     openai_api_key = os.getenv('OPENAI_API_KEY')
#     deepgram_api_key = os.getenv('DEEPGRAM_API_KEY')

#     if not openai_api_key or not deepgram_api_key:
#         print("Error: API keys not found in .env file")
#         exit(1)
    
#     while True:
#         print("\n1. Basic Text Analysis")
#         print("2. Enhanced Voice Analysis")
#         print("3. Record and Analyze Custom Text")
#         print("4. Exit")
        
#         choice = input("\nSelect an option (1-4): ")
        
#         if choice == "1":
#             print("\nAvailable test texts:")
#             for key, text_data in TEST_TEXTS.items():
#                 print(f"{key}. {text_data['title']}")
            
#             text_choice = input("\nSelect a text (1-4): ")
#             if text_choice in TEST_TEXTS:
#                 reference_text = TEST_TEXTS[text_choice]["text"]
#                 simple_speech_analysis(openai_api_key, deepgram_api_key, reference_text)
#             else:
#                 print("Invalid text selection. Please try again.")
                
#         elif choice == "2":
#             print("\nAvailable test texts:")
#             for key, text_data in TEST_TEXTS.items():
#                 print(f"{key}. {text_data['title']}")
            
#             text_choice = input("\nSelect a text (1-4): ")
#             if text_choice in TEST_TEXTS:
#                 reference_text = TEST_TEXTS[text_choice]["text"]
#                 enhanced_speech_analysis(openai_api_key, deepgram_api_key, reference_text)
#             else:
#                 print("Invalid text selection. Please try again.")
        
#         elif choice == "3":
#             custom_text = input("\nEnter your custom text for analysis: ")
#             if custom_text:
#                 enhanced_speech_analysis(openai_api_key, deepgram_api_key, custom_text)
#             else:
#                 print("No text entered. Please try again.")
                
#         elif choice == "4":
#             print("\nThank you for using the Advanced Speech Analysis Tool!")
#             break
#         else:
#             print("Invalid choice. Please select 1-4.")

# if __name__ == "__main__":
#     main()  







# import openai
# import sounddevice as sd
# import numpy as np
# import wave
# import requests
# import pyttsx3
# import tempfile
# import os
# import json
# import time
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import webrtcvad
# from datetime import datetime
# from dotenv import load_dotenv

# # Predefined test sentence
# TEST_TEXT = "She walks to the park every morning with her friendly golden retriever."

# # Function to initialize text-to-speech engine
# def init_tts_engine():
#     engine = pyttsx3.init()
#     engine.setProperty('rate', 150)
#     engine.setProperty('volume', 0.9)
#     return engine

# # Function to speak text using TTS engine
# def speak_text(text, engine):
#     print(f"🔊 AI: {text}")
#     engine.say(text)
#     engine.runAndWait()

# # Function to record and save audio
# def record_audio(duration=5, sample_rate=16000):
#     print("🎤 Speak now...")
#     recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
#     sd.wait()
#     recording = (recording * 32767).astype(np.int16)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"recording_{timestamp}.wav"
#     with wave.open(filename, 'wb') as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)
#         wf.setframerate(sample_rate)
#         wf.writeframes(recording.tobytes())
#     print(f"Audio saved as: {filename}")
#     return filename

# # Function to transcribe audio using Deepgram
# def transcribe_audio(audio_file_path, deepgram_api_key):
#     url = "https://api.deepgram.com/v1/listen"
#     headers = {"Authorization": f"Token {deepgram_api_key}"}
#     params = {"model": "nova-2", "language": "en", "punctuate": "true", "diarize": "false", "detect_language": "true", "utterances": "true", "detect_topics": "true", "summarize": "v2"}
#     with open(audio_file_path, 'rb') as audio:
#         response = requests.post(url, headers=headers, params=params, data=audio)
#     if response.status_code == 200:
#         result = response.json()
#         transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
#         confidence = result["results"]["channels"][0]["alternatives"][0]["confidence"]
#         return transcript, confidence, audio_file_path
#     print(f"Deepgram error: {response.status_code} - {response.text}")
#     os.unlink(audio_file_path)
#     return None, None, audio_file_path

# # Function to analyze speech for language proficiency
# def analyze_speech(reference_text, spoken_text, confidence, audio_file_path, openai_api_key):
#     openai.api_key = openai_api_key
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": """
#                     You are a friendly English teacher who gives clear, personal feedback. 
#                     Compare the spoken text to the reference text. 
#                     Point out what’s good and what needs work in grammar, pronunciation, fluency, and accent. 
#                     Suggest simple ways to improve. 
#                     Use the transcription confidence score ({confidence}) to judge pronunciation if below 0.95. 
#                     Scores must be between 0.0 and 1.0 (0% to 100%).
#                 """},
#                 {"role": "user", "content": f"""
#                     Reference text: "{reference_text}"
#                     Spoken text: "{spoken_text}"
#                     Transcription confidence: {confidence}
#                     Audio file: {audio_file_path}
#                     Provide analysis in strict JSON format:
#                     {{
#                         "grammar": {{"errors": [{{"error": "", "correction": ""}}], "score": 0.0, "feedback": ""}},
#                         "pronunciation": {{"mispronounced_words": [{{"word": "", "issue": "", "suggestion": ""}}], "score": 0.0, "feedback": ""}},
#                         "accent": {{"patterns": [], "feedback": ""}},
#                         "fluency": {{"issues": [], "score": 0.0, "feedback": ""}},
#                         "punctuation": {{"issues": [], "score": 0.0, "feedback": ""}},
#                         "overall": {{"score": 0.0, "strengths": [], "areas_for_improvement": [], "summary": ""}}
#                     }}
#                 """}
#             ],
#             temperature=0.7
#         )
#         content = response.choices[0].message.content.strip()
#         json_start = content.index('{')
#         json_end = content.rindex('}') + 1
#         json_content = content[json_start:json_end]
#         return json.loads(json_content)
#     except (ValueError, json.JSONDecodeError) as e:
#         print(f"JSON parsing error: {str(e)} - Response content: {content}")
#         return {
#             "grammar": {"errors": [], "score": 0.5, "feedback": "I couldn’t check this fully, but keep practicing!"},
#             "pronunciation": {"mispronounced_words": [], "score": 0.5, "feedback": "Sounded okay, but let’s keep working!"},
#             "accent": {"patterns": [], "feedback": "Your style is fine for now!"},
#             "fluency": {"issues": [], "score": 0.5, "feedback": "You’re getting there!"},
#             "punctuation": {"issues": [], "score": 0.5, "feedback": "Hard to tell, but you’re doing fine!"},
#             "overall": {"score": 0.5, "strengths": ["Trying hard"], "areas_for_improvement": ["Keep practicing"], "summary": "Good effort, let’s polish it up!"}
#         }
#     except Exception as e:
#         print(f"API error: {str(e)}")
#         return {"error": "Failed to analyze speech", "message": str(e)}

# # Function to analyze audio features
# def analyze_audio_features(audio_file_path):
#     y, sr = librosa.load(audio_file_path, sr=None)
#     duration = librosa.get_duration(y=y, sr=sr)
#     f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
#     f0_cleaned = f0[~np.isnan(f0)]
#     mean_f0, std_f0, min_f0, max_f0 = (np.mean(f0_cleaned), np.std(f0_cleaned), np.min(f0_cleaned), np.max(f0_cleaned)) if len(f0_cleaned) > 0 else (0, 0, 0, 0)
#     rms = librosa.feature.rms(y=y)[0]
#     onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
#     speech_rate = len(onsets) / duration if len(onsets) > 0 else 0
#     non_silent_intervals = librosa.effects.split(y, top_db=30)
#     silent_intervals = []
#     if len(non_silent_intervals) > 0:
#         for i in range(len(non_silent_intervals) - 1):
#             if (non_silent_intervals[i+1][0] / sr) - (non_silent_intervals[i][1] / sr) > 0.2:
#                 silent_intervals.append(((non_silent_intervals[i][1] / sr), (non_silent_intervals[i+1][0] / sr)))
#         if non_silent_intervals[0][0] > 0 and (non_silent_intervals[0][0] / sr) > 0.2:
#             silent_intervals.insert(0, (0, non_silent_intervals[0][0] / sr))
#         if (duration - (non_silent_intervals[-1][1] / sr)) > 0.2:
#             silent_intervals.append((non_silent_intervals[-1][1] / sr, duration))
#     return {
#         "pitch": {"mean": mean_f0, "std": std_f0, "range": max_f0 - min_f0 if max_f0 > 0 and min_f0 > 0 else 0},
#         "volume": {"mean": np.mean(rms)},
#         "speech_rate": speech_rate,
#         "pauses": {"count": len(silent_intervals)}
#     }

# # Function to visualize audio features
# def visualize_audio_features(audio_file_path, save_path=None):
#     y, sr = librosa.load(audio_file_path, sr=None)
#     plt.figure(figsize=(10, 8))
#     plt.subplot(3, 1, 1)
#     librosa.display.waveshow(y, sr=sr)
#     plt.title('Waveform')
#     plt.subplot(3, 1, 2)
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
#     librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogram')
#     plt.subplot(3, 1, 3)
#     f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
#     times = librosa.times_like(f0, sr=sr)
#     plt.plot(times, f0, label='F0', color='b')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Frequency (Hz)')
#     plt.title('Pitch Contour')
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#         return save_path
#     else:
#         plt.show()
#         return None

# # Function to analyze voice characteristics
# def analyze_voice_characteristics(audio_features):
#     analysis = {}
#     mean_pitch = audio_features["pitch"]["mean"]
#     analysis["voice_type"] = "adult male" if 85 <= mean_pitch <= 180 else "adult female" if 165 <= mean_pitch <= 255 else "hard to tell"
#     pitch_std = audio_features["pitch"]["std"]
#     analysis["pitch_variation"] = "stays pretty flat" if pitch_std < 10 else "has some ups and downs" if pitch_std < 30 else "really lively"
#     speech_rate = audio_features["speech_rate"]
#     analysis["speech_rate"] = "fast" if speech_rate >= 4 else "steady" if 2 <= speech_rate < 4 else "slow"
#     pause_count = audio_features["pauses"]["count"]
#     analysis["fluency"] = "smooth with no breaks" if pause_count < 2 else "normal with some pauses" if pause_count <= 5 else "choppy"
#     analysis["volume"] = "soft" if audio_features["volume"]["mean"] < 0.05 else "normal" if audio_features["volume"]["mean"] <= 0.1 else "loud"
#     analysis["brightness"] = "warm" if audio_features["pitch"]["mean"] < 150 else "bright"
#     analysis["summary"] = f"Your voice sounds like an {analysis['voice_type']}’s, {analysis['pitch_variation']}, and you speak {analysis['speech_rate']} and {analysis['fluency']}. It has a {analysis['brightness']} tone and is {analysis['volume']}."
#     return analysis

# # Function to read wave file
# def read_wave(path):
#     with wave.open(path, 'rb') as wf:
#         num_channels = wf.getnchannels()
#         sample_width = wf.getsampwidth()
#         sample_rate = wf.getframerate()
#         pcm_data = wf.readframes(wf.getnframes())
#         return pcm_data, sample_rate

# # Function to generate audio frames
# def frame_generator(frame_duration_ms, audio, sample_rate):
#     n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
#     offset = 0
#     while offset + n < len(audio):
#         yield audio[offset:offset + n]
#         offset += n

# # Function to analyze speech activity
# def analyze_speech_activity(file_path, aggressiveness=3):
#     audio, sample_rate = read_wave(file_path)
#     vad = webrtcvad.Vad(aggressiveness)
#     if sample_rate not in [8000, 16000, 32000, 48000]:
#         return {"speech_segments_count": 0}
#     frame_duration_ms = 30
#     frames = list(frame_generator(frame_duration_ms, audio, sample_rate))
#     voiced_frames = [vad.is_speech(frame, sample_rate) for frame in frames if len(frame) >= 640]
#     speech_runs = []
#     current_run = 0
#     for is_speech in voiced_frames:
#         if is_speech:
#             current_run += 1
#         elif current_run > 0:
#             speech_runs.append(current_run)
#             current_run = 0
#     if current_run > 0:
#         speech_runs.append(current_run)
#     return {"speech_segments_count": len(speech_runs) if speech_runs else 0}

# # Function to generate personal feedback report
# def generate_proficiency_report(audio_features, voice_characteristics, speech_activity, text_analysis):
#     report = {
#         "overall": {"score": min(int(text_analysis["overall"]["score"] * 100) if text_analysis else 88, 100), "summary": text_analysis["overall"]["summary"] if text_analysis else "You’re doing great, just a few things to tweak!"},
#         "grammar": {"good": text_analysis["grammar"]["feedback"] if text_analysis and text_analysis["grammar"]["score"] > 0.8 else "You’re trying hard!", "work": ", ".join([f"you said '{err['error']}' instead of '{err['correction']}'" for err in text_analysis["grammar"]["errors"]]) if text_analysis and text_analysis["grammar"]["errors"] else "sometimes you skip who’s doing it", "tip": "Try starting with 'She' or 'I' every time."},
#         "pronunciation": {"good": "You sound clear!" if text_analysis and "Pronunciation" in text_analysis["overall"]["strengths"] else "I can understand you well!", "work": "you’re talking fast" if audio_features['speech_rate'] > 5 else "" + (", ".join([f"'{w['word']}' didn’t sound quite right" for w in text_analysis["pronunciation"]["mispronounced_words"]]) if text_analysis and text_analysis["pronunciation"]["mispronounced_words"] else ""), "tip": "Slow down a little." if audio_features['speech_rate'] > 5 else "Keep practicing tricky words!"},
#         "accent": {"good": "Your voice has a nice tone!", "work": "it stays flat sometimes" if audio_features['pitch']['range'] < 90 else "", "tip": "Let your voice rise and fall more."},
#         "fluency": {"good": f"You flow smoothly with {speech_activity['speech_segments_count']} parts!", "work": "you’re going fast" if audio_features['speech_rate'] > 5 else "", "tip": "Take a breath between ideas." if audio_features['speech_rate'] > 5 else "You’re doing awesome!"},
#         "voice": {"good": voice_characteristics["summary"], "work": "it’s soft" if "soft" in voice_characteristics["volume"] else "", "tip": "Speak up a bit!" if "soft" in voice_characteristics["volume"] else "Add some excitement!"}
#     }
#     for section in ["pronunciation", "accent", "fluency", "voice"]:
#         report[section]["work"] = report[section]["work"].strip() or "nothing big here"

#     print("\n===== 📋 Hey, Here’s Your Feedback! =====")
#     print(f"Overall: You got {report['overall']['score']}%! {report['overall']['summary']}")
#     print(f"Grammar: Nice job: {report['grammar']['good']} | Work on: {report['grammar']['work']} | Try this: {report['grammar']['tip']}")
#     print(f"Pronunciation: Nice job: {report['pronunciation']['good']} | Work on: {report['pronunciation']['work']} | Try this: {report['pronunciation']['tip']}")
#     print(f"Accent: Nice job: {report['accent']['good']} | Work on: {report['accent']['work']} | Try this: {report['accent']['tip']}")
#     print(f"Fluency: Nice job: {report['fluency']['good']} | Work on: {report['fluency']['work']} | Try this: {report['fluency']['tip']}")
#     print(f"Voice: Nice job: {report['voice']['good']} | Work on: {report['voice']['work']} | Try this: {report['voice']['tip']}")
#     return report

# # Speech analysis function
# def speech_analysis(openai_api_key, deepgram_api_key, reference_text=TEST_TEXT):
#     print("\n===== 🗣️ Speech Analysis =====")
#     tts_engine = init_tts_engine()
#     print(f"\nPlease read this:")
#     print(f"\"{reference_text}\"")
#     speak_text(reference_text, tts_engine)
#     suggested_duration = 6.0
#     duration_input = input(f"\nRecording duration (suggested: {suggested_duration}s, press Enter to use): ")
#     duration = float(duration_input) if duration_input else suggested_duration
#     print("\nGet ready to speak...")
#     time.sleep(1)
#     audio_file = record_audio(duration=duration)
#     spoken_text, confidence, audio_file_path = transcribe_audio(audio_file, deepgram_api_key)
#     if not spoken_text:
#         print("Sorry, I couldn’t hear you well. Try again!")
#         return
#     print(f"\nYou said: \"{spoken_text}\"")
#     print(f"Confidence: {confidence:.2%}")
#     print("\nChecking your voice...")
#     audio_features = analyze_audio_features(audio_file_path)
#     voice_characteristics = analyze_voice_characteristics(audio_features)
#     speech_activity = analyze_speech_activity(audio_file_path)
#     visualization_path = os.path.join(tempfile.gettempdir(), f"speech_analysis_{os.path.basename(audio_file_path)}.png")
#     visualize_audio_features(audio_file_path, save_path=visualization_path)
#     print("\nChecking your words...")
#     text_analysis = analyze_speech(reference_text, spoken_text, confidence, audio_file_path, openai_api_key)
#     if "error" in text_analysis:
#         print(f"Word check error: {text_analysis['error']}: {text_analysis['message']}")
#         text_analysis = None
#     report = generate_proficiency_report(audio_features, voice_characteristics, speech_activity, text_analysis)
#     print(f"\nYour voice graph is saved at: {visualization_path}")
#     return audio_features, voice_characteristics, speech_activity, text_analysis, report

# # Main function
# def main():
#     print("=== 🎓 Speech Analysis Tool ===")
#     load_dotenv()
#     openai_api_key = os.getenv('OPENAI_API_KEY')
#     deepgram_api_key = os.getenv('DEEPGRAM_API_KEY')
#     if not openai_api_key or not deepgram_api_key:
#         print("Error: API keys not found in .env file")
#         exit(1)
#     while True:
#         print("\n1. Start Speech Analysis")
#         print("2. Exit")
#         choice = input("\nChoose (1-2): ")
#         if choice == "1":
#             speech_analysis(openai_api_key, deepgram_api_key)
#         elif choice == "2":
#             print("\nThanks for practicing with me!")
#             break
#         else:
#             print("Oops, pick 1 or 2!")

# if __name__ == "__main__":
#     main()




########################################################################################################


import openai
import sounddevice as sd
import numpy as np
import wave
import requests
import pyttsx3
import tempfile
import os
import json
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import webrtcvad
from datetime import datetime
from dotenv import load_dotenv

# Predefined practice sentences at different difficulty levels
PRACTICE_SENTENCES = {
    "beginner": [
        "She walks to the park every morning with her dog.",
        "I like to eat breakfast before I go to work.",
        "The weather is nice today so we can go outside."
    ],
    "intermediate": [
        "She walks to the park every morning with her friendly golden retriever.",
        "The museum had an interesting exhibition about ancient civilizations.",
        "I believe we should reconsider our approach to solving this problem."
    ],
    "advanced": [
        "The professor eloquently articulated his perspective on the controversial philosophical theory.",
        "Despite the inclement weather, the determined hikers persevered through the mountainous terrain.",
        "The intricate relationship between economic policies and environmental sustainability requires nuanced analysis."
    ]
}

# Learning goals for feedback focus
LEARNING_GOALS = [
    "Overall improvement",
    "Grammar focus",
    "Pronunciation focus",
    "Fluency focus",
    "Accent reduction"
]

# Function to initialize text-to-speech engine
def init_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    voices = engine.getProperty('voices')
    if voices:
        female_voices = [v for v in voices if 'female' in v.name.lower()]
        if female_voices:
            engine.setProperty('voice', female_voices[0].id)
    return engine

# Function to speak text using TTS engine with emotion
def speak_text(text, engine, emotion="neutral"):
    print(f"🔊 AI: {text}")
    if emotion == "excited":
        engine.setProperty('rate', 170)
        engine.setProperty('volume', 1.0)
    elif emotion == "calm":
        engine.setProperty('rate', 130)
        engine.setProperty('volume', 0.8)
    elif emotion == "questioning":
        text = text + "?"
    engine.say(text)
    engine.runAndWait()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)

# Function to record and save audio with countdown
def record_audio(duration=5, sample_rate=16000):
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("🎤 Speak now...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    for i in range(int(duration)):
        time.sleep(1)
        print("●" * (i+1) + "○" * (int(duration)-i-1), end="\r")
    sd.wait()
    print("\nProcessing your speech...")
    recording = (recording * 32767).astype(np.int16)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("user_recordings", exist_ok=True)
    filename = f"user_recordings/recording_{timestamp}.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())
    print(f"Audio saved as: {filename}")
    return filename

# Function to transcribe audio using Deepgram with word-level confidence analysis
def transcribe_audio(audio_file_path, deepgram_api_key):
    url = "https://api.deepgram.com/v1/listen"
    headers = {"Authorization": f"Token {deepgram_api_key}"}
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
        with open(audio_file_path, 'rb') as audio:
            response = requests.post(url, headers=headers, params=params, data=audio)
        
        if response.status_code == 200:
            result = response.json()
            transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            confidence = result["results"]["channels"][0]["alternatives"][0]["confidence"]
            words = result["results"]["channels"][0]["alternatives"][0].get("words", [])
            
            # Flag low-confidence words as potential mispronunciations
            potential_errors = [w for w in words if w["confidence"] < 0.9]  # Threshold adjustable
            if potential_errors:
                print("Potential pronunciation issues detected in words:", 
                      [f"{w['word']} (confidence: {w['confidence']:.2f})" for w in potential_errors])
            
            return transcript, confidence, audio_file_path, words
        print(f"Deepgram error: {response.status_code} - {response.text}")
        return None, None, audio_file_path, []
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return None, None, audio_file_path, []

# Function to analyze speech with word-level confidence for pronunciation
def analyze_speech(reference_text, spoken_text, confidence, audio_file_path, openai_api_key, learning_goal="Overall improvement", learner_level="intermediate", words=[]):
    openai.api_key = openai_api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"""
                    You are a supportive English teacher who gives clear, personalized feedback. 
                    Compare the spoken text to the reference text.
                    Focus primarily on the student's chosen learning goal: {learning_goal}.
                    The student is at {learner_level} level.
                    Point out what's good and what needs work in grammar, pronunciation, fluency, and accent. 
                    Use word-level confidence scores from the 'words' data to identify potential mispronunciations.
                    If a word has a confidence below 0.9, consider it potentially mispronounced even if it matches the reference.
                    Suggest simple, actionable ways to improve that are appropriate for their level.
                    Use the transcription confidence score ({confidence}) to judge overall pronunciation if below 0.95.
                    For beginners, focus on basic patterns and common mistakes.
                    For intermediate learners, focus on natural expression and more complex structures.
                    For advanced learners, focus on subtle nuances, idioms, and native-like fluency.
                    All scores must be between 0.0 and 1.0 (0% to 100%).
                """},
                {"role": "user", "content": f"""
                    Reference text: "{reference_text}"
                    Spoken text: "{spoken_text}"
                    Transcription confidence: {confidence}
                    Audio file: {audio_file_path}
                    Word-level data: {json.dumps(words)}
                    Learning goal: {learning_goal}
                    Learner level: {learner_level}
                    
                    Provide analysis in strict JSON format:
                    {{
                        "grammar": {{"errors": [{{"error": "", "correction": "", "explanation": ""}}], "score": 0.0, "feedback": ""}},
                        "pronunciation": {{"mispronounced_words": [{{"word": "", "issue": "", "suggestion": ""}}], "score": 0.0, "feedback": ""}},
                        "accent": {{"patterns": [], "feedback": "", "exercises": []}},
                        "fluency": {{"issues": [], "score": 0.0, "feedback": "", "exercises": []}},
                        "vocabulary": {{"appropriate": true, "suggestions": [], "feedback": ""}},
                        "overall": {{"score": 0.0, "strengths": [], "areas_for_improvement": [], "summary": "", "next_steps": []}}
                    }}
                """}
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        json_start = content.index('{')
        json_end = content.rindex('}') + 1
        json_content = content[json_start:json_end]
        return json.loads(json_content)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"JSON parsing error: {str(e)} - Response content: {content if 'content' in locals() else 'No content'}")
        return {
            "grammar": {"errors": [], "score": 0.5, "feedback": "I couldn't check this fully, but keep practicing!"},
            "pronunciation": {"mispronounced_words": [], "score": 0.5, "feedback": "Sounded okay, but let's keep working!"},
            "accent": {"patterns": [], "feedback": "Your style is fine for now!", "exercises": ["Practice with tongue twisters"]},
            "fluency": {"issues": [], "score": 0.5, "feedback": "You're getting there!", "exercises": ["Read aloud for 5 minutes daily"]},
            "vocabulary": {"appropriate": True, "suggestions": [], "feedback": "Good word choices!"},
            "overall": {"score": 0.5, "strengths": ["Trying hard"], "areas_for_improvement": ["Keep practicing"], 
                        "summary": "Good effort, let's polish it up!", "next_steps": ["Practice daily for 10 minutes"]}
        }
    except Exception as e:
        print(f"API error: {str(e)}")
        return {"error": "Failed to analyze speech", "message": str(e)}

# Function to analyze audio features with more detailed metrics
def analyze_audio_features(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                               fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_cleaned = f0[~np.isnan(f0)]
    mean_f0 = np.mean(f0_cleaned) if len(f0_cleaned) > 0 else 0
    std_f0 = np.std(f0_cleaned) if len(f0_cleaned) > 0 else 0
    min_f0 = np.min(f0_cleaned) if len(f0_cleaned) > 0 else 0
    max_f0 = np.max(f0_cleaned) if len(f0_cleaned) > 0 else 0
    pitch_range = max_f0 - min_f0 if len(f0_cleaned) > 0 else 0
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    speech_rate = len(onsets) / duration if len(onsets) > 0 else 0
    non_silent_intervals = librosa.effects.split(y, top_db=30)
    silent_intervals = []
    if len(non_silent_intervals) > 0:
        for i in range(len(non_silent_intervals) - 1):
            silent_start = non_silent_intervals[i][1] / sr
            silent_end = non_silent_intervals[i+1][0] / sr
            if silent_end - silent_start > 0.2:
                silent_intervals.append((silent_start, silent_end))
        if non_silent_intervals[0][0] > 0 and (non_silent_intervals[0][0] / sr) > 0.2:
            silent_intervals.insert(0, (0, non_silent_intervals[0][0] / sr))
        if (duration - (non_silent_intervals[-1][1] / sr)) > 0.2:
            silent_intervals.append((non_silent_intervals[-1][1] / sr, duration))
    total_pause_time = sum(end - start for start, end in silent_intervals)
    avg_pause_length = total_pause_time / len(silent_intervals) if silent_intervals else 0
    segment_durations = [(interval[1] - interval[0]) / sr for interval in non_silent_intervals] if len(non_silent_intervals) > 1 else []
    rhythm_regularity = 1 - (np.std(segment_durations) / np.mean(segment_durations)) if segment_durations else 0
    return {
        "pitch": {"mean": mean_f0, "std": std_f0, "min": min_f0, "max": max_f0, "range": pitch_range},
        "volume": {"mean": mean_rms, "std": std_rms, "variability": std_rms / mean_rms if mean_rms > 0 else 0},
        "speech_rate": speech_rate,
        "pauses": {"count": len(silent_intervals), "total_time": total_pause_time, "average_length": avg_pause_length},
        "rhythm": {"regularity": rhythm_regularity},
        "duration": duration
    }

# Function to visualize audio features
def visualize_audio_features(audio_file_path, save_path=None, word_timings=None):
    y, sr = librosa.load(audio_file_path, sr=None)
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    non_silent_intervals = librosa.effects.split(y, top_db=30)
    for interval in non_silent_intervals:
        start = interval[0] / sr
        end = interval[1] / sr
        plt.axvspan(start, end, color='green', alpha=0.2)
    if word_timings:
        for word in word_timings:
            plt.axvline(x=word.get('start', 0), color='r', linestyle='--', alpha=0.5)
            plt.text(word.get('start', 0), max(abs(y))*0.8, word.get('word', ''), rotation=90, fontsize=8)
    plt.title('Speech Waveform (Green = Speaking)')
    plt.subplot(4, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (Frequency Components)')
    plt.subplot(4, 1, 3)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    times = librosa.times_like(f0, sr=sr)
    plt.plot(times, f0, label='Pitch', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Pitch Contour (Melody of Your Voice)')
    plt.subplot(4, 1, 4)
    rms = librosa.feature.rms(y=y)[0]
    frames = range(len(rms))
    t = librosa.frames_to_time(frames, sr=sr)
    plt.plot(t, rms, color='orange')
    plt.title('Volume/Energy Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        return save_path
    else:
        plt.show()
        return None

# Function to analyze voice characteristics
def analyze_voice_characteristics(audio_features):
    analysis = {}
    mean_pitch = audio_features["pitch"]["mean"]
    if 85 <= mean_pitch <= 180:
        analysis["voice_type"] = "adult male"
    elif 165 <= mean_pitch <= 255:
        analysis["voice_type"] = "adult female"
    elif mean_pitch > 255:
        analysis["voice_type"] = "higher pitched"
    else:
        analysis["voice_type"] = "lower pitched"
    pitch_std = audio_features["pitch"]["std"]
    if pitch_std < 10:
        analysis["pitch_variation"] = "stays fairly flat (monotone)"
        analysis["expressiveness_score"] = 0.3
    elif pitch_std < 30:
        analysis["pitch_variation"] = "has some natural ups and downs"
        analysis["expressiveness_score"] = 0.6
    else:
        analysis["pitch_variation"] = "is very expressive with good variation"
        analysis["expressiveness_score"] = 0.9
    speech_rate = audio_features["speech_rate"]
    if speech_rate >= 4:
        analysis["speech_rate"] = "fast"
        analysis["pace_score"] = 0.7
    elif 2 <= speech_rate < 4:
        analysis["speech_rate"] = "at a good conversational pace"
        analysis["pace_score"] = 0.9
    else:
        analysis["speech_rate"] = "somewhat slow and deliberate"
        analysis["pace_score"] = 0.5
    pause_count = audio_features["pauses"]["count"]
    if pause_count < 2:
        analysis["fluency"] = "smooth with almost no breaks"
        analysis["fluency_score"] = 0.9
    elif pause_count <= 5:
        analysis["fluency"] = "natural with appropriate pauses"
        analysis["fluency_score"] = 0.8
    else:
        analysis["fluency"] = "has several pauses that affect the flow"
        analysis["fluency_score"] = 0.5
    if audio_features["volume"]["mean"] < 0.05:
        analysis["volume"] = "quite soft (could be louder)"
        analysis["volume_score"] = 0.4
    elif audio_features["volume"]["mean"] <= 0.1:
        analysis["volume"] = "at a good level"
        analysis["volume_score"] = 0.8
    else:
        analysis["volume"] = "strong and clear"
        analysis["volume_score"] = 0.9
    analysis["brightness"] = "warm and deep" if audio_features["pitch"]["mean"] < 150 else "bright and clear"
    analysis["rhythm"] = "has a consistent, regular rhythm" if audio_features["rhythm"]["regularity"] > 0.7 else "has a varied, natural rhythm"
    analysis["overall_voice_score"] = (
        analysis.get("expressiveness_score", 0.5) +
        analysis.get("pace_score", 0.5) +
        analysis.get("fluency_score", 0.5) +
        analysis.get("volume_score", 0.5)
    ) / 4
    analysis["summary"] = f"Your voice sounds like an {analysis['voice_type']}'s, {analysis['pitch_variation']}, and you speak {analysis['speech_rate']} and {analysis['fluency']}. It has a {analysis['brightness']} tone and is {analysis['volume']}."
    analysis["tips"] = []
    if analysis.get("expressiveness_score", 0) < 0.6:
        analysis["tips"].append("Try varying your pitch more when asking questions or expressing emotions")
    if analysis.get("fluency_score", 0) < 0.7:
        analysis["tips"].append("Practice reading aloud to reduce pauses between words")
    if analysis.get("volume_score", 0) < 0.6:
        analysis["tips"].append("Speak a bit louder to sound more confident")
    if speech_rate > 4:
        analysis["tips"].append("Slow down slightly to improve clarity")
    elif speech_rate < 2:
        analysis["tips"].append("Try to speak a bit faster to sound more natural")
    return analysis

# Function to read wave file
def read_wave(path):
    with wave.open(path, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

# Function to generate audio frames
def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n

# Function to analyze speech activity
def analyze_speech_activity(file_path, aggressiveness=3):
    try:
        audio, sample_rate = read_wave(file_path)
        vad = webrtcvad.Vad(aggressiveness)
        if sample_rate not in [8000, 16000, 32000, 48000]:
            return {"speech_segments_count": 0, "error": f"Unsupported sample rate: {sample_rate}"}
        frame_duration_ms = 30
        frames = list(frame_generator(frame_duration_ms, audio, sample_rate))
        if not frames:
            return {"speech_segments_count": 0, "error": "No frames generated"}
        valid_frames = [frame for frame in frames if len(frame) >= 640]
        if not valid_frames:
            return {"speech_segments_count": 0, "error": "No valid frames after filtering"}
        voiced_frames = [vad.is_speech(frame, sample_rate) for frame in valid_frames]
        speech_runs = []
        current_run = 0
        for is_speech in voiced_frames:
            if is_speech:
                current_run += 1
            elif current_run > 0:
                speech_runs.append(current_run)
                current_run = 0
        if current_run > 0:
            speech_runs.append(current_run)
        total_frames = len(voiced_frames)
        speech_frames = sum(voiced_frames)
        speech_percentage = (speech_frames / total_frames) * 100 if total_frames > 0 else 0
        return {
            "speech_segments_count": len(speech_runs) if speech_runs else 0,
            "speech_percentage": speech_percentage,
            "average_segment_length": np.mean(speech_runs) if speech_runs else 0,
            "longest_segment": max(speech_runs) if speech_runs else 0,
            "shortest_segment": min(speech_runs) if speech_runs else 0
        }
    except Exception as e:
        print(f"Speech activity analysis error: {str(e)}")
        return {"speech_segments_count": 0, "error": str(e)}

# Function to generate personalized progress suggestions
def generate_progress_suggestions(text_analysis, voice_characteristics):
    suggestions = []
    if text_analysis and "grammar" in text_analysis:
        if text_analysis["grammar"]["score"] < 0.7:
            suggestions.append({
                "area": "Grammar", 
                "tip": "Try simple sentences first. Start with Subject + Verb + Object.",
                "exercise": "Write 5 simple sentences about your daily routine each day."
            })
        elif text_analysis["grammar"]["score"] < 0.9:
            suggestions.append({
                "area": "Grammar",
                "tip": "Practice connecting ideas with words like 'and', 'but', 'because'.",
                "exercise": "Take a paragraph from a book and rewrite it in your own words."
            })
    if "pronunciation" in text_analysis:
        if text_analysis["pronunciation"]["score"] < 0.7:
            suggestions.append({
                "area": "Pronunciation",
                "tip": "Focus on one sound at a time. Master the 'th' sound first.",
                "exercise": "Practice saying: 'The three thieves thought thoroughly.'"
            })
        elif text_analysis["pronunciation"]["score"] < 0.9:
            suggestions.append({
                "area": "Pronunciation",
                "tip": "Record yourself reading a paragraph and compare to a native speaker.",
                "exercise": "Shadow speech from a podcast or video - speak along with the recording."
            })
    fluency_score = text_analysis.get("fluency", {}).get("score", 0.7)
    if fluency_score < 0.7:
        suggestions.append({
            "area": "Fluency",
            "tip": "Read aloud for 5 minutes daily, focusing on smooth connections.",
            "exercise": "Try 'timed reading' - read a short paragraph in 30 seconds, then try again."
        })
    elif fluency_score < 0.9:
        suggestions.append({
            "area": "Fluency",
            "tip": "Practice chunking words together in natural phrases.",
            "exercise": "Record yourself telling a 1-minute story about your day."
        })
    if voice_characteristics.get("expressiveness_score", 0.8) < 0.6:
        suggestions.append({
            "area": "Expression",
            "tip": "Exaggerate your intonation when practicing - go higher on questions.",
            "exercise": "Read dialogue from a book with different character voices."
        })
    if len(suggestions) < 2:
        suggestions.append({
            "area": "Daily Practice",
            "tip": "Consistency is key! Even 10 minutes daily is better than an hour once a week.",
            "exercise": "Set a daily English alarm - when it rings, speak English for 5 minutes."
        })
    return suggestions[:3]

# Function to generate personal feedback report
def generate_proficiency_report(audio_features, voice_characteristics, speech_activity, text_analysis, learning_goal="Overall improvement"):
    if not text_analysis or "error" in text_analysis:
        grammar_score = pronunciation_score = fluency_score = 0.7
        overall_score = 70
    else:
        grammar_score = text_analysis.get("grammar", {}).get("score", 0.7)
        pronunciation_score = text_analysis.get("pronunciation", {}).get("score", 0.7)
        fluency_score = text_analysis.get("fluency", {}).get("score", 0.7)
        if learning_goal == "Grammar focus":
            overall_score = int((grammar_score * 0.5 + pronunciation_score * 0.25 + fluency_score * 0.25) * 100)
        elif learning_goal == "Pronunciation focus":
            overall_score = int((grammar_score * 0.25 + pronunciation_score * 0.5 + fluency_score * 0.25) * 100)
        elif learning_goal == "Fluency focus":
            overall_score = int((grammar_score * 0.25 + pronunciation_score * 0.25 + fluency_score * 0.5) * 100)
        else:
            overall_score = int((grammar_score + pronunciation_score + fluency_score) / 3 * 100)
    overall_score = min(overall_score, 100)
    progress_suggestions = generate_progress_suggestions(text_analysis, voice_characteristics)
    report = {
        "overall": {
            "score": overall_score,
            "summary": text_analysis["overall"]["summary"] if text_analysis and "overall" in text_analysis else "You're making good progress!",
            "learning_goal": learning_goal,
            "next_steps": text_analysis.get("overall", {}).get("next_steps", ["Keep practicing regularly"])
        },
        "grammar": {
            "good": text_analysis["grammar"]["feedback"] if text_analysis and "grammar" in text_analysis and grammar_score > 0.8 else "You're working hard on your sentence structure!",
            "work": ", ".join([f"'{err['error']}' → '{err['correction']}' ({err['explanation']})" for err in text_analysis["grammar"]["errors"]]) if text_analysis and "grammar" in text_analysis and text_analysis["grammar"]["errors"] else "Focus on subject-verb agreement and tenses",
            "tip": "Try writing down what you want to say before speaking it."
        },
        "pronunciation": {
            "good": "Your pronunciation is clear and understandable!" if text_analysis and "pronunciation" in text_analysis and pronunciation_score > 0.8 else "I can understand what you're saying!",
            "work": "speaking too quickly" if audio_features['speech_rate'] > 5 else "" + (", ".join([f"'{w['word']}' - {w['suggestion']}" for w in text_analysis["pronunciation"]["mispronounced_words"]]) if text_analysis and "pronunciation" in text_analysis and text_analysis["pronunciation"]["mispronounced_words"] else ""),
            "tip": "Slow down and stress important words." if audio_features['speech_rate'] > 5 else "Record yourself and compare with native speakers."
        },
        "fluency": {
            "good": f"You speak with natural flow in {speech_activity.get('speech_segments_count', 0)} segments!",
            "work": "try connecting your thoughts more smoothly" if speech_activity.get('speech_segments_count', 0) > 6 else "",
            "tip": "Practice 'linking'"
        },
        "voice": {
            "description": voice_characteristics.get("summary", "Your voice has good qualities for English speaking."),
            "tips": voice_characteristics.get("tips", ["Practice speaking with more expression"])
        },
        "suggestions": progress_suggestions
    }
    return report

# Main function
def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    if not openai_api_key or not deepgram_api_key:
        print("Error: Missing API keys. Please set OPENAI_API_KEY and DEEPGRAM_API_KEY in your environment variables.")
        return
    tts_engine = init_tts_engine()
    print("\n🌟 Welcome to English Speaking Practice Assistant! 🌟")
    print("\nSelect your English level:")
    for i, level in enumerate(["beginner", "intermediate", "advanced"]):
        print(f"{i+1}. {level.capitalize()}")
    level_choice = input("Enter the number (1-3) or press Enter for intermediate: ")
    learner_level = ["beginner", "intermediate", "advanced"][int(level_choice) - 1] if level_choice.isdigit() and 1 <= int(level_choice) <= 3 else "intermediate"
    print("\nWhat would you like to focus on improving?")
    for i, goal in enumerate(LEARNING_GOALS):
        print(f"{i+1}. {goal}")
    goal_choice = input("Enter the number (1-5) or press Enter for overall improvement: ")
    learning_goal = LEARNING_GOALS[int(goal_choice) - 1] if goal_choice.isdigit() and 1 <= int(goal_choice) <= len(LEARNING_GOALS) else "Overall improvement"
    print(f"\nHere are some practice sentences for {learner_level} level:")
    for i, sentence in enumerate(PRACTICE_SENTENCES[learner_level]):
        print(f"{i+1}. {sentence}")
    choice = input("\nEnter the number to practice one of these sentences, or type your own: ")
    if choice.isdigit() and 1 <= int(choice) <= len(PRACTICE_SENTENCES[learner_level]):
        reference_text = PRACTICE_SENTENCES[learner_level][int(choice) - 1]
    else:
        reference_text = choice
    print("\n📢 Listen to this sentence:")
    speak_text(reference_text, tts_engine)
    print("\nNow it's your turn to practice. I'll record your speech.")
    record_duration = 15
    audio_file_path = record_audio(duration=record_duration)
    print("Analyzing your speech...")
    transcript, confidence, _, word_timings = transcribe_audio(audio_file_path, deepgram_api_key)
    if not transcript:
        print("❌ Error: Could not transcribe your speech. Please try again.")
        speak_text("I couldn't hear that clearly. Let's try again.", tts_engine, emotion="calm")
        return
    print(f"📝 You said: '{transcript}'")
    audio_features = analyze_audio_features(audio_file_path)
    speech_activity = analyze_speech_activity(audio_file_path)
    voice_characteristics = analyze_voice_characteristics(audio_features)
    text_analysis = analyze_speech(reference_text, transcript, confidence, audio_file_path, 
                                  openai_api_key, learning_goal, learner_level, word_timings)
    report = generate_proficiency_report(audio_features, voice_characteristics, 
                                        speech_activity, text_analysis, learning_goal)
    visualization_path = os.path.join("user_recordings", f"visualization_{os.path.basename(audio_file_path).split('.')[0]}.png")
    visualize_audio_features(audio_file_path, save_path=visualization_path, word_timings=word_timings)
    print("\n📊 Your Speech Analysis Results:")
    print(f"Overall Score: {report['overall']['score']}/100")
    print(f"Summary: {report['overall']['summary']}")
    print("\n🔤 Grammar:")
    print(f"Strengths: {report['grammar']['good']}")
    if report['grammar']['work']:
        print(f"Areas to work on: {report['grammar']['work']}")
    print("\n🗣️ Pronunciation:")
    print(f"Strengths: {report['pronunciation']['good']}")
    if report['pronunciation']['work']:
        print(f"Areas to work on: {report['pronunciation']['work']}")
    print("\n🌊 Fluency:")
    print(f"Strengths: {report['fluency']['good']}")
    if report['fluency']['work']:
        print(f"Areas to work on: {report['fluency']['work']}")
    print("\n🎵 Voice characteristics:")
    print(report['voice']['description'])
    print("\n📈 Personalized suggestions for improvement:")
    for i, suggestion in enumerate(report['suggestions']):
        print(f"{i+1}. {suggestion['area']}: {suggestion['tip']}")
        print(f"   Exercise: {suggestion['exercise']}")
    if report['overall']['score'] >= 85:
        speak_text("Excellent job! Your English speaking is very good.", tts_engine, emotion="excited")
    elif report['overall']['score'] >= 70:
        speak_text("Good work! You're making great progress with your English.", tts_engine, emotion="excited")
    else:
        speak_text("Thanks for practicing! Keep working on your English - you're improving!", tts_engine, emotion="calm")
    print("\nWould you like to practice again? (y/n)")
    if input().lower().startswith('y'):
        main()
    else:
        print("Thank you for practicing! Keep up the good work with your English learning!")

if __name__ == "__main__":
    main()