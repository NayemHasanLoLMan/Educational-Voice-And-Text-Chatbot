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