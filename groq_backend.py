from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import whisper
import os
import uuid
import torch
from TTS.api import TTS
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from pydub import AudioSegment

import noisereduce as nr
import librosa
import soundfile as sf

from starlette.background import BackgroundTask
from settings import Settings

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Groq LLaMA
# Use key from env or fallback sample key
client = Groq(api_key=Settings.GROQ_API_KEY)

# Whisper model 
# Switch to a smaller model for faster inference (with trade-off in accuracy)
whisper_model = whisper.load_model("tiny")  # or 'base', 'small', 'medium', etc.

# Coqui TTS setup
add_safe_globals([XttsConfig, XttsAudioConfig])
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Request schemas
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

# Existing Groq chat route
@app.post("/chat")
def chat(data: ChatRequest):
    last_message = data.messages[-1].content.lower().strip()

    # Check for greeting
    if last_message in ["hi", "hello", "hey", "hey there", "good morning", "good evening"]:
        return {
            "reply": "Hello! I'm **IntervieWin**, your AI mock interviewer. I'm here to help you prepare for your interviews with realistic questions and feedback. Ready to begin?"
        }
    messages = [{"role": "system", "content": (
                "You are IntervieWin — a professional AI mock interviewer. "
                "You're interviewing a candidate for a AI Engineer position. "
                "Ask very short and specific technical questions or give short answers (maximum 7–10 words)."
                "Focused on AI topics such as machine learning, deep learning, NLP, model evaluation, or deployment."
                "Avoid generic chatbot behavior or off-topic replies."
            )}] + \
               [m.dict() for m in data.messages[-3:]]
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        # messages=[m.dict() for m in data.messages]
        messages = messages # last 3 turns only # minimize tokens, avoid sending unnecessary chat history
    )
    return {"reply": response.choices[0].message.content}

# Whisper transcription route
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        result = whisper_model.transcribe(temp_path)
        return {"text": result["text"]}
    finally:
        os.remove(temp_path)


@app.post("/generate-voice")
async def generate_voice(text: str = Form(...)):
    temp_raw = "audios/saved_speaker.wav"
    intermediate_wav = f"intermediate_{uuid.uuid4().hex}.wav"
    clean_wav_path = f"clean_speaker_{uuid.uuid4().hex}.wav"
    # output_path = f"voice_{uuid.uuid4().hex}.wav"
    output_path = f"audios/voice.wav"


    try:
        # Step 1: Convert saved voice to WAV
        sound = AudioSegment.from_file(temp_raw)
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(intermediate_wav, format="wav")

        # Step 2: Noise reduction
        y, sr = librosa.load(intermediate_wav, sr=None)
        reduced_audio = nr.reduce_noise(y=y, sr=sr)
        sf.write(clean_wav_path, reduced_audio, sr)

        # Step 3: Generate voice
        tts.tts_to_file(
            text=text,
            speaker_wav=clean_wav_path,
            language="en",
            file_path=output_path
        )

        if not os.path.exists(output_path):
            raise RuntimeError("TTS generation failed. Output file not found.")

        # Cleanup task
        cleanup = BackgroundTask(
            lambda: [os.remove(f) for f in [intermediate_wav, clean_wav_path, output_path] if os.path.exists(f)]
        )

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="voice.wav",
            background=cleanup
        )

    except Exception as e:
        raise RuntimeError(f"Voice generation error: {e}")