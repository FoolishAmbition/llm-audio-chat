from fastrtc import ReplyOnPause, Stream, AdditionalOutputs, get_tts_model
import numpy as np
from numpy.typing import NDArray

from dotenv import load_dotenv
from groq import Groq

import os
import time
import tempfile
import wave
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tts_client = get_tts_model(model="kokoro")


def audio_to_wav_file(audio_data, sample_rate):
    """Convert audio data to a temporary WAV file for Groq processing"""
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return temp_file.name


def transcribe_with_groq(audio):
    """Transcribe audio using Groq's Whisper API"""
    sample_rate, audio_data = audio

    wav_file = audio_to_wav_file(audio_data, sample_rate)

    try:
        with open(wav_file, "rb") as file:
            translation = groq_client.audio.translations.create(
                file=(wav_file, file.read()),
                model="whisper-large-v3",
                response_format="json",
                temperature=0.0
            )
            return translation.text
    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)


def generate_response(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chatbot: list[dict] | None = None
):
    chatbot = chatbot or []
    messages = [{"role": msg["role"], "content": msg["content"]}
                for msg in chatbot]
    start = time.time()
    text = transcribe_with_groq(audio)
    print("transcription", time.time() - start)
    print("prompt", text)

    chatbot.append({"role": "user", "content": text})

    yield AdditionalOutputs(chatbot)

    messages.append({"role": "user", "content": text})
    response_text = (
        groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=512,
            messages=messages,
        )
        .choices[0]
        .message.content
    )

    chatbot.append({"role": "assistant", "content": response_text})

    yield response_text


def response(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chatbot: list[dict] | None = None
):
    gen = generate_response(audio, chatbot)

    next(gen)  # Consume the transcript update before reading the reply text.

    response_text = next(gen)

    print(response_text)

    yield from tts_client.stream_tts_sync(response_text)


stream = Stream(
    handler=ReplyOnPause(response, input_sample_rate=16000),
    modality="audio",
    mode="send-receive",
    ui_args={
        "title": "LLM Voice Chat (Powered by Groq, Kokoro, and WebRTC ⚡️)"},
)

stream.ui.launch()
