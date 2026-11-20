import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import gradio as gr
import numpy as np
from fastrtc import AdditionalOutputs, ReplyOnPause, Stream, get_tts_model
from faster_whisper import WhisperModel

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:7b"


def generate_reply(messages: list[dict[str, str]], max_tokens: int = 160) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful personal assistant. Answer clearly and briefly so your replies are easy to listen to."},
            *messages,
        ],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)["message"]["content"]
    except HTTPError as error:
        if error.code == 404:
            raise gr.Error(f"Download the local model with: ollama pull {MODEL}") from error
        raise gr.Error(f"Ollama returned an error (HTTP {error.code}).") from error
    except (URLError, TimeoutError) as error:
        raise gr.Error("Make sure Ollama is running, then try again.") from error


def transcribe(speech_recognizer: WhisperModel, samples: np.ndarray, hotwords: str = "") -> str:
    if samples.size == 0:
        return ""

    # FastRTC supplies 16 kHz audio. Whisper expects normalized mono samples.
    waveform = samples.reshape(-1).astype(np.float32)
    if samples.dtype == np.int16:
        waveform /= 32768.0
    segments, _ = speech_recognizer.transcribe(
        waveform, language="en", beam_size=1, condition_on_previous_text=False,
        hotwords=hotwords or None,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def build_demo(whisper_model: str = "medium") -> Stream:
    speech_recognizer = WhisperModel(whisper_model, device="cpu", compute_type="int8")
    voice = get_tts_model("kokoro")

    def respond(audio: tuple[int, np.ndarray], *ui_inputs):
        # FastRTC may pass the connection ID before the conversation input.
        history = ui_inputs[-1] if ui_inputs else None
        _, samples = audio
        transcript = transcribe(speech_recognizer, samples)
        if not transcript:
            return

        messages = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in (history or [])
        ]
        messages.append({"role": "user", "content": transcript})
        yield AdditionalOutputs(messages)

        reply = generate_reply(messages)
        yield AdditionalOutputs([*messages, {"role": "assistant", "content": reply}])
        yield from voice.stream_tts_sync(reply)

    conversation = gr.Chatbot(type="messages", label="Conversation")
    return Stream(
        handler=ReplyOnPause(respond, input_sample_rate=16000),
        modality="audio",
        mode="send-receive",
        additional_inputs=[conversation],
        additional_outputs=[conversation],
        additional_outputs_handler=lambda _, messages: messages,
        ui_args={"title": "Local Voice Assistant"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the local voice assistant.")
    parser.add_argument("--whisper-model", default="medium", help="Faster Whisper model (default: medium).")
    args = parser.parse_args()
    build_demo(args.whisper_model).ui.launch()
