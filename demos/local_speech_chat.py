import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from fastrtc import AdditionalOutputs, ReplyOnPause, Stream, get_tts_model

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant_app.ollama import generate_reply
from assistant_app.speech import load_recognizer, transcribe


def build_demo(whisper_model: str = "medium") -> Stream:
    speech_recognizer = load_recognizer(whisper_model)
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

        try:
            reply = generate_reply(messages)
        except RuntimeError as error:
            raise gr.Error(str(error)) from error
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
