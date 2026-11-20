"""Run one conversation turn off the UI thread."""

from dataclasses import dataclass
from datetime import datetime
from threading import Event

from fastrtc import KokoroTTSOptions
from PySide6.QtCore import QThread, Signal

from .ollama import choose_lookup, generate_reply
from .speech import for_whisper, load_recognizer, play_reply, transcribe
from .transcript import spoken_text
from .web_search import FORECAST_SOURCE, search_web, singapore_tomorrow, source_links


@dataclass(frozen=True)
class TurnSettings:
    model: str
    voice: KokoroTTSOptions
    max_tokens: int
    hotwords: str
    use_search: bool
    auto_lookup: bool


def built_in_reply(question: str) -> str | None:
    question = " ".join(question.lower().rstrip("?!. ").split())
    if question in {"what time is it", "what time is it right now", "what time is ti right now",
                    "what is the current time", "what's the time", "what's the time right now"}:
        now = datetime.now().astimezone()
        return f"Your computer's local time is {now.strftime('%I:%M %p').lstrip('0')}."
    if question in {"can you access the internet", "can you acces the internet", "do you have internet access"}:
        return "Enable **Automatic web lookup** to let me decide when to look things up, or enable **Search the web for next question** for a single turn."
    return None


class AssistantWorker(QThread):
    message = Signal(str, str)
    status = Signal(str)

    def __init__(self, settings: TurnSettings, history, text, recording, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.history = history
        self.text = text
        self.recording = recording
        self.stop_audio = Event()

    def run(self):
        try:
            text = self.text
            if self.recording is not None:
                rate, samples = self.recording
                self.status.emit(f"Transcribing with Whisper {self.settings.model}…")
                samples = for_whisper(samples, rate)
                recognizer = load_recognizer(self.settings.model)
                text = transcribe(recognizer, samples, self.settings.hotwords)
                if self.isInterruptionRequested():
                    return
                if not text:
                    self.status.emit("No speech recognized.")
                    return

            self.message.emit("user", text)
            sources = []
            reply = built_in_reply(text)
            try:
                route = "search" if self.settings.use_search else "local"
                if reply is None and self.settings.auto_lookup:
                    self.status.emit("Deciding whether to look this up…")
                    decision = choose_lookup(text)
                    if decision != "local":
                        route = decision
                if self.isInterruptionRequested():
                    return
                if reply is None and route == "singapore_tomorrow":
                    self.status.emit("Checking the Singapore weather forecast…")
                    reply = singapore_tomorrow()
                    sources = [{"url": FORECAST_SOURCE}]
                elif reply is None and route == "search":
                    self.status.emit("Searching the web…")
                    sources = search_web(text)
            except RuntimeError as error:
                reply = str(error)
            if self.isInterruptionRequested():
                return
            if reply is None:
                self.status.emit("Thinking…")
                reply = generate_reply(
                    [*self.history, {"role": "user", "content": text}], self.settings.max_tokens, sources
                )
            if self.isInterruptionRequested():
                return
            self.message.emit("assistant", reply + source_links(sources) if sources else reply)
            if self.recording is not None and not self.stop_audio.is_set():
                self.status.emit("Speaking…")
                play_reply(spoken_text(reply), self.settings.voice, self.stop_audio)
            self.status.emit("Ready")
        except Exception as error:
            self.status.emit(f"Could not finish the turn: {error}")
