import json
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QStandardPaths, QThread, Signal
from PySide6.QtWidgets import QDialog, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout

from .speech import for_whisper

PROMPTS = (
    "What should I work on first today?",
    "Remind me to check my calendar on Thursday at three fifteen.",
    "Explain the difference between a class and an object in Python.",
    "Open my notes about Ollama, Kokoro, and CTranslate2.",
    "I said fifteen, not fifty. Can you repeat the number?",
    "The first approach seems simpler, but I want to understand why it works. Walk me through it step by step, then give me a short example I can try myself.",
    "Help me plan a small software project. I want to break it into milestones and test each part before I move on.",
    "Today I spent some time debugging a program. I found the problem after I checked what the function was actually receiving. Next time, I want to write a small test before I change the code.",
    "Is that option B for process, option D for quality, or both?",
    "Question three is a multi-select. A, product. B, process. C, profit. D, quality. E, packaging.",
    "Please compare fifteen percent, fifty percent, and zero point five percent.",
    "My appointment moved from Tuesday the twelfth to Thursday the twenty-first.",
    "The meeting starts at nine oh five and should last about forty-five minutes.",
    "I meant cache, as in temporary storage, rather than cash, as in money.",
    "Tell me why this variable is null and whether I should check for an empty string first.",
    "Explain the difference between an API request, a response, and a status code.",
    "Could you compare Git fetch, Git pull, and Git merge using one example?",
    "Before you answer, let me finish reading the choices. The first is planning, the second is analysis, and the third is implementation.",
    "I have two questions. First, how do I reproduce the bug? Second, how can I check that the same problem does not happen again?",
    "I'm revising for a software engineering quiz. I want to understand how requirements, design, implementation, and testing fit together. Explain one stage at a time and give me a simple example for each.",
)


def sample_directory() -> Path:
    return Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppLocalDataLocation
    )) / "voice-samples"


def approved_prompts(directory: Path) -> list[str]:
    prompts = []
    for metadata_path in directory.glob("*.json"):
        if not metadata_path.with_suffix(".wav").is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["transcript"].strip() and isinstance(metadata["prompt"], str):
                prompts.append(metadata["prompt"])
        except (OSError, ValueError, AttributeError, KeyError, TypeError):
            continue
    return prompts


def save_sample(directory: Path, audio: np.ndarray, rate: int, transcript: str, prompt: str):
    directory.mkdir(parents=True, exist_ok=True)
    sample_id = uuid4().hex
    audio_path = directory / f"{sample_id}.wav"
    metadata_path = directory / f"{sample_id}.json"
    samples = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    try:
        with wave.open(str(audio_path), "wb") as output:
            output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            output.writeframes(samples.tobytes())
        metadata_path.write_text(json.dumps({
            "audio": audio_path.name, "transcript": transcript, "prompt": prompt,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        audio_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise


class ReviewWorker(QThread):
    reviewed = Signal(str)
    failed = Signal(str)

    def __init__(self, audio, rate, recognize, parent):
        super().__init__(parent)
        self.audio = audio
        self.rate = rate
        self.recognize = recognize

    def run(self):
        try:
            audio = for_whisper(self.audio, self.rate)
            self.reviewed.emit(self.recognize(audio))
        except Exception as error:
            self.failed.emit(str(error))


class VoiceSetup(QDialog):
    def __init__(self, recognize, device, parent):
        super().__init__(parent)
        self.setWindowTitle("Set up your voice")
        self.resize(520, 360)
        self.recognize = recognize
        self.device = device
        self.index = 0
        self.microphone = None
        self.worker = None
        self.audio = None
        self.directory = sample_directory()
        saved_prompts = set(approved_prompts(self.directory))
        self.prompts = [prompt for prompt in PROMPTS if prompt not in saved_prompts]

        self.prompt = QLabel()
        self.prompt.setWordWrap(True)
        self.transcript = QPlainTextEdit(placeholderText="Review and correct what Whisper heard")
        self.transcript.textChanged.connect(self.update_approval)
        self.record = QPushButton("Record")
        self.record.clicked.connect(self.toggle_recording)
        self.approve = QPushButton("Approve and save sample")
        self.approve.clicked.connect(self.approve_sample)
        self.playback = QPushButton("Play recording")
        self.playback.clicked.connect(self.play_sample)
        self.skip = QPushButton("Skip phrase")
        self.skip.clicked.connect(self.next_prompt)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        for widget in (self.prompt, self.record, self.transcript, self.playback, self.approve, self.skip, self.status):
            layout.addWidget(widget)
        self.show_prompt()

    def show_prompt(self):
        sd.stop()
        self.audio = None
        self.transcript.clear()
        complete = self.index >= len(self.prompts)
        self.record.setEnabled(not complete)
        self.skip.setEnabled(not complete)
        self.update_approval()
        if complete:
            self.prompt.setText("Voice setup complete")
            self.status.setText(f"Approved samples are saved locally in {self.directory}. Training is a separate step.")
        else:
            self.prompt.setText(f"Phrase {self.index + 1} of {len(self.prompts)} remaining prompts\n\n{self.prompts[self.index]}")
            self.status.setText("Read the phrase naturally. Only approved samples are saved.")

    def toggle_recording(self):
        if self.microphone is None:
            sd.stop()
            self.audio = None
            self.transcript.clear()
            self.update_approval()
            self.frames = []
            try:
                self.rate = int(sd.query_devices(self.device, kind="input")["default_samplerate"])
                self.microphone = sd.InputStream(
                    device=self.device, samplerate=self.rate, channels=1, dtype="float32",
                    callback=lambda block, *_: self.frames.append(block[:, 0].copy()),
                )
                self.microphone.start()
            except sd.PortAudioError as error:
                self.close_microphone()
                self.status.setText(f"Could not start the microphone: {error}")
                return
            self.record.setText("Stop and review")
            self.skip.setEnabled(False)
            self.status.setText("Recording…")
            return

        self.close_microphone()
        self.record.setText("Record again")
        if not self.frames:
            self.skip.setEnabled(True)
            self.status.setText("No audio captured. Try again.")
            return
        self.audio = np.concatenate(self.frames)
        self.frames.clear()
        self.record.setEnabled(False)
        self.status.setText("Checking what Whisper heard…")
        self.worker = ReviewWorker(self.audio, self.rate, self.recognize, self)
        self.worker.reviewed.connect(self.review_sample)
        self.worker.failed.connect(self.review_failed)
        self.worker.finished.connect(self.finish_review)
        self.worker.start()

    def review_sample(self, transcript):
        self.transcript.setPlainText(transcript)
        self.status.setText("Edit the transcript to match what you actually said, then approve it.")

    def review_failed(self, error):
        self.status.setText(f"Whisper could not review the audio. You can enter the exact transcript manually. {error}")

    def finish_review(self):
        self.worker.deleteLater()
        self.worker = None
        self.record.setEnabled(True)
        self.skip.setEnabled(True)
        self.update_approval()

    def update_approval(self):
        ready = self.audio is not None and self.worker is None
        self.playback.setEnabled(ready)
        self.approve.setEnabled(ready and bool(self.transcript.toPlainText().strip()))

    def play_sample(self):
        try:
            sd.play(self.audio, self.rate)
        except sd.PortAudioError as error:
            self.status.setText(f"Could not play the recording: {error}")

    def approve_sample(self):
        transcript = self.transcript.toPlainText().strip()
        if self.audio is None or not transcript or self.worker is not None:
            return
        try:
            save_sample(self.directory, self.audio, self.rate, transcript, self.prompts[self.index])
        except OSError as error:
            self.status.setText(f"Could not save the sample: {error}")
            return
        self.next_prompt()

    def next_prompt(self):
        self.index += 1
        self.record.setText("Record")
        self.show_prompt()

    def close_microphone(self):
        if self.microphone is not None:
            self.microphone.close()
            self.microphone = None

    def done(self, result):
        if self.worker is not None:
            self.status.setText("Wait for Whisper to finish reviewing this sample.")
            return
        self.close_microphone()
        sd.stop()
        super().done(result)
