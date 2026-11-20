import sys
from collections import deque
from functools import lru_cache
from itertools import chain
from math import gcd
from pathlib import Path
from threading import Event

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from fastrtc import KokoroTTSOptions, SileroVadOptions, get_silero_model, get_tts_model
from PySide6.QtCore import QSettings, QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)
from scipy.signal import resample_poly

from demos.local_speech_chat import generate_reply, transcribe
from voice_setup import VoiceSetup, approved_prompts, sample_directory

MARKDOWN = (
    QTextDocument.MarkdownFeature.MarkdownDialectGitHub
    | QTextDocument.MarkdownFeature.MarkdownNoHTML
)


def spoken_text(markdown: str) -> str:
    document = QTextDocument()
    document.setMarkdown(markdown, MARKDOWN)
    return document.toPlainText().replace("*", "").strip()


class AssistantWorker(QThread):
    message = Signal(str, str)
    status = Signal(str)

    def __init__(self, model, history, text, recording, voice_options, max_tokens, hotwords, parent):
        super().__init__(parent)
        self.model = model
        self.history = history
        self.text = text
        self.recording = recording
        self.voice_options = voice_options
        self.max_tokens = max_tokens
        self.hotwords = hotwords
        self.stop_audio = Event()

    def run(self):
        try:
            text = self.text
            if self.recording is not None:
                rate, samples = self.recording
                self.status.emit(f"Transcribing with Whisper {self.model}…")
                divisor = gcd(rate, 16000)
                samples = resample_poly(samples, 16000 // divisor, rate // divisor)
                recognizer = load_recognizer(self.model)
                text = transcribe(recognizer, samples, self.hotwords)
                if self.isInterruptionRequested():
                    return
                if not text:
                    self.status.emit("No speech recognized.")
                    return

            self.message.emit("user", text)
            self.status.emit("Thinking…")
            reply = generate_reply(
                [*self.history, {"role": "user", "content": text}], self.max_tokens
            )
            if self.isInterruptionRequested():
                return
            self.message.emit("assistant", reply)
            if self.recording is not None and not self.stop_audio.is_set():
                self.status.emit("Speaking…")
                self.play_reply(spoken_text(reply))
            self.status.emit("Ready")
        except Exception as error:
            self.status.emit(f"Could not finish the turn: {error}")

    def play_reply(self, speech):
        if not speech or self.stop_audio.is_set():
            return
        voice = get_tts_model("kokoro")
        chunks = iter(voice.stream_tts_sync(speech, self.voice_options))
        first = next(chunks, None)
        if first is None or self.stop_audio.is_set():
            return
        rate, samples = first
        block_size = max(1, rate // 25)
        with sd.OutputStream(samplerate=rate, channels=1, dtype=samples.dtype,
                             blocksize=block_size, latency="low") as output:
            for _, samples in chain((first,), chunks):
                waveform = samples.reshape(-1, 1)
                for start in range(0, len(waveform), block_size):
                    if self.stop_audio.is_set():
                        output.abort()
                        return
                    output.write(waveform[start:start + block_size])


@lru_cache(maxsize=1)
def load_recognizer(model):
    return WhisperModel(model, device="cpu", compute_type="int8")


class AssistantWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Voice Assistant")
        self.resize(560, 520)
        self.history = []
        self.microphone = None
        self.worker = None
        self.closing = False
        self.voice_enabled = False
        self.audio_queue = deque(maxlen=16)
        self.pre_roll = deque(maxlen=2)
        self.turn_audio = []
        self.silent_blocks = 0
        self.listen_timer = QTimer(self)
        self.listen_timer.setInterval(50)
        self.listen_timer.timeout.connect(self.check_microphone)

        self.model = QComboBox()
        self.model.addItems(["medium", "small", "base"])
        self.microphone_choice = QComboBox()
        self.microphone_choice.addItem("System default", None)
        try:
            for index, device in enumerate(sd.query_devices()):
                if device["max_input_channels"] > 0:
                    self.microphone_choice.addItem(device["name"], index)
        except sd.PortAudioError:
            pass  # Device availability is checked again when voice mode starts.

        self.pause = QDoubleSpinBox()
        self.pause.setRange(0.5, 5.0)
        self.pause.setSingleStep(0.25)
        self.pause.setValue(2.5)
        self.pause.setSuffix(" s")
        self.manual_send = QCheckBox("Wait for me to send")
        self.sensitivity = QComboBox()
        for label, threshold in (("Low", 0.65), ("Normal", 0.5), ("High", 0.35)):
            self.sensitivity.addItem(label, threshold)
        self.sensitivity.setCurrentIndex(1)
        self.voice = QComboBox()
        self.voice.addItems(["af_heart", "af_bella", "am_adam"])
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.75, 1.5)
        self.speed.setSingleStep(0.1)
        self.speed.setValue(1.0)
        self.speed.setSuffix("×")
        self.reply_length = QComboBox()
        for label, limit in (("Short", 80), ("Normal", 160), ("Long", 320)):
            self.reply_length.addItem(label, limit)
        self.reply_length.setCurrentIndex(1)
        self.preferences = QSettings("LocalVoiceAssistant", "VoiceSettings")
        self.manual_send.setChecked(self.preferences.value("manual_send", False, type=bool))
        self.manual_send.toggled.connect(self.change_send_mode)
        self.vocabulary = QLineEdit(placeholderText="e.g. Kokoro, CTranslate2, Ollama")
        self.vocabulary.setMaxLength(300)
        self.restore_voice_settings()
        self.microphone_choice.currentIndexChanged.connect(self.change_microphone)
        self.vocabulary.setText(self.preferences.value("vocabulary", "", type=str))
        self.vocabulary.textChanged.connect(
            lambda text: self.preferences.setValue("vocabulary", text)
        )

        self.voice_settings = QWidget()
        settings = QFormLayout(self.voice_settings)
        for label, control in (
            ("Microphone", self.microphone_choice), ("Whisper model", self.model),
            ("Pause before reply", self.pause), ("Turn ending", self.manual_send),
            ("Speech sensitivity", self.sensitivity),
            ("Voice", self.voice), ("Speaking speed", self.speed),
            ("Reply length", self.reply_length),
            ("Words to recognize", self.vocabulary),
        ):
            settings.addRow(label, control)
        self.voice_settings.hide()
        settings_button = QPushButton("Voice settings", checkable=True)
        settings_button.toggled.connect(self.voice_settings.setVisible)
        self.setup_button = QPushButton("Set up your voice")
        self.setup_button.clicked.connect(self.open_voice_setup)
        self.voice_profile = QLabel()
        self.voice_profile.setWordWrap(True)
        self.vocabulary.textChanged.connect(self.refresh_voice_profile)
        self.model.currentTextChanged.connect(self.refresh_voice_profile)
        self.refresh_voice_profile()
        self.transcript = QTextEdit(readOnly=True)
        self.text = QLineEdit(placeholderText="Type a question and press Enter")
        self.text.returnPressed.connect(self.send_text)
        self.record = QPushButton("Enable voice mode")
        self.record.clicked.connect(self.toggle_voice_mode)
        self.send_voice = QPushButton("Send voice turn")
        self.send_voice.setEnabled(False)
        self.send_voice.setVisible(self.manual_send.isChecked())
        self.send_voice.clicked.connect(self.submit_voice_turn)
        self.stop_speaking = QPushButton("Stop speaking")
        self.stop_speaking.setEnabled(False)
        self.stop_speaking.clicked.connect(self.stop_reply_audio)
        self.reset = QPushButton("New conversation")
        self.reset.clicked.connect(self.clear_conversation)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        for widget in (self.transcript, self.text, self.record, self.send_voice, self.stop_speaking, self.reset,
                       settings_button, self.voice_settings, self.setup_button, self.voice_profile, self.status):
            layout.addWidget(widget)

    def send_text(self):
        text = self.text.text().strip()
        if text:
            self.text.clear()
            self.start_turn(text=text)

    def restore_voice_settings(self):
        choices = {
            "microphone": self.microphone_choice,
            "whisper_model": self.model,
            "sensitivity": self.sensitivity,
            "voice": self.voice,
            "reply_length": self.reply_length,
        }
        for key, control in choices.items():
            saved = self.preferences.value(key, control.currentText(), type=str)
            if control.findText(saved) >= 0:
                control.setCurrentText(saved)
            control.currentTextChanged.connect(
                lambda value, key=key: self.preferences.setValue(key, value)
            )

        for key, control in (("pause_seconds", self.pause), ("speaking_speed", self.speed)):
            control.setValue(self.preferences.value(key, control.value(), type=float))
            control.valueChanged.connect(
                lambda value, key=key: self.preferences.setValue(key, value)
            )

    def change_send_mode(self, manual):
        self.preferences.setValue("manual_send", manual)
        self.send_voice.setVisible(manual)
        self.send_voice.setEnabled(manual and bool(self.turn_audio))
        self.silent_blocks = 0

    def open_voice_setup(self):
        self.close_microphone()
        model = self.model.currentText()

        def recognize(audio):
            return transcribe(load_recognizer(model), audio)

        VoiceSetup(recognize, self.microphone_choice.currentData(), self).exec()
        self.refresh_voice_profile()
        if self.voice_enabled:
            self.open_microphone()

    def refresh_voice_profile(self):
        vocabulary = "Saved vocabulary loaded" if self.vocabulary.text().strip() else "No saved vocabulary"
        count = len(approved_prompts(sample_directory()))
        self.voice_profile.setText(
            f"Voice profile: {vocabulary} · Approved samples: {count}\n"
            f"Whisper {self.model.currentText()} selected · No personalized model loaded"
        )

    def toggle_voice_mode(self):
        self.voice_enabled = not self.voice_enabled
        if not self.voice_enabled:
            self.close_microphone()
            self.record.setText("Enable voice mode")
            if self.worker is None:
                self.status.setText("Voice mode off")
            return
        self.record.setText("Disable voice mode")
        if self.worker is None:
            self.open_microphone()

    def open_microphone(self):
        try:
            self.speech_detector = get_silero_model()
            device = self.microphone_choice.currentData()
            self.rate = int(sd.query_devices(device, kind="input")["default_samplerate"])
            self.microphone = sd.InputStream(
                device=device,
                samplerate=self.rate, blocksize=self.rate // 4,
                channels=1, dtype="float32",
                callback=lambda audio, *_: self.audio_queue.append(audio[:, 0].copy()),
            )
            self.microphone.start()
        except (sd.PortAudioError, RuntimeError) as error:
            self.toggle_voice_mode()
            self.status.setText(f"Could not start voice mode: {error}")
            return
        self.listen_timer.start()
        self.status.setText("Listening…")

    def change_microphone(self):
        if self.voice_enabled and self.worker is None:
            self.close_microphone()
            self.open_microphone()

    def check_microphone(self):
        options = SileroVadOptions(
            threshold=self.sensitivity.currentData(), min_speech_duration_ms=80
        )
        pause_samples = self.pause.value() * self.rate
        while self.audio_queue:
            samples = self.audio_queue.popleft()
            divisor = gcd(self.rate, 16000)
            waveform = resample_poly(samples, 16000 // divisor, self.rate // divisor)
            _, speech = self.speech_detector.vad((16000, waveform), options)
            if not self.turn_audio:
                if speech:
                    self.turn_audio.extend(self.pre_roll)
                    self.turn_audio.append(samples)
                    if self.manual_send.isChecked():
                        self.send_voice.setEnabled(True)
                        self.status.setText("Listening. Click Send voice turn when finished.")
                else:
                    self.pre_roll.append(samples)
                continue

            if len(self.turn_audio) >= 480 and self.manual_send.isChecked():
                self.status.setText("Recording limit reached. Click Send voice turn.")
                continue
            self.turn_audio.append(samples)
            self.silent_blocks = 0 if speech else self.silent_blocks + 1
            pause_detected = self.silent_blocks * (self.rate // 4) >= pause_samples
            # Bound the recording even when speech continues without a pause.
            recording_limit_reached = len(self.turn_audio) >= 480
            if not self.manual_send.isChecked() and (pause_detected or recording_limit_reached):
                self.submit_voice_turn()
                return

    def submit_voice_turn(self):
        if self.turn_audio and self.worker is None:
            self.start_turn(recording=(self.rate, np.concatenate(self.turn_audio)))

    def close_microphone(self):
        self.listen_timer.stop()
        if self.microphone is not None:
            self.microphone.close()
            self.microphone = None
        self.audio_queue.clear()
        self.pre_roll.clear()
        self.turn_audio.clear()
        self.send_voice.setEnabled(False)
        self.silent_blocks = 0

    def start_turn(self, text="", recording=None):
        # Pause capture during the reply so the assistant cannot hear itself.
        self.close_microphone()
        self.set_busy(True)
        voice_options = KokoroTTSOptions(voice=self.voice.currentText(), speed=self.speed.value())
        self.worker = AssistantWorker(
            self.model.currentText(), self.history.copy(), text, recording,
            voice_options, self.reply_length.currentData(), self.vocabulary.text().strip(), self,
        )
        self.worker.message.connect(self.add_message)
        self.worker.status.connect(self.status.setText)
        self.worker.finished.connect(self.finish_turn)
        self.stop_speaking.setEnabled(recording is not None)
        self.worker.start()

    def stop_reply_audio(self):
        if self.worker is not None:
            self.worker.stop_audio.set()
            self.stop_speaking.setEnabled(False)

    def finish_turn(self):
        self.worker.deleteLater()
        self.worker = None
        self.stop_speaking.setEnabled(False)
        if self.closing:
            self.close()
            return
        self.set_busy(False)
        if self.voice_enabled:
            self.open_microphone()

    def set_busy(self, busy):
        for widget in (self.text, self.reset, self.voice_settings, self.setup_button):
            widget.setEnabled(not busy)

    def add_message(self, role, text):
        self.history.append({"role": role, "content": text})
        label = "You" if role == "user" else "Assistant"
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.transcript.document().isEmpty():
            cursor.insertBlock()
        cursor.insertMarkdown(f"**{label}**", MARKDOWN)
        cursor.insertBlock()
        if role == "assistant":
            cursor.insertMarkdown(text, MARKDOWN)
        else:
            cursor.insertText(text)
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    def clear_conversation(self):
        self.history.clear()
        self.transcript.clear()
        self.status.setText("Listening…" if self.voice_enabled else "Ready")

    def closeEvent(self, event):
        if self.worker is not None:
            self.closing = True
            self.stop_reply_audio()
            self.worker.requestInterruption()
            self.status.setText("Stopping audio and closing…")
            event.ignore()
            return
        self.close_microphone()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Local Voice Assistant")
    icon_path = Path(__file__).with_name("assets") / "voice-assistant.ico"
    app.setWindowIcon(QIcon(str(icon_path)))
    window = AssistantWindow()
    window.show()
    sys.exit(app.exec())
