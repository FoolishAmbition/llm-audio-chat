"""Desktop window and coordination of user actions."""

from sounddevice import PortAudioError
from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from fastrtc import KokoroTTSOptions

from .capture import MicrophoneCapture
from .settings import VoiceSettings
from .speech import load_recognizer, transcribe
from .transcript import TranscriptView
from .turn import AssistantWorker, TurnSettings
from .voice_setup import VoiceSetup, approved_prompts, sample_directory


class AssistantWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Voice Assistant")
        self.resize(560, 520)
        self.history = []
        self.worker = None
        self.closing = False
        self.voice_enabled = False
        self.capture = MicrophoneCapture(self)
        self.capture.turn_ready.connect(self.start_voice_turn)
        self.capture.speech_started.connect(self.on_speech_started)
        self.capture.recording_limit.connect(self.on_recording_limit)

        self.voice_settings = VoiceSettings()
        self.voice_settings.hide()
        self.voice_settings.manual_send.toggled.connect(self.change_send_mode)
        self.voice_settings.microphone_choice.currentIndexChanged.connect(self.change_microphone)
        self.voice_settings.pause.valueChanged.connect(self.configure_capture)
        self.voice_settings.sensitivity.currentIndexChanged.connect(self.configure_capture)
        self.configure_capture()
        settings_button = QPushButton("Voice settings", checkable=True)
        settings_button.toggled.connect(self.voice_settings.setVisible)
        self.setup_button = QPushButton("Set up your voice")
        self.setup_button.clicked.connect(self.open_voice_setup)
        self.voice_profile = QLabel()
        self.voice_profile.setWordWrap(True)
        self.voice_settings.vocabulary.textChanged.connect(self.refresh_voice_profile)
        self.voice_settings.model.currentTextChanged.connect(self.refresh_voice_profile)
        self.refresh_voice_profile()
        self.transcript = TranscriptView()
        self.web_search = QCheckBox("Search the web for next question")
        self.web_search.setToolTip("Sends the text of the next question to public search engines.")
        self.auto_lookup = QCheckBox("Automatic web lookup")
        self.auto_lookup.setToolTip("Lets the assistant decide when to look up current facts. Search may send your question text online.")
        self.auto_lookup.setChecked(self.voice_settings.automatic_lookup_enabled())
        self.auto_lookup.toggled.connect(self.voice_settings.save_automatic_lookup)
        self.text = QLineEdit(placeholderText="Type a question and press Enter")
        self.text.returnPressed.connect(self.send_text)
        self.record = QPushButton("Enable voice mode")
        self.record.clicked.connect(self.toggle_voice_mode)
        self.send_voice = QPushButton("Send voice turn")
        self.send_voice.setEnabled(False)
        self.send_voice.setVisible(self.voice_settings.manual_send.isChecked())
        self.send_voice.clicked.connect(self.capture.submit)
        self.stop_speaking = QPushButton("Stop speaking")
        self.stop_speaking.setEnabled(False)
        self.stop_speaking.clicked.connect(self.stop_reply_audio)
        self.reset = QPushButton("New conversation")
        self.reset.clicked.connect(self.clear_conversation)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        for widget in (
            self.transcript, self.text, self.web_search, self.auto_lookup,
            self.record, self.send_voice, self.stop_speaking, self.reset,
            settings_button, self.voice_settings, self.setup_button, self.voice_profile, self.status,
        ):
            layout.addWidget(widget)

    def send_text(self):
        text = self.text.text().strip()
        if text:
            self.text.clear()
            self.start_turn(text=text)

    def change_send_mode(self, manual):
        self.voice_settings.save_send_mode(manual)
        self.send_voice.setVisible(manual)
        self.send_voice.setEnabled(manual and bool(self.capture.turn_audio))
        self.configure_capture()

    def open_voice_setup(self):
        self.close_microphone()
        model = self.voice_settings.model.currentText()

        def recognize(audio):
            return transcribe(load_recognizer(model), audio)

        VoiceSetup(recognize, self.voice_settings.microphone_choice.currentData(), self).exec()
        self.refresh_voice_profile()
        if self.voice_enabled:
            self.open_microphone()

    def refresh_voice_profile(self):
        vocabulary = "Saved vocabulary loaded" if self.voice_settings.vocabulary.text().strip() else "No saved vocabulary"
        count = len(approved_prompts(sample_directory()))
        self.voice_profile.setText(
            f"Voice profile: {vocabulary} · Approved samples: {count}\n"
            f"Whisper {self.voice_settings.model.currentText()} selected · No personalized model loaded"
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

    def configure_capture(self):
        self.capture.configure(
            self.voice_settings.pause.value(),
            self.voice_settings.sensitivity.currentData(),
            self.voice_settings.manual_send.isChecked(),
        )

    def open_microphone(self):
        try:
            self.capture.start(self.voice_settings.microphone_choice.currentData())
        except (PortAudioError, RuntimeError) as error:
            self.toggle_voice_mode()
            self.status.setText(f"Could not start voice mode: {error}")
            return
        self.status.setText("Listening?")

    def change_microphone(self):
        if self.voice_enabled and self.worker is None:
            self.close_microphone()
            self.open_microphone()

    def on_speech_started(self):
        if self.voice_settings.manual_send.isChecked():
            self.send_voice.setEnabled(True)
            self.status.setText("Listening. Click Send voice turn when finished.")

    def on_recording_limit(self):
        self.status.setText("Recording limit reached. Click Send voice turn.")

    def start_voice_turn(self, recording):
        if self.worker is None:
            self.start_turn(recording=recording)

    def close_microphone(self):
        self.capture.stop()
        self.send_voice.setEnabled(False)

    def start_turn(self, text="", recording=None):
        # Pause capture during the reply so the assistant cannot hear itself.
        self.close_microphone()
        self.set_busy(True)
        voice_options = KokoroTTSOptions(voice=self.voice_settings.voice.currentText(), speed=self.voice_settings.speed.value())
        settings = TurnSettings(
            model=self.voice_settings.model.currentText(),
            voice=voice_options,
            max_tokens=self.voice_settings.reply_length.currentData(),
            hotwords=self.voice_settings.vocabulary.text().strip(),
            use_search=self.web_search.isChecked(),
            auto_lookup=self.auto_lookup.isChecked(),
        )
        self.worker = AssistantWorker(settings, self.history.copy(), text, recording, self)
        self.web_search.setChecked(False)
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
            previous_status = self.status.text()
            self.open_microphone()
            if self.capture.stream is not None and previous_status.startswith("Could not finish"):
                self.status.setText(f"{previous_status} Listening…")

    def set_busy(self, busy):
        for widget in (self.text, self.reset, self.voice_settings, self.setup_button, self.web_search, self.auto_lookup):
            widget.setEnabled(not busy)

    def add_message(self, role, text):
        self.history.append({"role": role, "content": text})
        self.transcript.append_message(role, text)

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
