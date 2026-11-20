"""Voice controls and local settings persistence."""

import sounddevice as sd
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLineEdit, QWidget


class VoiceSettings(QWidget):
    def __init__(self):
        super().__init__()
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
        self.vocabulary = QLineEdit(placeholderText="e.g. Kokoro, CTranslate2, Ollama")
        self.vocabulary.setMaxLength(300)
        self.restore_voice_settings()
        self.vocabulary.setText(self.preferences.value("vocabulary", "", type=str))
        self.vocabulary.textChanged.connect(
            lambda text: self.preferences.setValue("vocabulary", text)
        )

        settings = QFormLayout(self)
        for label, control in (
            ("Microphone", self.microphone_choice), ("Whisper model", self.model),
            ("Pause before reply", self.pause), ("Turn ending", self.manual_send),
            ("Speech sensitivity", self.sensitivity),
            ("Voice", self.voice), ("Speaking speed", self.speed),
            ("Reply length", self.reply_length),
            ("Words to recognize", self.vocabulary),
        ):
            settings.addRow(label, control)


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

    def automatic_lookup_enabled(self):
        return self.preferences.value("auto_lookup", False, type=bool)

    def save_automatic_lookup(self, enabled):
        self.preferences.setValue("auto_lookup", enabled)

    def save_send_mode(self, manual):
        self.preferences.setValue("manual_send", manual)
