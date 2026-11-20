"""Collect microphone audio and detect the end of a voice turn."""

from collections import deque

import numpy as np
import sounddevice as sd
from fastrtc import SileroVadOptions, get_silero_model
from PySide6.QtCore import QObject, QTimer, Signal

from .speech import for_whisper


class MicrophoneCapture(QObject):
    turn_ready = Signal(object)
    speech_started = Signal()
    recording_limit = Signal()

    BLOCKS_PER_SECOND = 4
    MAX_BLOCKS = 120 * BLOCKS_PER_SECOND

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stream = None
        self.audio_queue = deque(maxlen=16)
        self.pre_roll = deque(maxlen=2)
        self.turn_audio = []
        self.silent_blocks = 0
        self.configure(2.5, 0.5, False)
        self.timer = QTimer(self, interval=50)
        self.timer.timeout.connect(self.process_audio)

    def configure(self, pause_seconds, threshold, manual_send):
        if manual_send != getattr(self, "manual_send", False):
            self.silent_blocks = 0
        self.pause_seconds = pause_seconds
        self.manual_send = manual_send
        self.vad_options = SileroVadOptions(threshold=threshold, min_speech_duration_ms=80)

    def start(self, device):
        self.detector = get_silero_model()
        self.rate = int(sd.query_devices(device, kind="input")["default_samplerate"])
        self.block_size = self.rate // self.BLOCKS_PER_SECOND
        self.stream = sd.InputStream(
            device=device, samplerate=self.rate, blocksize=self.block_size,
            channels=1, dtype="float32", callback=self.collect_audio,
        )
        self.stream.start()
        self.timer.start()

    def collect_audio(self, samples, *_):
        self.audio_queue.append(samples[:, 0].copy())

    def process_audio(self):
        while self.audio_queue:
            samples = self.audio_queue.popleft()
            waveform = for_whisper(samples, self.rate)
            _, speech = self.detector.vad((16000, waveform), self.vad_options)
            if not self.turn_audio:
                if speech:
                    self.turn_audio.extend(self.pre_roll)
                    self.turn_audio.append(samples)
                    self.speech_started.emit()
                else:
                    self.pre_roll.append(samples)
                continue

            if len(self.turn_audio) >= self.MAX_BLOCKS and self.manual_send:
                self.recording_limit.emit()
                continue
            self.turn_audio.append(samples)
            self.silent_blocks = 0 if speech else self.silent_blocks + 1
            pause_detected = self.silent_blocks * self.block_size >= self.pause_seconds * self.rate
            if not self.manual_send and (pause_detected or len(self.turn_audio) >= self.MAX_BLOCKS):
                self.submit()
                return

    def submit(self):
        if self.turn_audio:
            self.turn_ready.emit((self.rate, np.concatenate(self.turn_audio)))

    def stop(self):
        self.timer.stop()
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self.audio_queue.clear()
        self.pre_roll.clear()
        self.turn_audio.clear()
        self.silent_blocks = 0
