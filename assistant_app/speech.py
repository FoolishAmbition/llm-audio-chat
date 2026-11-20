"""Speech recognition, resampling, and cancellable playback."""

from functools import lru_cache
from itertools import chain
from math import gcd

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from fastrtc import get_tts_model
from scipy.signal import resample_poly


def for_whisper(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000:
        return samples
    divisor = gcd(rate, 16000)
    return resample_poly(samples, 16000 // divisor, rate // divisor)


@lru_cache(maxsize=1)
def load_recognizer(model):
    return WhisperModel(model, device="cpu", compute_type="int8")


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


def play_reply(speech, voice_options, stop_audio):
    if not speech or stop_audio.is_set():
        return
    voice = get_tts_model("kokoro")
    chunks = iter(voice.stream_tts_sync(speech, voice_options))
    first = next(chunks, None)
    if first is None or stop_audio.is_set():
        return
    rate, samples = first
    block_size = max(1, rate // 25)
    with sd.OutputStream(samplerate=rate, channels=1, dtype=samples.dtype,
                         blocksize=block_size, latency="low") as output:
        for _, samples in chain((first,), chunks):
            waveform = samples.reshape(-1, 1)
            for start in range(0, len(waveform), block_size):
                if stop_audio.is_set():
                    output.abort()
                    return
                output.write(waveform[start:start + block_size])
