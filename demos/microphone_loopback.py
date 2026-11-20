from fastrtc import Stream, ReplyOnPause
import numpy as np

def echo(audio: tuple[int, np.ndarray]):
    sample_rate, samples = audio
    duration = samples.shape[-1] / sample_rate
    print(f"Pause detected. Echoing {duration:.1f} seconds of audio.", flush=True)
    yield audio

stream = Stream(
    handler=ReplyOnPause(echo),
    modality="audio", 
    mode="send-receive",
)

stream.ui.launch()
