import numpy as np
import sounddevice as sd

SAMPLE_RATE = 22050
CHANNELS = 1


def record(duration: int = 5) -> tuple[np.ndarray, int]:
	"""
	Record audio from microphone.

	Args:
	    duration: seconds to record

	Returns:
	    (audio_data, sample_rate)
	"""
	audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
	sd.wait()
	return audio.flatten(), SAMPLE_RATE
