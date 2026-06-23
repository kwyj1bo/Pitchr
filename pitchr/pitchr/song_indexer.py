import json

import frappe
import numpy as np
import soundfile as sf

from pitchr.pitch.pipeline import FRAME_SIZE, HOP_SIZE, SAMPLE_RATE, audio_to_contour, extract_pitch_track


def load_audio(file_path: str) -> np.ndarray:
	audio, sr = sf.read(file_path, dtype="float32")
	if audio.ndim > 1:
		audio = audio.mean(axis=1)
	if sr != SAMPLE_RATE:
		raise ValueError(f"Expected sample rate {SAMPLE_RATE}, got {sr}")
	return audio


def extract_pitch_sequence(audio: np.ndarray) -> np.ndarray:
	"""Per-frame pitch track (kept for backward compatibility)."""
	return extract_pitch_track(audio, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE)


def index_song(song_name: str) -> None:
	doc = frappe.get_doc("Song", song_name)
	audio = load_audio(doc.file_path)
	# Index with the exact pipeline used at query time so the stored fingerprint
	# is representationally consistent with hummed queries.
	contour = audio_to_contour(audio, SAMPLE_RATE)
	doc.melody_contour = json.dumps(contour.tolist())
	doc.save()
