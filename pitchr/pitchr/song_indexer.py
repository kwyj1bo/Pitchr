import json

import frappe
import numpy as np
import soundfile as sf

from pitchr.pitch.contour import normalize
from pitchr.pitch.yin import detect_pitch

FRAME_SIZE = 2048
HOP_SIZE = 512
SAMPLE_RATE = 22050


def load_audio(file_path: str) -> np.ndarray:
	audio, sr = sf.read(file_path, dtype="float32")
	if audio.ndim > 1:
		audio = audio.mean(axis=1)
	if sr != SAMPLE_RATE:
		raise ValueError(f"Expected sample rate {SAMPLE_RATE}, got {sr}")
	return audio


def extract_pitch_sequence(audio: np.ndarray) -> np.ndarray:
	pitches = []
	for start in range(0, len(audio) - FRAME_SIZE, HOP_SIZE):
		frame = audio[start : start + FRAME_SIZE]
		pitch = detect_pitch(frame)
		pitches.append(pitch)
	return np.array(pitches, dtype=np.float32)


def index_song(song_name: str) -> None:
	doc = frappe.get_doc("Song", song_name)
	audio = load_audio(doc.file_path)
	pitch_sequence = extract_pitch_sequence(audio)
	contour = normalize(pitch_sequence)
	doc.melody_contour = json.dumps(contour.tolist())
	doc.save()
