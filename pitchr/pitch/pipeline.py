"""Framework-independent query-by-humming pipeline.

This module contains the full recognition logic with **no Frappe dependency** so
it can be unit-tested, benchmarked, and reused from a CLI or the Frappe API
alike.  The Frappe layer is only responsible for storage (loading song contours
from the database) and HTTP.

    audio (float32 PCM, mono)  ->  pitch track (YIN per frame)
                               ->  key-invariant interval contour
                               ->  subsequence-DTW match against song contours
"""

import numpy as np

from pitchr.pitch.match import rank_matches
from pitchr.pitch.melody import pitch_track_to_intervals
from pitchr.pitch.yin import detect_pitch

SAMPLE_RATE = 22050
FRAME_SIZE = 2048
HOP_SIZE = 512

# Accept decision. The absolute subsequence-DTW cost of the best candidate is a
# weak signal on its own (many wrong songs score similarly), so acceptance also
# requires a confidence MARGIN: the best candidate must beat the runner-up by at
# least DEFAULT_MATCH_MARGIN. A true match stands out from the field; a hum with
# no real match scores close to the whole pack. Both thresholds were calibrated
# against the benchmark (see benchmark/run_benchmark.py --negatives).
DEFAULT_MATCH_THRESHOLD = 1.0
DEFAULT_MATCH_MARGIN = 0.3


def decide_match(
	ranked: list[tuple[str, float]],
	threshold: float = DEFAULT_MATCH_THRESHOLD,
	margin: float = DEFAULT_MATCH_MARGIN,
) -> bool:
	"""Accept the top candidate only if it is both close AND clearly ahead."""
	if not ranked:
		return False
	best_score = ranked[0][1]
	if best_score > threshold:
		return False
	if len(ranked) >= 2 and (ranked[1][1] - best_score) < margin:
		return False
	return True


def extract_pitch_track(
	audio: np.ndarray,
	sample_rate: int = SAMPLE_RATE,
	frame_size: int = FRAME_SIZE,
	hop_size: int = HOP_SIZE,
) -> np.ndarray:
	"""Run YIN over overlapping frames -> one pitch (Hz) per frame, 0.0 = unvoiced."""
	audio = np.asarray(audio, dtype=np.float32)
	if len(audio) < frame_size:
		return np.zeros(0, dtype=np.float32)
	pitches = []
	for start in range(0, len(audio) - frame_size, hop_size):
		frame = audio[start : start + frame_size]
		pitches.append(detect_pitch(frame, sample_rate))
	return np.array(pitches, dtype=np.float32)


def audio_to_contour(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
	"""Audio -> key-invariant interval contour (the song/query fingerprint)."""
	track = extract_pitch_track(audio, sample_rate)
	return pitch_track_to_intervals(track)


def recognize_contour(
	query_contour: np.ndarray,
	song_contours: list[tuple[str, np.ndarray]],
	threshold: float = DEFAULT_MATCH_THRESHOLD,
	top_k: int = 5,
	margin: float = DEFAULT_MATCH_MARGIN,
) -> dict:
	"""Match a query contour against a list of ``(name, contour)`` song fingerprints.

	Returns a dict with the best match, accept/reject decision, and the top-k
	ranked candidates (useful for UIs and for debugging accuracy).
	"""
	ranked = rank_matches(query_contour, song_contours)
	if not ranked:
		return {"matched": False, "song_name": None, "score": None, "candidates": []}

	best_name, best_score = ranked[0]
	matched = decide_match(ranked, threshold, margin)
	return {
		"matched": matched,
		"song_name": best_name if matched else None,
		"score": round(best_score, 4),
		"candidates": [{"song_name": n, "score": round(s, 4)} for n, s in ranked[:top_k]],
	}


def recognize_audio(
	audio: np.ndarray,
	song_contours: list[tuple[str, np.ndarray]],
	sample_rate: int = SAMPLE_RATE,
	threshold: float = DEFAULT_MATCH_THRESHOLD,
	top_k: int = 5,
	margin: float = DEFAULT_MATCH_MARGIN,
) -> dict:
	"""End-to-end: raw audio -> recognition result."""
	query_contour = audio_to_contour(audio, sample_rate)
	return recognize_contour(query_contour, song_contours, threshold, top_k, margin)
