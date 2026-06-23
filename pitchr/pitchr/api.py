import base64
import json

import frappe
import numpy as np
from frappe import _

from pitchr.pitch.pipeline import DEFAULT_MATCH_THRESHOLD, audio_to_contour, recognize_contour

SAMPLE_RATE = 22050


def _load_song_contours() -> list[tuple[str, np.ndarray]]:
	"""Load indexed song fingerprints from the database."""
	songs = frappe.get_all("Song", fields=["song_name", "melody_contour"])
	candidates = []
	for song in songs:
		if not song.get("melody_contour"):
			continue
		contour = np.array(json.loads(song["melody_contour"]), dtype=np.float32)
		if len(contour) >= 2:
			candidates.append((song["song_name"], contour))
	return candidates


@frappe.whitelist(allow_guest=True)  # nosemgrep: frappe-semgrep-rules.rules.security.guest-whitelisted-method
def recognize(audio_b64: str) -> dict:
	"""Recognize a hummed melody.

	``audio_b64`` is base64-encoded raw mono float32 PCM at 22.05 kHz; the
	browser decodes and resamples its recording before upload (see www/pitchr.js)
	so the backend never has to depend on an audio codec.
	"""
	audio_bytes = base64.b64decode(audio_b64)
	audio = np.frombuffer(audio_bytes, dtype=np.float32)

	query_contour = audio_to_contour(audio, SAMPLE_RATE)
	candidates = _load_song_contours()

	return recognize_contour(query_contour, candidates, threshold=DEFAULT_MATCH_THRESHOLD)


@frappe.whitelist()
def index_song(song_name: str) -> dict:
	from pitchr.pitchr.song_indexer import index_song as run_indexer

	try:
		run_indexer(song_name)
		return {"success": True, "message": _("Song indexed successfully")}
	except Exception as e:
		frappe.throw(str(e))
