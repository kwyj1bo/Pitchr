import base64
import json

import frappe
import numpy as np
from frappe import _

from pitchr.pitch.contour import normalize
from pitchr.pitch.dtw import find_best_match
from pitchr.pitch.yin import detect_pitch
from pitchr.pitchr.song_indexer import extract_pitch_sequence

MATCH_THRESHOLD = 50.0
SAMPLE_RATE = 22050


@frappe.whitelist(allow_guest=True)  # nosemgrep: frappe-semgrep-rules.rules.security.guest-whitelisted-method
def recognize(audio_b64: str) -> dict:
	audio_bytes = base64.b64decode(audio_b64)
	audio = np.frombuffer(audio_bytes, dtype=np.float32)

	pitch_sequence = extract_pitch_sequence(audio)
	query_contour = normalize(pitch_sequence)

	songs = frappe.get_all("Song", fields=["song_name", "melody_contour"])

	candidates = []
	for song in songs:
		if not song.get("melody_contour"):
			continue
		contour = np.array(json.loads(song["melody_contour"]), dtype=np.float32)
		candidates.append((song["song_name"], contour))

	if not candidates:
		return {"matched": False, "song_name": None, "score": None}

	best_name, best_score = find_best_match(query_contour, candidates)

	matched = best_score < MATCH_THRESHOLD

	return {
		"matched": matched,
		"song_name": best_name if matched else None,
		"score": round(best_score, 2),
	}


@frappe.whitelist()
def index_song(song_name: str) -> dict:
	from pitchr.pitchr.song_indexer import index_song as run_indexer

	try:
		run_indexer(song_name)
		return {"success": True, "message": _("Song indexed successfully")}
	except Exception as e:
		frappe.throw(str(e))
