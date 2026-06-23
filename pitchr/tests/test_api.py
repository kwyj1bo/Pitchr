import base64
import json
import tempfile
from unittest.mock import patch

import numpy as np
import soundfile as sf
from frappe.tests.utils import FrappeTestCase


class TestAPI(FrappeTestCase):
	def _melody_audio(self, freqs=(440.0, 494.0, 523.0, 587.0), note_dur=0.5) -> np.ndarray:
		"""A short multi-note melody (recognition needs >= 2 distinct notes)."""
		sr = 22050
		chunks = []
		for f in freqs:
			t = np.linspace(0, note_dur, int(sr * note_dur), endpoint=False)
			env = np.ones_like(t)
			a = int(0.02 * sr)
			env[:a] = np.linspace(0, 1, a)
			env[-a:] = np.linspace(1, 0, a)
			chunks.append((np.sin(2 * np.pi * f * t) * env).astype(np.float32))
			chunks.append(np.zeros(int(sr * 0.05), dtype=np.float32))  # gap between notes
		return np.concatenate(chunks)

	def _create_audio_b64(self, freqs=(440.0, 494.0, 523.0, 587.0)) -> str:
		return base64.b64encode(self._melody_audio(freqs).tobytes()).decode("utf-8")

	def test_recognize_no_songs_returns_unmatched(self):
		audio_b64 = self._create_audio_b64()
		with patch("pitchr.pitchr.api.frappe.get_all", return_value=[]):
			from pitchr.pitchr.api import recognize

			result = recognize(audio_b64)
		self.assertFalse(result["matched"])
		self.assertIsNone(result["song_name"])

	def test_recognize_matching_song(self):
		# Same melody for query and indexed song -> should match.
		audio_b64 = self._create_audio_b64()

		from pitchr.pitch.pipeline import audio_to_contour

		song_contour = audio_to_contour(self._melody_audio())

		mock_songs = [
			{"song_name": "Test Song", "melody_contour": json.dumps(song_contour.tolist())},
			{"song_name": "Other Song", "melody_contour": json.dumps([7.0, -7.0, 5.0, -5.0])},
		]

		with patch("pitchr.pitchr.api.frappe.get_all", return_value=mock_songs):
			from pitchr.pitchr.api import recognize

			result = recognize(audio_b64)

		self.assertTrue(result["matched"])
		self.assertEqual(result["song_name"], "Test Song")

	def test_index_song_success(self):
		with patch("pitchr.pitchr.song_indexer.index_song") as mock_indexer:
			from pitchr.pitchr.api import index_song

			result = index_song("Test Song")
		mock_indexer.assert_called_once_with("Test Song")
		self.assertTrue(result["success"])
