import base64
import json
import tempfile
from unittest.mock import patch

import numpy as np
import soundfile as sf
from frappe.tests.utils import FrappeTestCase


class TestAPI(FrappeTestCase):
	def _create_audio_b64(self, frequency: float = 440.0, duration: float = 3.0) -> str:
		sample_rate = 22050
		t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
		audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
		return base64.b64encode(audio.tobytes()).decode("utf-8")

	def test_recognize_no_songs_returns_unmatched(self):
		audio_b64 = self._create_audio_b64()
		with patch("pitchr.pitchr.api.frappe.get_all", return_value=[]):
			from pitchr.pitchr.api import recognize

			result = recognize(audio_b64)
		self.assertFalse(result["matched"])
		self.assertIsNone(result["song_name"])

	def test_recognize_matching_song(self):
		audio_b64 = self._create_audio_b64(440.0)

		t = np.linspace(0, 3.0, int(22050 * 3.0), endpoint=False)
		song_audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

		from pitchr.pitch.contour import normalize
		from pitchr.pitchr.song_indexer import extract_pitch_sequence

		song_contour = normalize(extract_pitch_sequence(song_audio))

		mock_songs = [
			{"song_name": "Test Song", "melody_contour": json.dumps(song_contour.tolist())},
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
