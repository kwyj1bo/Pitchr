import json
import tempfile
from unittest.mock import patch

import numpy as np
import soundfile as sf
from frappe.tests.utils import FrappeTestCase

from pitchr.pitchr.song_indexer import extract_pitch_sequence, load_audio


class TestSongIndexer(FrappeTestCase):
	def _create_temp_audio(
		self, freqs=(440.0, 494.0, 523.0, 587.0), note_dur: float = 0.5, sample_rate: int = 22050
	) -> str:
		chunks = []
		for f in freqs:
			t = np.linspace(0, note_dur, int(sample_rate * note_dur), endpoint=False)
			env = np.ones_like(t)
			a = int(0.02 * sample_rate)
			env[:a] = np.linspace(0, 1, a)
			env[-a:] = np.linspace(1, 0, a)
			chunks.append((np.sin(2 * np.pi * f * t) * env).astype(np.float32))
			chunks.append(np.zeros(int(sample_rate * 0.05), dtype=np.float32))
		audio = np.concatenate(chunks)
		tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
		sf.write(tmp.name, audio, sample_rate)
		return tmp.name

	def test_load_audio_returns_mono(self):
		path = self._create_temp_audio()
		audio = load_audio(path)
		self.assertEqual(audio.ndim, 1)
		self.assertIsInstance(audio, np.ndarray)

	def test_load_audio_wrong_sample_rate(self):
		t = np.linspace(0, 1.0, 44100, endpoint=False)
		audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
		tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
		sf.write(tmp.name, audio, 44100)
		with self.assertRaises(ValueError):
			load_audio(tmp.name)

	def test_extract_pitch_sequence_returns_array(self):
		path = self._create_temp_audio()
		audio = load_audio(path)
		pitches = extract_pitch_sequence(audio)
		self.assertIsInstance(pitches, np.ndarray)
		self.assertGreater(len(pitches), 0)

	def test_index_song_saves_contour(self):
		path = self._create_temp_audio()
		mock_doc = type("Doc", (), {"file_path": path, "melody_contour": None, "save": lambda self: None})()

		with patch("pitchr.pitchr.song_indexer.frappe.get_doc", return_value=mock_doc):
			with patch("pitchr.pitchr.song_indexer.frappe.db.commit"):
				from pitchr.pitchr.song_indexer import index_song

				index_song("Test Song")

		self.assertIsNotNone(mock_doc.melody_contour)
		contour = json.loads(mock_doc.melody_contour)
		# Note-level interval contour: a 4-note melody yields 3 intervals.
		self.assertGreaterEqual(len(contour), 2)
