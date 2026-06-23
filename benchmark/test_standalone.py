"""Frappe-free unit + integration tests for the recognition pipeline.

Run with no bench and no database:

    python -m pytest benchmark/test_standalone.py
    python -m unittest benchmark.test_standalone
"""

import unittest

import numpy as np

from benchmark import synth
from benchmark.corpus import builtin_corpus
from pitchr.pitch.match import rank_matches, subsequence_dtw_distance
from pitchr.pitch.melody import notes_to_intervals, pitch_track_to_intervals, segment_notes
from pitchr.pitch.pipeline import audio_to_contour, extract_pitch_track, recognize_contour
from pitchr.pitch.yin import detect_pitch


def _sine(freq, duration=1.0, sr=22050):
	t = np.linspace(0, duration, int(sr * duration), endpoint=False)
	return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestYINBounds(unittest.TestCase):
	def test_detects_within_vocal_range(self):
		for f in (110.0, 220.0, 440.0, 880.0):
			self.assertAlmostEqual(detect_pitch(_sine(f)), f, delta=f * 0.02)

	def test_rejects_subharmonic_noise(self):
		# White noise must not yield a confident very-low pitch.
		rng = np.random.default_rng(0)
		noise = rng.normal(0, 1, 2048).astype(np.float32)
		p = detect_pitch(noise)
		self.assertTrue(p == 0.0 or 80 <= p <= 1000)


class TestMelody(unittest.TestCase):
	def test_segments_constant_pitch_to_one_note(self):
		track = np.full(40, 440.0, dtype=np.float32)
		notes = segment_notes(track)
		self.assertEqual(len(notes), 1)

	def test_intervals_key_invariant(self):
		# Same melody two octaves apart -> identical semitone intervals.
		low = np.array([60, 64, 67, 72], dtype=np.float32)
		high = low + 24
		self.assertTrue(np.allclose(notes_to_intervals(low), notes_to_intervals(high)))

	def test_silence_yields_empty(self):
		track = np.zeros(40, dtype=np.float32)
		self.assertEqual(len(pitch_track_to_intervals(track)), 0)


class TestSubsequenceDTW(unittest.TestCase):
	def test_query_matches_embedded_subsequence(self):
		ref = np.array([5, -3, 2, 1, -2, 4, 0, -1], dtype=np.float32)
		query = ref[2:5]  # a fragment
		self.assertAlmostEqual(subsequence_dtw_distance(query, ref), 0.0, places=5)

	def test_partial_query_beats_unrelated(self):
		ref = np.array([1, 2, -1, 3, -2, 1, 1], dtype=np.float32)
		query = np.array([2, -1, 3], dtype=np.float32)
		good = subsequence_dtw_distance(query, ref)
		bad = subsequence_dtw_distance(query, np.array([-5, -5, -5, -5], dtype=np.float32))
		self.assertLess(good, bad)

	def test_empty_is_inf(self):
		self.assertEqual(subsequence_dtw_distance(np.zeros(0), np.arange(3.0)), float("inf"))


class TestPipelineEndToEnd(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.songs = builtin_corpus()
		cls.db = [(s.name, np.array(s.contour, np.float32)) for s in cls.songs]

	def test_clean_hum_recognized_top1(self):
		# A clean rendition of every song should be recognized as itself.
		correct = 0
		for s in self.songs:
			audio = synth.synthesize(s.notes, synth.CLEAN, seed=1)
			contour = audio_to_contour(audio)
			ranked = rank_matches(contour, self.db)
			if ranked and ranked[0][0] == s.name:
				correct += 1
		self.assertGreaterEqual(correct / len(self.songs), 0.95)

	def test_transposed_fragment_recognized(self):
		# Key-shifted partial hum still maps to the right song (top-3).
		target = next(s for s in self.songs if s.name == "Ode to Joy")
		audio = synth.synthesize(target.notes, synth.LIGHT, seed=3)
		res = recognize_contour(audio_to_contour(audio), self.db, threshold=99, top_k=3)
		names = [c["song_name"] for c in res["candidates"]]
		self.assertIn("Ode to Joy", names)

	def test_extract_pitch_track_shape(self):
		audio = synth.synthesize(self.songs[0].notes, synth.CLEAN, seed=1)
		track = extract_pitch_track(audio)
		self.assertGreater(len(track), 0)
		self.assertTrue(np.all(track >= 0))


if __name__ == "__main__":
	unittest.main(verbosity=2)
