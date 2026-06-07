import numpy as np
from frappe.tests.utils import FrappeTestCase

from pitchr.pitch.dtw import dtw_distance, find_best_match


class TestDTW(FrappeTestCase):
	def test_identical_sequences_have_zero_distance(self):
		a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
		self.assertAlmostEqual(dtw_distance(a, a), 0.0)

	def test_similar_sequences_have_low_distance(self):
		a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
		b = np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float32)
		self.assertLess(dtw_distance(a, b), 1.0)

	def test_different_sequences_have_high_distance(self):
		a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
		b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
		self.assertGreater(dtw_distance(a, b), 10.0)

	def test_distance_is_symmetric(self):
		a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
		b = np.array([1.5, 2.5, 3.5], dtype=np.float32)
		self.assertAlmostEqual(dtw_distance(a, b), dtw_distance(b, a), places=5)

	def test_find_best_match_returns_closest(self):
		query = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
		candidates = [
			("song_a", np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float32)),
			("song_b", np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)),
			("song_c", np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)),
		]
		name, score = find_best_match(query, candidates)
		self.assertEqual(name, "song_a")
		self.assertLess(score, 1.0)

	def test_find_best_match_empty_candidates(self):
		query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
		name, score = find_best_match(query, [])
		self.assertEqual(name, "")
		self.assertEqual(score, np.inf)
