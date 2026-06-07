import numpy as np
from frappe.tests.utils import FrappeTestCase

from pitchr.pitch.contour import normalize, remove_silence, resample, to_intervals


class TestContourNormalization(FrappeTestCase):
	def test_remove_silence_filters_low_values(self):
		pitch = np.array([0.0, 30.0, 440.0, 494.0, 0.0, 523.0], dtype=np.float32)
		result = remove_silence(pitch)
		self.assertTrue(np.all(result > 50.0))
		self.assertEqual(len(result), 3)

	def test_to_intervals_same_melody_different_key(self):
		low = np.array([220.0, 247.0, 261.0], dtype=np.float32)
		high = np.array([440.0, 494.0, 523.0], dtype=np.float32)
		self.assertTrue(np.allclose(to_intervals(low), to_intervals(high), atol=0.01))

	def test_resample_output_length(self):
		contour = np.random.rand(50).astype(np.float32)
		result = resample(contour, target_length=100)
		self.assertEqual(len(result), 100)

	def test_normalize_returns_fixed_length(self):
		pitch = np.array([440.0, 494.0, 523.0, 494.0, 440.0] * 10, dtype=np.float32)
		result = normalize(pitch)
		self.assertEqual(len(result), 100)

	def test_normalize_silence_returns_zeros(self):
		pitch = np.zeros(100, dtype=np.float32)
		result = normalize(pitch)
		self.assertTrue(np.all(result == 0.0))
