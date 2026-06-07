from unittest.mock import patch

import numpy as np
from frappe.tests.utils import FrappeTestCase

from pitchr.pitch.yin import detect_pitch


class TestYINPitchDetection(FrappeTestCase):
	def _generate_sine(self, frequency: float, duration: float = 1.0, sample_rate: int = 22050) -> np.ndarray:
		t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
		return np.sin(2 * np.pi * frequency * t).astype(np.float32)

	def test_detects_440hz(self):
		audio = self._generate_sine(440.0)
		detected = detect_pitch(audio)
		self.assertAlmostEqual(detected, 440.0, delta=440.0 * 0.01)

	def test_detects_220hz(self):
		audio = self._generate_sine(220.0)
		detected = detect_pitch(audio)
		self.assertAlmostEqual(detected, 220.0, delta=220.0 * 0.01)

	def test_silence_returns_zero(self):
		audio = np.zeros(22050, dtype=np.float32)
		detected = detect_pitch(audio)
		self.assertEqual(detected, 0.0)