from unittest.mock import patch

import frappe
import numpy as np
from frappe.tests.utils import FrappeTestCase


class TestAudioCapture(FrappeTestCase):
	def test_record_shape(self):
		duration = 2
		fake_audio = np.zeros((duration * 22050, 1), dtype="float32")

		with patch("sounddevice.rec", return_value=fake_audio):
			with patch("sounddevice.wait"):
				from pitchr.audio.capture import record

				audio, sr = record(duration)

		self.assertIsInstance(audio, np.ndarray)
		self.assertEqual(sr, 22050)
		self.assertEqual(len(audio), duration * 22050)

	def test_record_range(self):
		duration = 2
		fake_audio = np.random.uniform(-1.0, 1.0, (duration * 22050, 1)).astype("float32")

		with patch("sounddevice.rec", return_value=fake_audio):
			with patch("sounddevice.wait"):
				from pitchr.audio.capture import record

				audio, _ = record(duration)

		self.assertGreaterEqual(audio.min(), -1.0)
		self.assertLessEqual(audio.max(), 1.0)
