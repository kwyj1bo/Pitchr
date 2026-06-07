import numpy as np


SAMPLE_RATE = 22050
MIN_FREQ = 80
MAX_FREQ = 1000
THRESHOLD = 0.15


def difference(audio: np.ndarray) -> np.ndarray:
	tau_max = len(audio) // 2
	diff = np.zeros(tau_max)
	for tau in range(1, tau_max):
		diff[tau] = np.sum((audio[: tau_max - tau] - audio[tau : tau_max]) ** 2)
	return diff


def cumulative_mean_normalized_difference(diff: np.ndarray) -> np.ndarray:
	cmnd = np.zeros(len(diff))
	cmnd[0] = 1.0
	cumsum = 0.0
	for tau in range(1, len(diff)):
		cumsum += diff[tau]
		cmnd[tau] = diff[tau] * tau / cumsum
	return cmnd


def absolute_threshold(cmnd: np.ndarray, threshold: float = THRESHOLD) -> int:
	for tau in range(2, len(cmnd)):
		if cmnd[tau] < threshold:
			while tau + 1 < len(cmnd) and cmnd[tau + 1] < cmnd[tau]:
				tau += 1
			return tau
	return -1


def parabolic_interpolation(cmnd: np.ndarray, tau: int) -> float:
	if tau <= 0 or tau >= len(cmnd) - 1:
		return float(tau)
	denominator = cmnd[tau - 1] - 2 * cmnd[tau] + cmnd[tau + 1]
	if denominator == 0:
		return float(tau)
	return tau + (cmnd[tau - 1] - cmnd[tau + 1]) / (2 * denominator)


def detect_pitch(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
	diff = difference(audio)
	cmnd = cumulative_mean_normalized_difference(diff)
	tau = absolute_threshold(cmnd)
	if tau == -1:
		return 0.0
	refined_tau = parabolic_interpolation(cmnd, tau)
	if refined_tau == 0:
		return 0.0
	return sample_rate / refined_tau