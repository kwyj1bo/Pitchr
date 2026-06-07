import numpy as np


def remove_silence(pitch_sequence: np.ndarray, threshold: float = 50.0) -> np.ndarray:
	return pitch_sequence[pitch_sequence > threshold]


def to_intervals(pitch_sequence: np.ndarray) -> np.ndarray:
	if len(pitch_sequence) < 2:
		return np.array([])
	with np.errstate(divide="ignore", invalid="ignore"):
		ratios = pitch_sequence[1:] / pitch_sequence[:-1]
		ratios = np.where(ratios > 0, np.log2(ratios), 0.0)
	return ratios.astype(np.float32)


def resample(contour: np.ndarray, target_length: int = 100) -> np.ndarray:
	if len(contour) == 0:
		return np.zeros(target_length, dtype=np.float32)
	indices = np.linspace(0, len(contour) - 1, target_length)
	return np.interp(indices, np.arange(len(contour)), contour).astype(np.float32)


def normalize(pitch_sequence: np.ndarray, target_length: int = 100) -> np.ndarray:
	cleaned = remove_silence(pitch_sequence)
	if len(cleaned) < 2:
		return np.zeros(target_length, dtype=np.float32)
	intervals = to_intervals(cleaned)
	return resample(intervals, target_length)
