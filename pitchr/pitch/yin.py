import numpy as np

SAMPLE_RATE = 22050
MIN_FREQ = 80
MAX_FREQ = 1000
THRESHOLD = 0.15


def difference(audio: np.ndarray) -> np.ndarray:
	r"""YIN difference function.

	d(\tau) = \sum_{j=0}^{M-\tau-1} (x_j - x_{j+\tau})^2,  M = len(audio)//2

	Expanded as A(\tau) + B(\tau) - 2 C(\tau), where the energy terms A/B come
	from prefix sums and the correlation term C is the autocorrelation computed
	with the FFT.  This is mathematically identical to the naive O(M^2) double
	loop but runs in O(M log M), which makes indexing and recognition fast
	enough for long recordings and large corpora.
	"""
	m = len(audio) // 2
	diff = np.zeros(m)
	if m < 2:
		return diff

	x = np.asarray(audio[:m], dtype=np.float64)
	sq = x * x
	prefix = np.cumsum(sq)          # prefix[i] = sum_{0..i} x^2
	total = prefix[-1]

	# Autocorrelation via FFT: corr[tau] = sum_j x[j] x[j+tau].
	n_fft = 1 << (2 * m - 1).bit_length()
	spectrum = np.fft.rfft(x, n_fft)
	corr = np.fft.irfft(spectrum * np.conj(spectrum), n_fft)[:m]

	tau = np.arange(1, m)
	a = prefix[m - tau - 1]          # sum of first (m - tau) squared samples
	b = total - prefix[tau - 1]      # sum of squared samples from tau .. m-1
	diff[1:] = a + b - 2.0 * corr[1:]
	return np.maximum(diff, 0.0)     # guard tiny negatives from FFT round-off


def cumulative_mean_normalized_difference(diff: np.ndarray) -> np.ndarray:
	cmnd = np.zeros(len(diff))
	cmnd[0] = 1.0
	cumsum = 0.0
	for tau in range(1, len(diff)):
		cumsum += diff[tau]
		cmnd[tau] = diff[tau] * tau / cumsum
	return cmnd


def absolute_threshold(
	cmnd: np.ndarray,
	threshold: float = THRESHOLD,
	tau_min: int = 2,
	tau_max: int | None = None,
) -> int:
	if tau_max is None or tau_max > len(cmnd):
		tau_max = len(cmnd)
	tau_min = max(tau_min, 2)
	for tau in range(tau_min, tau_max):
		if cmnd[tau] < threshold:
			while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
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
	# Restrict the lag search to the plausible vocal range so background noise
	# and sub-harmonics cannot produce spurious very-low/high pitch estimates.
	tau_min = max(int(sample_rate / MAX_FREQ), 2)
	tau_max = min(int(sample_rate / MIN_FREQ) + 1, len(cmnd))
	tau = absolute_threshold(cmnd, tau_min=tau_min, tau_max=tau_max)
	if tau == -1:
		return 0.0
	refined_tau = parabolic_interpolation(cmnd, tau)
	if refined_tau == 0:
		return 0.0
	return sample_rate / refined_tau
