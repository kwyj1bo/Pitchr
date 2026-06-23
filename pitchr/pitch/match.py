"""Subsequence DTW matching for query-by-humming.

A user typically hums a *fragment* of a song (often the chorus), not the whole
thing.  Global DTW between a short query and a full-length song melody is
therefore wrong: it forces the short query to stretch across the entire song.

Subsequence DTW instead finds the contiguous span of the song melody that best
aligns to the (entire) query, and reports that span's alignment cost.  The cost
is normalized by query length so scores are comparable across queries of
different lengths and across songs of different lengths.

Both the query and the song are represented as key-invariant interval sequences
(see ``melody.notes_to_intervals``).
"""

import numpy as np


def subsequence_dtw_distance(query: np.ndarray, reference: np.ndarray) -> float:
	"""Cost of aligning the whole ``query`` to the best subspan of ``reference``.

	Returns the accumulated DTW cost normalized by the query length.  Lower is
	better.  ``inf`` if either sequence is empty.
	"""
	n = len(query)
	m = len(reference)
	if n == 0 or m == 0:
		return float("inf")

	query = query.astype(np.float64)
	reference = reference.astype(np.float64)

	# Accumulated cost. Row 0 (first query element) may align to ANY reference
	# position for free -> the match can start anywhere in the reference.
	prev = np.abs(reference - query[0])

	for i in range(1, n):
		local = np.abs(reference - query[i])
		curr = np.empty(m)
		# j == 0: only the diagonal/vertical predecessors exist.
		curr[0] = local[0] + prev[0]
		for j in range(1, m):
			curr[j] = local[j] + min(prev[j - 1], prev[j], curr[j - 1])
		prev = curr

	# Best subspan ends at argmin of the last query row.
	return float(prev.min() / n)


def find_best_match(
	query: np.ndarray,
	candidates: list[tuple[str, np.ndarray]],
) -> tuple[str, float]:
	"""Return ``(name, score)`` of the lowest-cost candidate. Lower score = better."""
	best_name = ""
	best_score = float("inf")
	for name, contour in candidates:
		score = subsequence_dtw_distance(query, np.asarray(contour))
		if score < best_score:
			best_score = score
			best_name = name
	return best_name, best_score


def rank_matches(
	query: np.ndarray,
	candidates: list[tuple[str, np.ndarray]],
) -> list[tuple[str, float]]:
	"""All candidates sorted by ascending score (best first)."""
	scored = [(name, subsequence_dtw_distance(query, np.asarray(contour))) for name, contour in candidates]
	scored.sort(key=lambda x: x[1])
	return scored
