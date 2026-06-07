import numpy as np


def dtw_distance(sequence_a: np.ndarray, sequence_b: np.ndarray) -> float:
	n = len(sequence_a)
	m = len(sequence_b)

	cost = np.full((n + 1, m + 1), np.inf)
	cost[0][0] = 0.0

	for i in range(1, n + 1):
		for j in range(1, m + 1):
			distance = abs(float(sequence_a[i - 1]) - float(sequence_b[j - 1]))
			cost[i][j] = distance + min(
				cost[i - 1][j],
				cost[i][j - 1],
				cost[i - 1][j - 1],
			)

	return float(cost[n][m])


def find_best_match(query: np.ndarray, candidates: list[tuple[str, np.ndarray]]) -> tuple[str, float]:
	if not candidates:
		return "", np.inf

	best_name = ""
	best_score = np.inf

	for name, contour in candidates:
		score = dtw_distance(query, contour)
		if score < best_score:
			best_score = score
			best_name = name

	return best_name, best_score
