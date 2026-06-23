"""Accuracy benchmark for the Pitchr query-by-humming pipeline.

For every song in the corpus and every distortion profile, synthesize N hummed
queries (different random seeds), run the full recognition pipeline against the
whole database, and report:

  * Top-1 accuracy   : fraction where the correct song ranks #1
  * Top-3 accuracy   : fraction where the correct song is in the top 3
  * MRR              : mean reciprocal rank of the correct song
  * Accept@thresh    : fraction accepted (score <= threshold) AND correct

Run:
    python -m benchmark.run_benchmark
    python -m benchmark.run_benchmark --midi-dir /path/to/midis --queries 5
"""

import argparse
import time

import numpy as np

from benchmark import synth
from benchmark.corpus import builtin_corpus, load_midi_corpus
from pitchr.pitch.pipeline import DEFAULT_MATCH_THRESHOLD, audio_to_contour
from pitchr.pitch.match import rank_matches


def _build_db(songs):
	return [(s.name, np.array(s.contour, dtype=np.float32)) for s in songs if len(s.contour) >= 2]


def evaluate(songs, profiles, queries_per_song=3, threshold=DEFAULT_MATCH_THRESHOLD, verbose=False):
	db = _build_db(songs)
	names = {n for n, _ in db}
	results = {}

	for profile in profiles:
		top1 = top3 = rr_sum = accept_correct = total = 0
		t0 = time.time()
		for song in songs:
			if song.name not in names or len(song.notes) < 4:
				continue
			for q in range(queries_per_song):
				audio = synth.synthesize(song.notes, profile, seed=1000 * q + 7)
				contour = audio_to_contour(audio)
				ranked = rank_matches(contour, db)
				if not ranked:
					total += 1
					continue
				rank = next((i for i, (n, _) in enumerate(ranked) if n == song.name), None)
				total += 1
				if rank is not None:
					rr_sum += 1.0 / (rank + 1)
					if rank == 0:
						top1 += 1
						if ranked[0][1] <= threshold:
							accept_correct += 1
					if rank < 3:
						top3 += 1
				elif verbose:
					print(f"    MISS [{profile.name}] {song.name}")
		dt = time.time() - t0
		results[profile.name] = {
			"top1": top1 / total if total else 0.0,
			"top3": top3 / total if total else 0.0,
			"mrr": rr_sum / total if total else 0.0,
			"accept_correct": accept_correct / total if total else 0.0,
			"n": total,
			"secs": dt,
		}
	return results


def print_report(results, n_songs):
	print()
	print(f"  Pitchr recognition benchmark  ({n_songs} songs in database)")
	print("  " + "-" * 68)
	print(f"  {'profile':<12}{'top-1':>9}{'top-3':>9}{'MRR':>9}{'accept✓':>10}{'queries':>9}")
	print("  " + "-" * 68)
	for name, r in results.items():
		print(
			f"  {name:<12}{r['top1']:>8.1%}{r['top3']:>9.1%}{r['mrr']:>9.3f}"
			f"{r['accept_correct']:>10.1%}{r['n']:>9}"
		)
	print("  " + "-" * 68)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--midi-dir", help="directory of .mid files to use as the corpus")
	ap.add_argument("--midi-limit", type=int, default=None)
	ap.add_argument("--queries", type=int, default=3, help="queries synthesized per song")
	ap.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
	ap.add_argument("--verbose", action="store_true")
	args = ap.parse_args()

	if args.midi_dir:
		songs = load_midi_corpus(args.midi_dir, limit=args.midi_limit)
		print(f"  Loaded {len(songs)} songs from MIDI dir {args.midi_dir}")
	else:
		songs = builtin_corpus()

	results = evaluate(
		songs, synth.PROFILES, queries_per_song=args.queries,
		threshold=args.threshold, verbose=args.verbose,
	)
	print_report(results, len(songs))


if __name__ == "__main__":
	main()
