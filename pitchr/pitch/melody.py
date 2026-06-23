"""Note-level melody representation.

The frame-level pitch track produced by YIN is noisy: a single sustained note
spans many frames, and short octave/halving errors produce spikes.  For
query-by-humming the robust unit of comparison is the *note*, not the frame.

This module turns a frame-level pitch track (Hz per frame, ``0.0`` for silence
or unvoiced frames) into a compact, key-invariant melody representation:

    pitch track (Hz/frame)  ->  notes (semitones)  ->  intervals (delta semitones)

Intervals between consecutive notes are key-invariant by construction (a
transposition adds a constant to every note, which cancels in the difference),
so no global normalization is needed.  That is what makes subsequence matching
of a hummed fragment against a full song possible.
"""

import numpy as np

A4_HZ = 440.0


def hz_to_semitones(pitch_hz: np.ndarray) -> np.ndarray:
	"""Convert frequencies (Hz) to semitones relative to A4 (440 Hz).

	Zero / non-positive entries (silence, unvoiced) are returned as NaN so they
	can be ignored downstream without being confused with a real pitch.
	"""
	pitch_hz = np.asarray(pitch_hz, dtype=np.float64)
	out = np.full(pitch_hz.shape, np.nan)
	voiced = pitch_hz > 0
	out[voiced] = 12.0 * np.log2(pitch_hz[voiced] / A4_HZ)
	return out


def segment_notes(
	pitch_hz: np.ndarray,
	min_note_frames: int = 3,
	merge_tolerance: float = 0.6,
) -> np.ndarray:
	"""Segment a frame-level pitch track into note-level semitone values.

	Consecutive voiced frames whose pitch stays within ``merge_tolerance``
	semitones of the running note median are merged into a single note.  Notes
	shorter than ``min_note_frames`` are discarded as noise.  Silence
	(unvoiced frames) ends the current note.

	Returns a 1-D array of note pitches in semitones (relative to A4).
	"""
	semis = hz_to_semitones(pitch_hz)
	notes: list[float] = []
	current: list[float] = []

	def flush() -> None:
		if len(current) >= min_note_frames:
			notes.append(float(np.median(current)))
		current.clear()

	for s in semis:
		if np.isnan(s):
			flush()
			continue
		if not current:
			current.append(s)
			continue
		if abs(s - np.median(current)) <= merge_tolerance:
			current.append(s)
		else:
			flush()
			current.append(s)
	flush()

	return np.array(notes, dtype=np.float32)


def notes_to_intervals(notes: np.ndarray) -> np.ndarray:
	"""Key-invariant interval sequence: successive semitone differences.

	Large jumps (> 1 octave) are clamped, since they almost always come from
	YIN octave errors rather than real melodic leaps.
	"""
	if len(notes) < 2:
		return np.zeros(0, dtype=np.float32)
	intervals = np.diff(notes.astype(np.float64))
	intervals = np.clip(intervals, -12.0, 12.0)
	return intervals.astype(np.float32)


def pitch_track_to_intervals(
	pitch_hz: np.ndarray,
	min_note_frames: int = 3,
	merge_tolerance: float = 0.6,
) -> np.ndarray:
	"""Full convenience pipeline: frame pitch track -> key-invariant intervals."""
	notes = segment_notes(pitch_hz, min_note_frames, merge_tolerance)
	return notes_to_intervals(notes)
