"""Build the searchable song database.

Two sources, same output format — a list of ``Song`` records each carrying the
note sequence and its key-invariant reference contour:

  * ``builtin_corpus()``      : the bundled public-domain melodies (no download)
  * ``load_midi_corpus(dir)`` : every ``.mid`` file in a directory, melody track
                                extracted to a note sequence

``load_midi_corpus`` is how the database scales to "all songs": point it at any
collection of MIDI files (e.g. the Nottingham folk set or a Lakh MIDI subset)
and the same pipeline indexes and searches them.
"""

import glob
import os
from dataclasses import dataclass

from benchmark import melodies
from benchmark.synth import notes_to_reference_contour


@dataclass
class Song:
	name: str
	notes: list[tuple[int, float]]
	contour: list[float]  # reference interval contour stored in the "DB"


def builtin_corpus() -> list[Song]:
	songs = []
	for name, notes in melodies.MELODIES.items():
		contour = notes_to_reference_contour(notes).tolist()
		songs.append(Song(name=name.strip(), notes=notes, contour=contour))
	return songs


def _extract_melody_from_midi(path: str) -> list[tuple[int, float]]:
	"""Extract a monophonic melody (highest sounding note over time) from a MIDI file.

	Uses the track with the most notes as the melody, then flattens any overlap
	by keeping the highest pitch — a simple, robust heuristic for QBH corpora.
	"""
	import mido

	mid = mido.MidiFile(path)
	ticks_per_beat = mid.ticks_per_beat or 480

	best_track, best_count = None, -1
	for track in mid.tracks:
		count = sum(1 for m in track if m.type == "note_on" and m.velocity > 0)
		if count > best_count:
			best_track, best_count = track, count
	if best_track is None or best_count < 2:
		return []

	# Collect (start_beat, end_beat, pitch) note events.
	events, active, t = [], {}, 0
	for msg in best_track:
		t += msg.time
		if msg.type == "note_on" and msg.velocity > 0:
			active[msg.note] = t
		elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
			if msg.note in active:
				start = active.pop(msg.note)
				events.append((start, t, msg.note))
	if not events:
		return []
	events.sort()

	# Flatten to a monophonic sequence in time order (highest note wins ties).
	notes: list[tuple[int, float]] = []
	for start, end, pitch in events:
		beats = max((end - start) / ticks_per_beat, 0.05)
		if notes and start == events[0][0]:
			pass
		notes.append((pitch, round(beats, 3)))
	return notes


def load_midi_corpus(directory: str, limit: int | None = None) -> list[Song]:
	songs = []
	paths = sorted(glob.glob(os.path.join(directory, "**", "*.mid"), recursive=True))
	paths += sorted(glob.glob(os.path.join(directory, "**", "*.midi"), recursive=True))
	for path in paths:
		try:
			notes = _extract_melody_from_midi(path)
		except Exception:
			continue
		if len(notes) < 6:
			continue
		contour = notes_to_reference_contour(notes).tolist()
		if len(contour) < 4:
			continue
		name = os.path.splitext(os.path.basename(path))[0]
		songs.append(Song(name=name, notes=notes, contour=contour))
		if limit and len(songs) >= limit:
			break
	return songs
