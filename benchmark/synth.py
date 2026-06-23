"""Synthesize realistic hummed-query audio from note sequences.

Real humming is an imperfect rendition of a melody.  To produce an honest
accuracy benchmark (rather than a circular one where queries equal the
database), the synthesizer injects the distortions a real hummer introduces:

  * key transposition          (people hum in whatever key is comfortable)
  * tempo change               (faster/slower than the original)
  * a partial fragment         (people hum a snippet, not the whole song)
  * per-note pitch error        (intonation drift)
  * continuous pitch drift/vibrato
  * timing jitter on note onsets
  * occasional wrong / dropped notes
  * additive background noise

A ``DistortionProfile`` bundles these so the benchmark can sweep clean -> heavy.
The rendered signal is a voice-like tone (fundamental + a few harmonics with an
amplitude envelope), at the pipeline's 22.05 kHz sample rate.
"""

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 22050


@dataclass
class DistortionProfile:
	name: str
	semitone_shift: float = 0.0          # key transposition (semitones)
	tempo_factor: float = 1.0            # >1 = faster (shorter notes)
	per_note_jitter: float = 0.0         # std dev of per-note pitch error (semitones)
	drift: float = 0.0                   # slow pitch-drift amplitude (semitones)
	vibrato: float = 0.0                 # vibrato depth (semitones)
	timing_jitter: float = 0.0           # std dev of onset timing error (fraction of note)
	wrong_note_prob: float = 0.0         # P(a note is sung at the wrong pitch)
	drop_note_prob: float = 0.0          # P(a note is skipped)
	noise_level: float = 0.0             # additive white-noise RMS
	fragment: tuple[float, float] | None = None  # (start_frac, end_frac) of melody to keep


CLEAN = DistortionProfile("clean")

LIGHT = DistortionProfile(
	"light", semitone_shift=2.0, tempo_factor=1.1, per_note_jitter=0.2,
	drift=0.1, vibrato=0.1, timing_jitter=0.05, noise_level=0.005,
	fragment=(0.0, 0.6),
)

REALISTIC = DistortionProfile(
	"realistic", semitone_shift=-3.0, tempo_factor=0.9, per_note_jitter=0.4,
	drift=0.3, vibrato=0.2, timing_jitter=0.12, wrong_note_prob=0.05,
	drop_note_prob=0.05, noise_level=0.02, fragment=(0.1, 0.6),
)

HEAVY = DistortionProfile(
	"heavy", semitone_shift=5.0, tempo_factor=1.25, per_note_jitter=0.7,
	drift=0.5, vibrato=0.3, timing_jitter=0.2, wrong_note_prob=0.12,
	drop_note_prob=0.12, noise_level=0.04, fragment=(0.15, 0.55),
)

PROFILES = [CLEAN, LIGHT, REALISTIC, HEAVY]


def _midi_to_hz(midi: float) -> float:
	return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _select_fragment(
	notes: list[tuple[int, float]], frag: tuple[float, float] | None
) -> list[tuple[int, float]]:
	if frag is None:
		return notes
	total = sum(d for _, d in notes)
	start_t, end_t = frag[0] * total, frag[1] * total
	out, t = [], 0.0
	for midi, dur in notes:
		if t >= start_t and t < end_t:
			out.append((midi, dur))
		t += dur
	return out or notes


def synthesize(
	notes: list[tuple[int, float]],
	profile: DistortionProfile = CLEAN,
	bpm: float = 120.0,
	seed: int | None = None,
	sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
	"""Render ``notes`` (list of ``(midi, beats)``) into a hummed-audio waveform."""
	rng = np.random.default_rng(seed)
	notes = _select_fragment(notes, profile.fragment)
	sec_per_beat = (60.0 / bpm) / profile.tempo_factor

	chunks = []
	drift_phase = rng.uniform(0, 2 * np.pi)
	for midi, beats in notes:
		dur = beats * sec_per_beat
		# timing jitter shortens/lengthens the note slightly
		dur *= 1.0 + rng.normal(0, profile.timing_jitter) if profile.timing_jitter else 1.0
		dur = max(dur, 0.04)
		n_samples = int(dur * sample_rate)
		if n_samples <= 0:
			continue

		if midi == 0:  # rest
			chunks.append(np.zeros(n_samples, dtype=np.float32))
			continue
		if profile.drop_note_prob and rng.random() < profile.drop_note_prob:
			chunks.append(np.zeros(n_samples, dtype=np.float32))  # dropped -> silence
			continue

		pitch = midi + profile.semitone_shift
		if profile.wrong_note_prob and rng.random() < profile.wrong_note_prob:
			pitch += rng.choice([-2, -1, 1, 2])
		if profile.per_note_jitter:
			pitch += rng.normal(0, profile.per_note_jitter)

		t = np.arange(n_samples) / sample_rate
		semi = np.full(n_samples, pitch, dtype=np.float64)
		if profile.drift:
			drift_phase += rng.uniform(0.1, 0.4)
			semi += profile.drift * np.sin(2 * np.pi * 0.5 * t + drift_phase)
		if profile.vibrato:
			semi += profile.vibrato * np.sin(2 * np.pi * 5.5 * t)

		freq = 440.0 * 2.0 ** ((semi - 69) / 12.0)
		phase = 2 * np.pi * np.cumsum(freq) / sample_rate
		# voice-like timbre: fundamental + decaying harmonics
		wave = np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)

		# amplitude envelope (attack/decay) so note boundaries are detectable
		env = np.ones(n_samples)
		a = min(int(0.02 * sample_rate), n_samples // 2)
		if a > 0:
			env[:a] = np.linspace(0, 1, a)
			env[-a:] = np.linspace(1, 0, a)
		chunks.append((wave * env).astype(np.float32))

	if not chunks:
		return np.zeros(sample_rate // 2, dtype=np.float32)

	audio = np.concatenate(chunks)
	peak = np.max(np.abs(audio))
	if peak > 0:
		audio = audio / peak * 0.9
	if profile.noise_level:
		audio = audio + rng.normal(0, profile.noise_level, len(audio)).astype(np.float32)
	return audio.astype(np.float32)


def notes_to_reference_contour(notes: list[tuple[int, float]]) -> np.ndarray:
	"""Ground-truth key-invariant interval contour straight from the score.

	This is what the *database* stores per song; in production it would come
	from running the audio pipeline over a clean recording, but deriving it from
	the score gives a clean, deterministic reference for the benchmark.
	"""
	# Collapse consecutive identical pitches: the audio pitch pipeline merges
	# same-pitch frames into one note, so the reference must do the same to stay
	# representationally consistent with extracted query contours.
	pitches: list[int] = []
	for m, _ in notes:
		if m == 0:
			continue
		if pitches and pitches[-1] == m:
			continue
		pitches.append(m)
	if len(pitches) < 2:
		return np.zeros(0, dtype=np.float32)
	intervals = np.diff(np.array(pitches, dtype=np.float64))
	intervals = np.clip(intervals, -12.0, 12.0)
	return intervals.astype(np.float32)
