# Pitchr Benchmark & Standalone Pipeline

A **Frappe-free** harness to develop, test, and measure the accuracy of the
query-by-humming engine. Everything here runs with plain Python — no bench, no
database.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Run the tests

```bash
python -m unittest benchmark.test_standalone   # unit + end-to-end, no Frappe
```

## Run the accuracy benchmark

```bash
# Built-in 25-song public-domain corpus
python -m benchmark.run_benchmark --queries 4

# Scale up to any directory of MIDI files (e.g. the Nottingham dataset)
python -m benchmark.run_benchmark --midi-dir /path/to/midis --midi-limit 500
```

## How it works

| Component | File | Role |
|-----------|------|------|
| Corpus | `corpus.py` | Build the song database from built-in melodies or a MIDI directory (skyline melody extraction). |
| Synthesizer | `synth.py` | Render note sequences into *hummed* audio with realistic distortion profiles (key shift, tempo, pitch jitter, drift/vibrato, wrong/dropped notes, noise, partial fragment). |
| Pipeline | `../pitchr/pitch/pipeline.py` | audio → YIN pitch track → note-level key-invariant interval contour → subsequence-DTW match. |
| Harness | `run_benchmark.py` | Top-1 / Top-3 / MRR per distortion profile. |

The synthesizer adds distortions **independent** of how the database is built,
so the benchmark is not circular: a key-shifted, tempo-changed, partially-hummed
rendition must still be matched back to the correct song.

## Distortion profiles

- **clean** – faithful rendition (sanity check / upper bound)
- **light** – small key shift, mild jitter, 60% fragment
- **realistic** – key shift, tempo change, intonation drift, occasional wrong/dropped notes, noise, ~50% fragment
- **heavy** – large shifts and high error rates (stress test)
