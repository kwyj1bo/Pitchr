# Pitchr 🎵

Hum a song. We'll name it.

## How it works

1. User hums a melody on the portal page
2. The browser decodes the recording to raw mono float32 PCM @ 22.05 kHz and uploads it
3. YIN pitch detection extracts a per-frame pitch track
4. The track is segmented into **notes** and turned into a key-invariant **interval contour**
5. **Subsequence DTW** finds the song whose melody best contains the hummed fragment
6. Result (plus top-k candidates) is returned to the user

## Tech

- **Pitch detection**: YIN algorithm (from scratch)
- **Matching**: subsequence Dynamic Time Warping (from scratch) — a partial hum matches a full song
- **Representation**: note-level semitone intervals (key- and tempo-invariant)
- **Framework**: Frappe
- **No ML, no external fingerprinting libraries**

## Modules

| Module | Description |
|--------|-------------|
| Audio Capture | Records mic input (browser Web Audio API / `sounddevice`) |
| YIN Pitch Detection | Extracts a pitch track; lag search bounded to the vocal range |
| Melody / Contour | Note segmentation + key-invariant interval contour |
| Subsequence DTW Matcher | Matches a hummed fragment against full-length songs |
| Pipeline | Framework-independent `audio → contour → match` (reused by API and benchmark) |
| Song Indexer | Indexes songs into the database with the same pipeline |
| API Layer | Frappe whitelisted REST endpoints |
| Portal Page | Frontend UI for humming |

## Accuracy & testing

The recognition engine is fully testable **without a Frappe bench** — see
[`benchmark/`](benchmark/README.md). A humming synthesizer generates queries
with realistic distortions (key shift, tempo change, intonation drift,
wrong/dropped notes, noise, partial fragments) and the harness reports Top-1 /
Top-3 / MRR per distortion profile.

```bash
pip install -r requirements-dev.txt
python -m unittest benchmark.test_standalone          # unit + end-to-end tests
python -m benchmark.run_benchmark --queries 4          # built-in 25-song corpus
python -m benchmark.run_benchmark --midi-dir mids/     # scale to any MIDI library
```

On the built-in corpus: **clean 100%**, **light 95%**, **realistic 71% Top-1 / 92% Top-3**.

## Installation

```bash
cd $PATH_TO_YOUR_BENCH
bench get-app https://github.com/kwyj1bo/Pitchr --branch develop
bench install-app pitchr
```

## Usage

Visit `/pitchr` on your Frappe site, press the mic button, and hum.

## Contributing

This app uses `pre-commit` for code formatting and linting. Install and enable it:

```bash
cd apps/pitchr
pre-commit install
```

Pre-commit is configured to use: ruff, eslint, prettier, pyupgrade.

## CI

- **CI**: Installs the app and runs unit tests on every push to `develop`
- **Linters**: Runs Frappe Semgrep Rules and pip-audit on every pull request

## License

MIT