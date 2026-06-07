# Pitchr 🎵

Hum a song. We'll name it.

## How it works

1. User hums a melody on the portal page
2. Audio is captured and sent to the backend
3. YIN pitch detection extracts the pitch sequence
4. Melody contour is normalized (key-invariant)
5. DTW matching finds the closest song in the database
6. Result is returned to the user

## Tech

- **Pitch detection**: YIN algorithm (from scratch)
- **Matching**: Dynamic Time Warping (from scratch)
- **Framework**: Frappe
- **No ML, no external fingerprinting libraries**

## Modules

| Module | Description |
|--------|-------------|
| Audio Capture | Records mic input via sounddevice |
| YIN Pitch Detection | Extracts pitch sequence from audio |
| Contour Normalization | Makes matching key-invariant |
| DTW Matcher | Finds best matching song |
| Song Indexer | Indexes songs into the database |
| API Layer | Frappe whitelisted REST endpoints |
| Portal Page | Frontend UI for humming |

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