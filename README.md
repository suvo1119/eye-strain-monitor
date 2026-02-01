# Eye Strain Monitor 👁️‍🗨️

A simple webcam tool that uses MediaPipe Face Landmarker to detect blinks and estimate eye strain (PERCLOS, blinks/min, average blink duration). It shows color-coded recommendations on high strain and generates session reports (text + per-minute CSV) when you close the program.

---

## 🔧 Requirements

- **OS:** Linux (tested)
- **Python:** 3.11 recommended
- **Camera:** Access to a webcam (device index 0 by default)

## 📦 Install

1. Clone the repository and cd into it:

```bash
git clone <repo-url>
cd eye-strain
```

2. Create and activate a virtual environment (recommended):

```bash
python3.11 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> The script automatically downloads the `face_landmarker_v2.task` model on first run if it is not present.

## ▶️ Run

```bash
python3 main.py
```

- Press `ESC` (or Ctrl+C) to quit; a session report is generated on exit.
- If the camera index 0 does not work, change `cv2.VideoCapture(0)` to another index.
- Let the program run for ~1 minute for reliable metrics.

## 📊 Reports (generated on exit)

- Reports are saved to the `reports/` folder with timestamped filenames, e.g.:
  - `reports/session_20260201_104312.txt` — human-readable summary
  - `reports/session_20260201_104312.csv` — minute-by-minute CSV log
- The CSV contains columns: `minute_index, start_iso, frames, blinks_in_min, blinks_per_min, perclos, avg_blink_ms`.

## ⚙️ Configuration & Tuning

Open `main.py` and adjust these constants near the top:

- `WINDOW_SECONDS` — rolling window size for metrics (default 60s)
- `BLINK_EAR_THRESH` — threshold for blink detection
- `PERCLOS_EAR_THRESH` — threshold to mark a frame as "eyes closed" for PERCLOS
- `RECOMMENDATION_COOLDOWN` — seconds between recommendation prompts
- `RECOMMENDATION_LINES` — the on-screen recommendation messages

Tweak thresholds for your lighting and webcam distance to improve sensitivity.

## 🔔 Optional features

- I can add an audible alert or desktop notification when **High** strain is detected.
- I can also add an option to auto-open the report after the session (e.g., with `xdg-open`).

If you'd like either, tell me which and I'll add it.

## 🌐 Demo / GitHub Pages

A live browser demo is available in the `docs/` folder (uses MediaPipe FaceMesh JS). To publish the demo on GitHub Pages:

1. Push your repo to GitHub (main branch).
2. A GitHub Actions workflow (`.github/workflows/deploy-pages.yml`) will deploy the `docs/` folder to GitHub Pages on every push to `main`.
3. After the first successful run, go to **Settings → Pages** in your repository and ensure the site is published. The URL will be `https://<your-username>.github.io/<repo-name>/`.

The demo opens the camera in the browser and runs the face mesh locally — it's the easiest way to provide a single clickable link on your CV that directly opens the camera and runs the monitor.

Local testing: to test the demo locally (recommended) run a simple static server from the repo root and open `http://localhost:8000/docs` in a browser that supports getUserMedia:

```bash
# from repo root
python3 -m http.server 8000
# then open http://localhost:8000/docs in Chrome/Edge
```

## ❗ Troubleshooting

- If you see Qt warnings like `Could not find the Qt platform plugin "wayland"`, either install the system Qt packages or try running with `QT_QPA_PLATFORM=xcb python3 main.py`.
- If the camera doesn't open, verify permissions and try different device indices.
- If `mediapipe` install fails on your Python version, use Python 3.11 (recommended). Use `python3.11 -m venv venv` to create the venv.

## 📝 Files

- `main.py` — main application (reports, CSV logging, recommendations)
- `requirements.txt` — Python dependencies
- `README.md` — this file
- `reports/` — generated session reports after each run

## 🔒 License & Acknowledgements

This project uses MediaPipe (Google). See the MediaPipe project for license details.

---

If you'd like, I can add automatic report opening, audible alerts, or CLI flags for report settings — tell me which option and I will implement it. 🎯