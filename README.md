# DeepFilterNet Audio Denoiser

A minimal local test tool to upload a noisy audio file, run it through
[DeepFilterNet3](https://github.com/rikorose/DeepFilterNet), and listen
to / download the cleaned result.

```
frontend (React/Vite) ──POST /api/process──► backend (FastAPI)
                                                   │
                                            DeepFilterNet3
                                                   │
                        ◄── streaming wav response ─┘
```

---

## Requirements

| Tool | Version |
|------|---------|
| Python | ≥ 3.9 |
| Node.js | ≥ 18 |
| npm | ≥ 9 |

---

## Setup & Run

### 1 — Backend

```bash
cd backend

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload
```

The backend listens on **http://localhost:8000**.

> **First run**: DeepFilterNet will automatically download its pretrained
> model weights (~30 MB) to `~/.cache/DeepFilterNet/` on the first request.
> This is a one-time download.

### 2 — Frontend

Open a second terminal:

```bash
cd frontend

npm install
npm run dev
```

The app is now available at **http://localhost:5173**.

---

## Usage

1. Open **http://localhost:5173** in your browser.
2. Drag-and-drop or click to upload a WAV, MP3, FLAC, OGG, or M4A file.
3. Click **✨ Clean Audio** — processing may take a few seconds.
4. Listen to the cleaned result in the audio player that appears.
5. Click **⬇ Download denoised.wav** to save it.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Returns `{"status": "ok"}` |
| `POST` | `/api/process` | Accepts `multipart/form-data` with a `file` field; returns a `audio/wav` response |

---

## Project structure

```
.
├── backend/
│   ├── main.py          # FastAPI app + DeepFilterNet inference
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js   # proxies /api → localhost:8000
│   └── src/
│       ├── main.jsx
│       ├── App.jsx      # UI: drop-zone, players, error handling
│       └── App.css      # dark theme styling
└── README.md
```

---

## Notes

- No database, no persistence — uploaded files are processed in OS temp
  directories and deleted immediately after the response is sent.
- DeepFilterNet processes audio at **48 kHz mono**. The backend automatically
  resamples and converts your file if it doesn't match.
- This is a local development tool — not intended for production deployment.
