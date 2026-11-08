"""FastAPI backend for the demo website.

    python -m api              # http://localhost:8000  (serves web/dist too, if built)
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musicml.config import MODELS_DIR  # noqa: E402
from musicml.predict import GenrePredictor, MoodPredictor, load_metadata  # noqa: E402

MAX_UPLOAD_MB = 30
ALLOWED = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif"}
DIST = ROOT / "web" / "dist"

app = FastAPI(title="Explainable Genre & Mood API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

state = {"genre": None, "mood": None, "errors": {}}


@app.on_event("startup")
def load_models():
    try:
        state["genre"] = GenrePredictor(MODELS_DIR)
    except Exception as e:  # keep the server up so /api/health can explain what is wrong
        state["errors"]["genre"] = str(e)
    try:
        state["mood"] = MoodPredictor(MODELS_DIR)
    except Exception as e:
        state["errors"]["mood"] = str(e)
    print("Genre models:", state["genre"].available if state["genre"] else state["errors"].get("genre"))
    print("Mood model:", "loaded" if state["mood"] else state["errors"].get("mood"))


@app.get("/api/health")
def health():
    g = state["genre"]
    return {"status": "ok", "genre_models": g.available if g else [], "mood": state["mood"] is not None,
            "errors": state["errors"]}


@app.get("/api/meta")
def meta():
    return load_metadata(MODELS_DIR)


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), offset: Optional[float] = Form(None)):
    g = state["genre"]
    if g is None:
        raise HTTPException(503, f"Genre models are not loaded: {state['errors'].get('genre')}")
    suffix = Path(file.filename or "clip.wav").suffix.lower() or ".wav"
    if suffix not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type {suffix}. Use WAV, MP3, FLAC, OGG or M4A.")
    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB. Trim it or upload a shorter clip.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        result = g.analyze_file(path, offset=offset, title=file.filename)
        result["id"] = "upload"
        return result
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Could not decode or analyse this file ({type(e).__name__}). "
                                 "MP3/M4A decoding needs ffmpeg installed.")
    finally:
        os.unlink(path)


class MoodRequest(BaseModel):
    features: dict


@app.post("/api/mood/predict")
def mood_predict(req: MoodRequest):
    m = state["mood"]
    if m is None:
        raise HTTPException(503, f"Mood model is not loaded: {state['errors'].get('mood')}")
    try:
        return m.predict(req.features)
    except ValueError as e:
        raise HTTPException(422, str(e))


# ----------------------------------------------------------------- built website (npm run build)
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        target = (DIST / path).resolve()
        if path and target.is_file() and DIST.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
