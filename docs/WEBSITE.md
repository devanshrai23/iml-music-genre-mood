# Demo website: run guide

The website has three parts:

| Part | Folder | What it does |
| --- | --- | --- |
| Model package | `musicml/` | Shared preprocessing, model classes, prediction and SHAP. Same maths as `src/`. |
| Backend | `api/` | FastAPI server: `/api/analyze` (upload audio), `/api/mood/predict`, `/api/health`. Also serves the built website. |
| Website | `web/` | React + Vite. Pages: Overview, Genre Lab, Why this genre?, Results, Mood. |

## 1. One-time setup (on the Mac with the datasets)

Datasets go where the main README says: `data/raw/Data/genres_original/` and `data/raw/dataset.csv`.

```bash
source .venv/bin/activate
pip install -r requirements.txt -r requirements-web.txt
brew install ffmpeg node   # ffmpeg decodes MP3/M4A uploads; node builds the website
```

## 2. Export the models (about 5–15 minutes on an M-series Mac)

```bash
python scripts/export_models.py
```

This runs any missing pipeline step for you (`extract_features.py`, `make_spectrograms.py`, `train_cnn.py`,
`train_crnn.py`), then writes to `models/`:

- `crnn.pt`, `cnn.pt`: the trained networks
- `rf.joblib`, `svm.joblib`: baselines fitted on the same song-level 80/20 split as the CRNN
- `mood_rf.joblib`: calibrated mood Random Forest (gitignored because it can be large; only the live backend needs it)
- `metadata.json`: test-split metrics, confusion matrices, global SHAP, mood presets

It prints each model's test accuracy, measured through `musicml/`. The CRNN should land near the README's 87%.
Retraining on a new machine can give a slightly different number.

## 3. Build the demo gallery

```bash
python scripts/build_demo_samples.py
```

This writes 20 pre-computed test-set songs (two per genre, never seen in training) to `web/public/samples/`,
plus mood examples. The deployed site works from these files even without the backend.
Use `--no-audio` if you don't want GTZAN audio inside the website (for example on a public deployment).

## 4. Run the demo (one command)

```bash
cd web && npm install && npm run build && cd ..
python -m api
```

Open http://127.0.0.1:8000. The status dot in the top bar reads **live models** when uploads work.

### While developing the website

```bash
python -m api                 # terminal 1: backend on :8000
cd web && npm run dev         # terminal 2: hot-reloading site on :5173, /api is proxied
```

## 5. Deploy the website to Vercel (showcase mode)

1. Commit `web/public/data/metrics.json` and `web/public/samples/` (created by steps 2 and 3).
2. On vercel.com: **Add New → Project**, import this repo, set **Root Directory** to `web`. Vercel reads `web/vercel.json`.
3. Deploy. Without a backend, the site shows the pre-computed songs, results and mood examples. Uploads and the mood sliders turn on
   automatically when it can reach an API.

PyTorch is too large for Vercel's serverless functions, so the live backend runs on your laptop during the demo.
To make uploads work on the public site later, host `api/` somewhere with a Python server (for example a Hugging Face
Docker Space) and set `VITE_API_URL` in the Vercel project settings.

## Demo-day checklist

- [ ] `python -m api` is running and the top bar says **live models**
- [ ] One MP3 of your own ready to upload (a well-known song works best)
- [ ] Walkthrough: Overview → Genre Lab (play a "rescue" sample, click slices) → upload a song → Why this genre? → Results (click a rock/country cell) → Mood (pick an example, drag a slider)

## Share with the team

After steps 2 and 3, commit the results so teammates and Vercel get them without retraining:

```bash
git add models/ web/public/data web/public/samples
git commit -m "Add exported models and demo samples"
git push
```
