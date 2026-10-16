# Music Genre and Mood Classification with Explainability

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-orange)

Predicts the **genre** and **mood** of music tracks and explains the predictions using **SHAP** (classical models) and **temporal attention weights** (CNN).

## Table of Contents

1. [Overview](#1-overview)
2. [Results](#2-results)
3. [Repository Structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Datasets](#5-datasets)
6. [Usage](#6-usage)
7. [Data Notes and Limitations](#7-data-notes-and-limitations)
8. [Roadmap](#8-roadmap)
9. [References](#9-references)

---

## 1. Overview

Music apps sort songs into genres (rock, jazz, classical) and moods (happy, sad, calm), but most classifiers only output a label. This project looks at **why** a model picks a label by comparing two kinds of explanation:

- **Classical models + SHAP:** a Random Forest and an SVM trained on handcrafted audio features, explained with TreeSHAP.
- **Deep model + attention:** a CNN on log-Mel spectrograms with a temporal attention layer, explained through its attention weights.

Two datasets are used, because GTZAN only has genre labels:

| Task | Dataset | Input | Models |
|---|---|---|---|
| Genre (10 classes) | GTZAN | 89 librosa features, or log-Mel spectrograms | Random Forest, SVM, Attention-CNN |
| Mood (4 classes) | Spotify Tracks (Kaggle) | 9 tabular audio features | Random Forest |

Mood labels come from a 0.5 threshold on Spotify's `valence` and `energy` (happy, tense, sad, calm). Both columns are **excluded from the model inputs** to prevent label leakage.

### Features

- 89 named audio features per song (MFCCs, chroma, spectral descriptors, zero-crossing rate, RMS energy, tempo), extracted in parallel with librosa
- Cached log-Mel spectrograms (128 mel bins) for deep learning
- Attention-CNN with a learned temporal attention layer over 40 time segments
- TreeSHAP feature attributions for both the genre and mood models
- Song-level evaluation that avoids train/test leakage from audio segments and duplicate tracks

## 2. Results

| Task | Model | Evaluation | Accuracy | Macro F1 |
|---|---|---|---|---|
| Genre (GTZAN, N=999) | Random Forest (300 trees) | 5-fold stratified CV | 74.5% | 0.740 |
| Genre (GTZAN, N=999) | SVM (RBF, C=10) | 5-fold stratified CV | 77.6% | 0.776 |
| Genre (GTZAN, N=999) | Attention-CNN (preliminary) | single 80/20 split | 83.0% | 0.826 |
| Mood (Spotify, N=89,741) | Random Forest (200 trees) | 5-fold stratified CV | 70.5% | 0.660 |

> **Note:** the CNN was evaluated on a single 80/20 split (about 200 clips), while the baselines use 5-fold cross-validation. The comparison is therefore approximate. See [Roadmap](#8-roadmap) for planned changes.

Observations:

- Classical and metal are the easiest genres for all models. Rock is the hardest, and is often confused with disco, country, blues, metal and reggae.
- The CNN improves most on rhythm-based genres (reggae, disco, hip-hop), where averaging features over 30 seconds likely hides beat structure.
- For mood, the *calm* class has the lowest recall (0.36). It is the smallest class and is often confused with *sad*.

## 3. Repository Structure

```
iml-music-genre-mood/
├── data/
│   ├── raw/                 # downloaded datasets (not committed)
│   └── processed/           # generated features and spectrograms (not committed)
├── notebooks/               # exploration notebooks
├── reports/                 # generated figures
│   ├── confusion_RandomForest.png
│   ├── confusion_SVM.png
│   ├── confusion_CNN.png
│   ├── confusion_Mood_RF.png
│   ├── shap_global_importance.png
│   ├── shap_mood_importance.png
│   └── cnn_attention_example.png
├── src/
│   ├── extract_features.py  # 89 librosa features per song -> CSV
│   ├── make_spectrograms.py # log-Mel spectrograms -> .npy files
│   ├── baseline.py          # Random Forest + SVM, 5-fold CV, confusion matrices
│   ├── train_cnn.py         # Attention-CNN training (PyTorch)
│   └── ...                  # mood baseline and SHAP analysis scripts
├── requirements.txt
├── .gitignore
└── README.md
```

## 4. Installation

Instructions are for macOS (developed on Apple Silicon). Notes for other platforms are at the end of this section.

### 4.1 System dependencies

```bash
# Xcode command line tools and Homebrew (skip any you already have)
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Git, Python 3.11, and ffmpeg (audio decoding)
brew install git python@3.11 ffmpeg
```

### 4.2 Clone the repository

```bash
git clone https://github.com/devanshrai23/iml-music-genre-mood.git
cd iml-music-genre-mood
```

### 4.3 Create a virtual environment and install packages

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # run this in every new terminal

pip install --upgrade pip
pip install -r requirements.txt
pip install torch                # required for the CNN
```

`requirements.txt` includes numpy, pandas, scikit-learn, librosa, soundfile, matplotlib, seaborn, shap, jupyterlab, ipykernel, joblib and kaggle. On Apple Silicon, the default PyTorch build supports the GPU through the MPS backend.

### 4.4 Verify the installation

```bash
python -c "import librosa, sklearn, shap, pandas, torch; print('librosa', librosa.__version__)"
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
ffmpeg -version | head -1
```

If a PyTorch operation is unsupported on MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to the CPU for that operation.

### 4.5 Other platforms

- **Linux:** install `ffmpeg` and Python 3.11 with your package manager (for example `sudo apt install ffmpeg python3.11 python3.11-venv`), then follow 4.2 to 4.4. Install the CUDA build of PyTorch to use an NVIDIA GPU.
- **Windows:** install Python 3.11 and ffmpeg (add it to `PATH`), activate the environment with `.venv\Scripts\activate`, then follow the same steps.
- **No local GPU:** the classical baselines run fine on a CPU. For the CNN, Google Colab's free GPU tier is sufficient.

## 5. Datasets

Datasets are not stored in this repository. Download them into `data/raw/`.

### 5.1 GTZAN (genre, raw audio)

- **Source:** https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
- **Contents:** 1,000 clips (10 genres x 100), each 30 seconds, 22,050 Hz mono WAV, about 1.2 GB. Also includes spectrogram images and two CSVs of precomputed features.
- **Download:** log in to Kaggle, click **Download**, and unzip into `data/raw/`. The expected layout is:

```
data/raw/Data/
├── genres_original/     # blues/, classical/, ..., rock/
├── images_original/
├── features_30_sec.csv
└── features_3_sec.csv
```

Command-line alternative (requires a Kaggle API token):

```bash
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification -p data/raw --unzip
```

### 5.2 Spotify Tracks Dataset (mood, tabular)

- **Source:** https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset (the leading dash in the URL is correct)
- **Contents:** about 114,000 rows and 21 columns (danceability, energy, valence, tempo, loudness, track_genre, and more).
- **Download:** place `dataset.csv` at `data/raw/dataset.csv`.

```bash
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/raw --unzip
```

### 5.3 Check the data

```bash
python -c "import librosa; y,sr=librosa.load('data/raw/Data/genres_original/blues/blues.00000.wav'); print(y.shape, sr)"
```

Expected output: about 660,000 samples at 22,050 Hz.

## 6. Usage

Run all commands from the repository root with the virtual environment active. The scripts expect the paths from [Datasets](#5-datasets); adjust the path constants at the top of each script if your folders differ.

| Step | Command | Output |
|---|---|---|
| 1. Extract features | `python src/extract_features.py` | `data/processed/gtzan_librosa_features.csv` |
| 2. Classical baselines | `python src/baseline.py` | Metrics in the terminal, confusion matrices in `reports/` |
| 3. Cache spectrograms | `python src/make_spectrograms.py` | `data/processed/mel/<genre>/*.npy` |
| 4. Train the Attention-CNN | `python src/train_cnn.py` | Trained model, confusion matrix, attention plots |

Notes:

- Step 1 takes a few minutes. It runs in parallel and skips unreadable files, so expect 999 rows (see [Data Notes](#7-data-notes-and-limitations)).
- Step 3 produces a few hundred MB of arrays, which are gitignored.
- Step 4 uses the MPS backend when available, otherwise the CPU.

## 7. Data Notes and Limitations

- **Corrupted file:** `jazz.00054.wav` in the Kaggle copy of GTZAN cannot be decoded and is skipped, which leaves 999 tracks.
- **GTZAN flaws:** the dataset contains a few duplicate recordings, repeated artists and debatable genre labels (Sturm, 2013). Reported accuracies should be read with this in mind.
- **Segment leakage:** `features_3_sec.csv` is not used with a random split, because segments of the same song would appear in both train and test. Features are extracted from full 30-second clips, and any segment-level training groups folds by song.
- **Mood circularity:** mood labels are computed from `valence` and `energy`, so both are excluded from the inputs. The remaining features are danceability, loudness, tempo, acousticness, instrumentalness, speechiness, liveness, mode and key. Loudness is typically correlated with energy, so it may partly stand in for the removed label.
- **Estimated labels:** `valence` and `energy` are Spotify's own audio estimates, not human ratings.
- **Duplicate tracks:** the same Spotify track appears under several genres, so rows are deduplicated by `track_id` (114,000 down to 89,741).
- **Unlinked datasets:** GTZAN clips have no Spotify IDs, so genre is modelled on GTZAN audio and mood on Spotify tabular features.

## 8. Roadmap

- [ ] Extend the CNN to a CRNN (CNN + BiGRU + attention) trained on 3-second slices with `GroupKFold` grouped by song
- [ ] Evaluate all models with the same 5-fold protocol
- [ ] Improve recall on the *calm* mood class (class weights, oversampling)
- [ ] Compare SHAP explanations with attention weights on the same test tracks, and test the attention peaks by masking
- [ ] Check generalization on a small FMA-small subset

## 9. References

1. G. Tzanetakis and P. Cook. Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing*, 2002. (GTZAN dataset)
2. B. L. Sturm. The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use. *arXiv:1306.1461*, 2013.
3. J. A. Russell. A circumplex model of affect. *Journal of Personality and Social Psychology*, 1980.
4. S. M. Lundberg and S.-I. Lee. A unified approach to interpreting model predictions. *NeurIPS*, 2017.
5. S. M. Lundberg et al. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2020.
6. B. McFee et al. librosa: Audio and music signal analysis in Python. *Proc. SciPy*, 2015.
7. D. S. Park et al. SpecAugment: A simple data augmentation method for automatic speech recognition. *Interspeech*, 2019.
