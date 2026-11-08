"""Shared inference code for the Explainable Genre & Mood demo.

Everything the website needs (preprocessing, model classes, prediction and
explanation) lives here so the API, the export script and the demo-sample
builder all process audio exactly the way the training scripts in ``src/`` did.
"""
from .config import GENRES, MOODS  # noqa: F401
