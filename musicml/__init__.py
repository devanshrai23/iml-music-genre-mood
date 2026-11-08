"""Shared inference code for the Explainable Genre & Mood demo.

Everything the website needs (preprocessing, model classes, prediction and
explanation) lives here so the API, the export script and the demo-sample
builder all process audio exactly the way the training scripts in ``src/`` did.

Import this package BEFORE numpy / librosa / torch. On Apple Silicon, PyTorch and
numba (used by librosa) each ship an OpenMP runtime; with multi-threading on, the
two can segfault in one process. The settings below are the same ones the
training scripts in src/ use.
"""
import faulthandler
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Print a Python traceback instead of a bare "segmentation fault" if a native crash happens.
faulthandler.enable()

# Load PyTorch's OpenMP runtime first (when installed) so librosa/numba reuse it.
try:  # noqa: SIM105
    import torch  # noqa: F401
except ImportError:
    pass

from .config import GENRES, MOODS  # noqa: E402,F401
