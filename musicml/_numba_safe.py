"""Pure-NumPy replacements for two librosa numba kernels.

On Apple Silicon, numba ``guvectorize`` kernels can segfault when PyTorch is
loaded in the same process (the team already hit this with ``beat_track``).
``chroma_stft`` reaches two such kernels through ``estimate_tuning``:
``librosa.core.pitch._pi_wrapper`` and ``librosa.util.utils._localmax``.
These replacements compute exactly the same values, so features stay identical
to the ones the models were trained on.
"""
import numpy as np


def _pi_wrapper(x, y):
    """Parabolic interpolation stencil along the last axis (writes into y)."""
    x = np.asarray(x)
    out = np.zeros(x.shape, dtype=y.dtype)
    if x.shape[-1] >= 3:
        left, mid, right = x[..., :-2], x[..., 1:-1], x[..., 2:]
        a = right + left - 2 * mid
        b = (right - left) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(np.abs(b) >= np.abs(a), 0, -b / a)
        out[..., 1:-1] = shift
    y[...] = out


def _localmax(x, y):
    """Local-maximum stencil along the last axis (writes into y)."""
    x = np.asarray(x)
    out = np.zeros(x.shape, dtype=bool)
    if x.shape[-1] >= 3:
        out[..., 1:-1] = (x[..., 1:-1] > x[..., :-2]) & (x[..., 1:-1] >= x[..., 2:])
    y[...] = out


def install():
    """Swap the numba kernels for the NumPy ones (idempotent)."""
    import librosa.core.pitch as pitch
    import librosa.util.utils as utils

    if hasattr(pitch, "_pi_wrapper"):
        pitch._pi_wrapper = _pi_wrapper
    if hasattr(utils, "_localmax"):
        utils._localmax = _localmax
