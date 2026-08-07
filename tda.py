# File: tda.py
# Topological Data Analysis (TDA) feature extraction.
#
# Computes a rolling Persistent Entropy (H_p) of a price series using
# Takens' time-delay embedding into a 2D reconstructed phase space
# ([price_t, price_{t-1}]) and Vietoris-Rips persistent homology (via
# ripser). This mirrors the standard TDA early-warning-signal technique
# (Gidea & Katz, 2018): loops (H1 homology) in the delay-embedded point
# cloud correspond to cyclical structure in the price series, and their
# persistent entropy summarizes how "organized" vs. "chaotic" that
# structure is over each trailing window.

from typing import Optional

import numpy as np
import pandas as pd
from ripser import ripser


def _takens_embedding(window: np.ndarray, delay: int = 1, dim: int = 2) -> np.ndarray:
    """
    Reconstruct a `dim`-dimensional phase space from a 1-D price window via
    Takens' time-delay embedding: point_i = [window[i], window[i+delay], ...].

    Returns an (n_points, dim) array; n_points = len(window) - (dim - 1) * delay.
    """
    n_points = len(window) - (dim - 1) * delay
    if n_points <= 0:
        raise ValueError(
            f"Window of length {len(window)} is too short for dim={dim}, delay={delay}."
        )
    return np.array([
        [window[i + j * delay] for j in range(dim)]
        for i in range(n_points)
    ])


def _persistent_entropy(diagram: np.ndarray) -> float:
    """
    Persistent entropy of a single persistence diagram:
        H_p = -sum(p_i * log(p_i)),  p_i = lifespan_i / sum(lifespans)

    Only finite, strictly positive lifespans contribute (this drops the
    single infinite-lifespan class every H0 diagram has). Falls back to
    0.0 for the edge case where a window collapses to zero total lifespan
    (e.g. a degenerate/near-constant price window), since entropy is
    undefined there.
    """
    if diagram.size == 0:
        return 0.0
    lifespans = diagram[:, 1] - diagram[:, 0]
    lifespans = lifespans[np.isfinite(lifespans)]
    lifespans = lifespans[lifespans > 0]

    total = lifespans.sum()
    if total <= 0:
        return 0.0

    p = lifespans / total
    return float(-np.sum(p * np.log(p)))


def _window_entropy(window: np.ndarray, delay: int = 1, dim: int = 2, homology_dim: int = 1) -> float:
    """
    Persistent entropy of one price window: Takens-embed it, run ripser,
    and score the resulting homology_dim persistence diagram.

    Any failure (too-short/degenerate window, numerical issue in ripser,
    a homology dimension ripser didn't compute, etc.) falls back to 0.0
    rather than propagating, since a single bad window shouldn't abort
    feature extraction for the whole series.
    """
    try:
        points = _takens_embedding(window, delay=delay, dim=dim)
        if len(points) < 2:
            return 0.0
        result = ripser(points, maxdim=homology_dim)
        diagram = result["dgms"][homology_dim]
        return _persistent_entropy(diagram)
    except Exception:
        return 0.0


def extract_tda_features(
    df: pd.DataFrame,
    price_col: str = "Close",
    window_size: int = 30,
    delay: int = 1,
    homology_dim: int = 1,
) -> pd.Series:
    """
    Rolling Persistent Entropy (H_p) of df[price_col], one value per row.

    For each row i (i >= window_size - 1), the trailing window of
    `window_size` prices ending at i is reconstructed into a 2D phase
    space via Takens' embedding ([price_t, price_{t-1}]), and the
    persistent entropy of its `homology_dim` Vietoris-Rips persistence
    diagram is computed with ripser.

    The first `window_size - 1` rows don't have a full trailing window;
    they are backfilled with the first fully-computed entropy value so
    the returned Series has no NaNs (any still-missing values, e.g. if
    df is shorter than window_size, fall back to 0.0).

    Returns a pd.Series named "tda_entropy", aligned to df's index.
    """
    prices = df[price_col].to_numpy(dtype=float)
    n = len(prices)
    entropies = np.full(n, np.nan)

    for i in range(window_size, n + 1):
        window = prices[i - window_size:i]
        entropies[i - 1] = _window_entropy(window, delay=delay, homology_dim=homology_dim)

    series = pd.Series(entropies, index=df.index, name="tda_entropy")
    series = series.bfill().fillna(0.0)
    return series
