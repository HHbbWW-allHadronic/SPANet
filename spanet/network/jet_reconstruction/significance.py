"""Weighted S/sqrt(B) metrics from detection scores, for use as a SPANet validation metric.

Pure NumPy (no torch / lightning) so it can be unit tested on its own.
All yields are sums of `WEIGHTS/physical` (xsec * lumi / n_events * generator weights), i.e. expected events at full lumi.
"""
from typing import Dict, Sequence

import numpy as np

# Same score binning as the OSCA scans (H1/W1 detection), so the number is directly comparable to the scan printouts.
DEFAULT_BIN_EDGES = (0.0, 0.48, 0.8, 0.92, 0.96, 1.0)
DEFAULT_EFFICIENCIES = (0.3, 0.5, 0.7)


def binned_significance(score, is_signal, weight, bin_edges=DEFAULT_BIN_EDGES, min_b_raw=5) -> float:
    """Z = sqrt(sum_bins S^2 / B). Bins with fewer than `min_b_raw` raw background events (or B <= 0) are skipped."""
    edges = np.asarray(bin_edges, dtype=np.float64)
    # np.digitize puts score == last edge past the end; fold it into the last bin.
    index = np.clip(np.digitize(score, edges[1:-1]), 0, len(edges) - 2)
    nbins = len(edges) - 1

    s = np.bincount(index[is_signal], weights=weight[is_signal], minlength=nbins)
    b = np.bincount(index[~is_signal], weights=weight[~is_signal], minlength=nbins)
    b_raw = np.bincount(index[~is_signal], minlength=nbins)

    ok = (b_raw >= min_b_raw) & (b > 0)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.sum(s[ok] ** 2 / b[ok])))


def s_over_sqrt_b_at_efficiency(score, is_signal, weight, efficiency, min_b_raw=5) -> float:
    """S/sqrt(B) for the loosest cut whose weighted signal efficiency is >= `efficiency`."""
    s_score, s_weight = score[is_signal], weight[is_signal]
    if s_score.size == 0 or s_weight.sum() <= 0:
        return float("nan")

    order = np.argsort(-s_score, kind="stable")
    cumulative = np.cumsum(s_weight[order]) / s_weight.sum()
    # With negative generator weights the cumulative efficiency is not monotonic, so take the FIRST index that reaches
    # the target (np.searchsorted would assume sorted input).
    reached = cumulative >= efficiency
    cut_index = int(np.argmax(reached)) if reached.any() else len(order) - 1
    threshold = s_score[order][cut_index]

    s = s_weight[s_score >= threshold].sum()
    passing_b = (~is_signal) & (score >= threshold)
    if passing_b.sum() < min_b_raw:
        return float("nan")
    b = weight[passing_b].sum()
    return float(s / np.sqrt(b)) if b > 0 else float("nan")


def significance_metrics(
    scores: Dict[str, np.ndarray],
    is_signal: np.ndarray,
    weight: np.ndarray,
    bin_edges: Sequence[float] = DEFAULT_BIN_EDGES,
    efficiencies: Sequence[float] = DEFAULT_EFFICIENCIES,
    min_b_raw: int = 5,
) -> Dict[str, float]:
    is_signal = np.asarray(is_signal, dtype=bool)
    weight = np.asarray(weight, dtype=np.float64)

    metrics = {}
    for name, score in scores.items():
        score = np.asarray(score, dtype=np.float64)
        metrics[f"signif/{name}/binned_Z"] = binned_significance(score, is_signal, weight, bin_edges, min_b_raw)
        for eff in efficiencies:
            metrics[f"signif/{name}/s_over_sqrtb_eff{int(round(eff * 100))}"] = \
                s_over_sqrt_b_at_efficiency(score, is_signal, weight, eff, min_b_raw)
    return metrics