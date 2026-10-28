"""Weighted S/sqrt(B) metrics from detection scores, for use as a SPANet validation metric.

Pure NumPy (no torch / lightning) so it can be unit tested on its own.
All yields are sums of `WEIGHTS/physical` (xsec * lumi / n_events * generator weights), i.e. expected events at full lumi.
"""
from typing import Dict, Optional, Sequence

import numpy as np

# Same score binning as the OSCA scans (H1/W1 detection), so the number is directly comparable to the scan printouts.
DEFAULT_BIN_EDGES = (0.0, 0.48, 0.8, 0.92, 0.96, 1.0)
DEFAULT_EFFICIENCIES = (0.3, 0.5, 0.7)
# Extra, finer UNIFORM binnings of [0, 1] that are logged next to the default metric (they never replace it).
DEFAULT_EXTRA_BIN_COUNTS = (10, 20, 50, 100)
# Number of equal-signal-efficiency bins (each holds 1/N of the signal yield), logged as binned_Z_e{N}.
DEFAULT_EFF_BIN_COUNTS = (5, 10, 20)
# Mass binning: n x n signal-quantile grid in (m_Hbb, m_W1) -> n^2 cells, logged as signif/mass/binned_Z_g{n};
# crossed with k equal-signal-efficiency bins of the joint score -> signif/mass/binned_Z_g{n}_e{k} (n^2 * k cells).
DEFAULT_MASS_GRID_COUNTS = (2, 3, 4, 5, 7, 10, 15, 20)
DEFAULT_MASS_CROSS_EFF = (3,)


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


def _bin_yields(score, is_signal, weight, edges):
    """Weighted signal yield, weighted background yield and raw background count per bin."""
    nbins = len(edges) - 1
    index = np.clip(np.digitize(score, edges[1:-1]), 0, nbins - 1)
    s = np.bincount(index[is_signal], weights=weight[is_signal], minlength=nbins)
    b = np.bincount(index[~is_signal], weights=weight[~is_signal], minlength=nbins)
    b_raw = np.bincount(index[~is_signal], minlength=nbins)
    b_sq = np.bincount(index[~is_signal], weights=weight[~is_signal] ** 2, minlength=nbins)
    return s, b, b_raw, b_sq


def binned_significance_with_coverage(score, is_signal, weight, bin_edges, min_b_raw=5):
    """Same Z as `binned_significance` (sparse bins are DROPPED), plus the fraction of the total signal yield that sits
    in the kept bins. Coverage < 1 means the finest, most signal-like bins were thrown away for lack of background rows."""
    s, b, b_raw, _ = _bin_yields(score, is_signal, weight, np.asarray(bin_edges, dtype=np.float64))
    ok = (b_raw >= min_b_raw) & (b > 0)
    if not ok.any():
        return float("nan"), float("nan")
    z = float(np.sqrt(np.sum(s[ok] ** 2 / b[ok])))
    total = s.sum()
    return z, (float(s[ok].sum() / total) if total > 0 else float("nan"))


def merged_significance(score, is_signal, weight, n_fine=100, min_b_raw=5, min_b_eff=10.0):
    """Z on a fine uniform grid of `n_fine` bins, with sparse bins MERGED instead of dropped.

    Walking down from the highest score, adjacent fine bins are accumulated until the merged bin holds at least
    `min_b_raw` raw background rows AND an effective background count (sum w)^2 / sum w^2 of at least `min_b_eff`
    (and B > 0). The effective count matters because a few low-weight background rows make B tiny and S^2/B huge; a raw
    row count alone does not protect against that when the weights vary a lot. The merge decision uses background only,
    so it cannot select on signal fluctuations, and no signal is thrown away. A leftover low-score remainder with too
    little background is folded into the last merged bin. Returns (Z, number_of_merged_bins).
    """
    edges = np.linspace(0.0, 1.0, n_fine + 1)
    s, b, b_raw, b_sq = _bin_yields(score, is_signal, weight, edges)

    merged_s, merged_b = [], []
    cs = cb = cb2 = 0.0
    cr = 0
    for i in range(n_fine - 1, -1, -1):
        cs += s[i]
        cb += b[i]
        cb2 += b_sq[i]
        cr += b_raw[i]
        if cr >= min_b_raw and cb > 0 and cb2 > 0 and (cb * cb / cb2) >= min_b_eff:
            merged_s.append(cs)
            merged_b.append(cb)
            cs = cb = cb2 = 0.0
            cr = 0
    if not merged_s:
        return float("nan"), 0
    merged_s[-1] += cs
    merged_b[-1] += cb

    ms, mb = np.asarray(merged_s), np.asarray(merged_b)
    ok = mb > 0
    if not ok.any():
        return float("nan"), 0
    return float(np.sqrt(np.sum(ms[ok] ** 2 / mb[ok]))), int(ok.sum())


def efficiency_binned_significance(score, is_signal, weight, n_bins=10, min_b_raw=5, min_b_eff=10.0):
    """Z with bins of equal SIGNAL EFFICIENCY: the edges are the weighted signal quantiles, so each bin holds 1/n_bins of
    the signal yield. Because the edges follow the signal, Z is unchanged by any monotonic transformation of the score
    (sigmoid, logit, geometric mean, ...), and the bins are narrow exactly where the signal is dense.

    With S_i = S_tot / n, Z^2 = (S_tot^2 / n^2) * sum_i 1/B_i; in the fine-bin limit Z^2 / (S_tot^2 / B_tot) is the average,
    over signal events, of (fraction of signal / fraction of background) in their bin.

    Sparse bins are MERGED from the top (highest score first) until the merged bin holds at least `min_b_raw` raw
    background rows and an effective background count (sum w)^2 / sum w^2 >= `min_b_eff`, as in `merged_significance`.
    The edges use signal only, so merging cannot select on background fluctuations that the edges depend on.
    Returns (Z, number_of_bins_used).
    """
    score = np.asarray(score, dtype=np.float64)
    s_score, s_weight = score[is_signal], weight[is_signal]
    total = s_weight.sum()
    if s_score.size == 0 or total <= 0:
        return float("nan"), 0

    order = np.argsort(s_score, kind="stable")
    # Fraction of the signal yield BELOW each signal event; made monotonic in case of negative generator weights.
    cumulative = np.maximum.accumulate(np.cumsum(s_weight[order]) / total)
    targets = np.arange(1, n_bins) / n_bins
    cut = np.clip(np.searchsorted(cumulative, targets, side="left"), 0, len(order) - 1)
    edges = np.unique(s_score[order][cut])           # ties (e.g. saturated scores) collapse into fewer bins
    nbins = len(edges) + 1
    index = np.searchsorted(edges, score, side="right")   # bin 0 = lowest score; score == edge goes to the upper bin

    s = np.bincount(index[is_signal], weights=weight[is_signal], minlength=nbins)
    b = np.bincount(index[~is_signal], weights=weight[~is_signal], minlength=nbins)
    b_raw = np.bincount(index[~is_signal], minlength=nbins)
    b_sq = np.bincount(index[~is_signal], weights=weight[~is_signal] ** 2, minlength=nbins)

    merged_s, merged_b = [], []
    cs = cb = cb2 = 0.0
    cr = 0
    for i in range(nbins - 1, -1, -1):
        cs += s[i]
        cb += b[i]
        cb2 += b_sq[i]
        cr += b_raw[i]
        if cr >= min_b_raw and cb > 0 and cb2 > 0 and (cb * cb / cb2) >= min_b_eff:
            merged_s.append(cs)
            merged_b.append(cb)
            cs = cb = cb2 = 0.0
            cr = 0
    if not merged_s:
        return float("nan"), 0
    merged_s[-1] += cs
    merged_b[-1] += cb
    ms, mb = np.asarray(merged_s), np.asarray(merged_b)
    ok = mb > 0
    if not ok.any():
        return float("nan"), 0
    return float(np.sqrt(np.sum(ms[ok] ** 2 / mb[ok]))), int(ok.sum())


def find_kinematic_columns(feature_infos):
    """feature_infos: sequence of objects with .name and .log_scale (SPANet FeatureInfo). Returns
    {'pt': (i, log), 'eta': ..., 'phi': ..., 'e' or 'mass': ...} or None if the four-vector cannot be rebuilt."""
    cols = {f.name: (i, bool(f.log_scale)) for i, f in enumerate(feature_infos)}
    if not all(k in cols for k in ("pt", "eta", "phi")) or not any(k in cols for k in ("e", "energy", "mass", "m")):
        return None
    out = {k: cols[k] for k in ("pt", "eta", "phi")}
    for k in ("e", "energy"):
        if k in cols:
            out["e"] = cols[k]
            return out
    out["mass"] = cols["mass"] if "mass" in cols else cols["m"]
    return out


def pair_invariant_mass(data, mask, pair_index, cols):
    """Invariant mass of the jets picked by `pair_index` (batch, n_jets_in_pair), from the SPANet source tensor.

    `data` is the batch source tensor (batch, max_jets, features) AS SPANet stores it: features flagged log_normalize are
    log(1 + x) (pt, e) -- undone with expm1 -- and the mean/std standardisation is applied later inside the network, so
    eta and phi are raw. `mask` marks real jets. A pair with an out-of-range index or a masked-out jet gives NaN.
    """
    data = np.asarray(data, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    pair_index = np.asarray(pair_index, dtype=np.int64)
    b, j = mask.shape

    def col(name):
        i, is_log = cols[name]
        x = data[..., i]
        return np.expm1(x) if is_log else x

    pt, eta, phi = col("pt"), col("eta"), col("phi")
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    if "e" in cols:
        en = col("e")
    else:
        en = np.sqrt(col("mass") ** 2 + px ** 2 + py ** 2 + pz ** 2)

    valid = np.all((pair_index >= 0) & (pair_index < j), axis=1)
    safe = np.clip(pair_index, 0, j - 1)
    valid &= np.all(np.take_along_axis(mask, safe, axis=1), axis=1)

    tot = [np.take_along_axis(v, safe, axis=1).sum(axis=1) for v in (en, px, py, pz)]
    m2 = tot[0] ** 2 - tot[1] ** 2 - tot[2] ** 2 - tot[3] ** 2
    m = np.sqrt(np.clip(m2, 0.0, None))
    return np.where(valid, m, np.nan)


def _signal_quantile_edges(x, is_signal, weight, n_bins):
    """Inner edges (n_bins - 1 of them, ties collapsed) at the weighted signal quantiles of x; NaNs are ignored."""
    x = np.asarray(x, dtype=np.float64)
    keep = is_signal & np.isfinite(x)
    xs, ws = x[keep], weight[keep]
    total = ws.sum()
    if xs.size == 0 or total <= 0 or n_bins < 2:
        return np.empty(0)
    order = np.argsort(xs, kind="stable")
    cumulative = np.maximum.accumulate(np.cumsum(ws[order]) / total)
    cut = np.clip(np.searchsorted(cumulative, np.arange(1, n_bins) / n_bins, side="left"), 0, len(order) - 1)
    return np.unique(xs[order][cut])


def mass_binned_significance(masses, is_signal, weight, n_per_axis=3, score=None, n_score_bins=0,
                             min_b_raw=5, min_b_eff=10.0):
    """Z with bins in the predicted MASSES (one axis per row of `masses`, e.g. m_Hbb and m_W1), optionally crossed with
    `n_score_bins` equal-signal-efficiency bins of `score`.

    Every axis is cut at its weighted SIGNAL quantiles (n_per_axis bins per axis), so the grid follows where the signal
    sits and needs no hand-picked mass windows; the cells are n_per_axis ** n_axes (times n_score_bins). Events whose mass
    is NaN (no valid jet pair) get their own extra bin per axis. Cells have no natural order, so a sparse cell (fewer than
    `min_b_raw` raw background rows, effective background count below `min_b_eff`, or B <= 0) is POOLED with the other
    sparse cells into one remainder bin; if that remainder is still too sparse it is added to the cell with the most
    background. The pooling uses background only. Returns (Z, number_of_bins_used).
    """
    masses = np.atleast_2d(np.asarray(masses, dtype=np.float64))
    is_signal = np.asarray(is_signal, dtype=bool)
    weight = np.asarray(weight, dtype=np.float64)

    axes = [masses[k] for k in range(masses.shape[0])]
    if score is not None and n_score_bins >= 2:
        axes.append(np.asarray(score, dtype=np.float64))
        counts = [n_per_axis] * masses.shape[0] + [n_score_bins]
    else:
        counts = [n_per_axis] * masses.shape[0]

    index = np.zeros(len(is_signal), dtype=np.int64)
    for x, n in zip(axes, counts):
        edges = _signal_quantile_edges(x, is_signal, weight, n)
        idx = np.searchsorted(edges, x, side="right")
        idx = np.where(np.isfinite(x), idx, len(edges) + 1)          # NaN -> its own bin
        index = index * (len(edges) + 2) + idx
    # compact the (sparse) cell ids
    cell_ids, inverse = np.unique(index, return_inverse=True)
    ncell = len(cell_ids)

    s = np.bincount(inverse[is_signal], weights=weight[is_signal], minlength=ncell)
    b = np.bincount(inverse[~is_signal], weights=weight[~is_signal], minlength=ncell)
    b_raw = np.bincount(inverse[~is_signal], minlength=ncell)
    b_sq = np.bincount(inverse[~is_signal], weights=weight[~is_signal] ** 2, minlength=ncell)

    with np.errstate(divide="ignore", invalid="ignore"):
        b_eff = np.where(b_sq > 0, b * b / b_sq, 0.0)
    good = (b_raw >= min_b_raw) & (b > 0) & (b_eff >= min_b_eff)
    if not good.any():
        return float("nan"), 0

    ks, kb = list(s[good]), list(b[good])
    rs, rb, rraw, rsq = s[~good].sum(), b[~good].sum(), b_raw[~good].sum(), b_sq[~good].sum()
    if (~good).any():
        r_eff = rb * rb / rsq if rsq > 0 else 0.0
        if rraw >= min_b_raw and rb > 0 and r_eff >= min_b_eff:
            ks.append(rs)
            kb.append(rb)
        else:                                                         # fold the remainder into the biggest-background cell
            j = int(np.argmax(kb))
            ks[j] += rs
            kb[j] += rb
    ks, kb = np.asarray(ks), np.asarray(kb)
    ok = kb > 0
    if not ok.any():
        return float("nan"), 0
    return float(np.sqrt(np.sum(ks[ok] ** 2 / kb[ok]))), int(ok.sum())


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
    extra_bin_counts: Sequence[int] = DEFAULT_EXTRA_BIN_COUNTS,
    min_b_eff: float = 10.0,
    eff_bin_counts: Sequence[int] = DEFAULT_EFF_BIN_COUNTS,
    masses: Optional[Dict[str, np.ndarray]] = None,
    mass_grid_counts: Sequence[int] = DEFAULT_MASS_GRID_COUNTS,
    mass_cross_eff: Sequence[int] = DEFAULT_MASS_CROSS_EFF,
) -> Dict[str, float]:
    is_signal = np.asarray(is_signal, dtype=bool)
    weight = np.asarray(weight, dtype=np.float64)

    metrics = {}
    # Method 3: bins in the predicted masses (m_Hbb, m_W1 from the jets SPANet assigned), optionally crossed with score bins.
    if masses:
        mass_matrix = np.vstack([np.asarray(v, dtype=np.float64) for v in masses.values()])
        cross_score = scores.get("joint", next(iter(scores.values()), None)) if scores else None
        for n in mass_grid_counts:
            zg, nbg = mass_binned_significance(mass_matrix, is_signal, weight, int(n), None, 0, min_b_raw, min_b_eff)
            metrics[f"signif/mass/binned_Z_g{int(n)}"] = zg
            metrics[f"signif/mass/binned_Z_g{int(n)}_nb"] = float(nbg)
            if cross_score is not None:
                for k in mass_cross_eff:
                    zx, nbx = mass_binned_significance(mass_matrix, is_signal, weight, int(n), cross_score, int(k),
                                                       min_b_raw, min_b_eff)
                    metrics[f"signif/mass/binned_Z_g{int(n)}_e{int(k)}"] = zx
                    metrics[f"signif/mass/binned_Z_g{int(n)}_e{int(k)}_nb"] = float(nbx)
    for name, score in scores.items():
        score = np.asarray(score, dtype=np.float64)
        # The default metric (checkpoint monitor / scan objective) is unchanged.
        metrics[f"signif/{name}/binned_Z"] = binned_significance(score, is_signal, weight, bin_edges, min_b_raw)
        # Finer uniform binnings, logged for comparison only:
        #   _u{N}      N uniform bins, sparse bins dropped (as the default metric does); _u{N}_cov = signal fraction kept
        #   _m{N}      N uniform fine bins, sparse bins merged from the top until raw AND effective background counts are
        #              large enough (min_b_raw, min_b_eff); _m{N}_nb = number of merged bins
        for n in extra_bin_counts:
            z, cov = binned_significance_with_coverage(score, is_signal, weight, np.linspace(0.0, 1.0, int(n) + 1), min_b_raw)
            metrics[f"signif/{name}/binned_Z_u{int(n)}"] = z
            metrics[f"signif/{name}/binned_Z_u{int(n)}_cov"] = cov
            zm, nb = merged_significance(score, is_signal, weight, int(n), min_b_raw, min_b_eff)
            metrics[f"signif/{name}/binned_Z_m{int(n)}"] = zm
            metrics[f"signif/{name}/binned_Z_m{int(n)}_nb"] = float(nb)
        # Equal-signal-efficiency bins (each holds 1/N of the signal), sparse bins merged; _e{N}_nb = bins actually used.
        for n in eff_bin_counts:
            ze, nbe = efficiency_binned_significance(score, is_signal, weight, int(n), min_b_raw, min_b_eff)
            metrics[f"signif/{name}/binned_Z_e{int(n)}"] = ze
            metrics[f"signif/{name}/binned_Z_e{int(n)}_nb"] = float(nbe)
        for eff in efficiencies:
            metrics[f"signif/{name}/s_over_sqrtb_eff{int(round(eff * 100))}"] = \
                s_over_sqrt_b_at_efficiency(score, is_signal, weight, eff, min_b_raw)
    return metrics