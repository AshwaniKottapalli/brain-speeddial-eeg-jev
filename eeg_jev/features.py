"""Turn an EEG epoch into a compact, semantically meaningful JSON descriptor Jev can read.

Physiology we lean on (motor imagery):
- Mu (8-13 Hz) and beta (13-30 Hz) power DEcreases (ERD) over the sensorimotor cortex contralateral to
  the imagined limb. C3 = left hemisphere (right hand), C4 = right hemisphere (left hand),
  Cz = midline (feet / both hands). Tongue: bilateral, more ventral.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import welch

BANDS = {"theta": (4, 8), "mu": (8, 13), "beta_low": (13, 20), "beta_high": (20, 30)}
KEY_CH = ["C3", "Cz", "C4"]
EXTRA_CH = ["FC3", "FC4", "CP3", "CP4", "C1", "C2"]


def _bandpower(seg, sfreq):
    f, p = welch(seg, fs=sfreq, nperseg=min(seg.shape[-1], int(sfreq)), axis=-1)
    out = {}
    for b, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        out[b] = p[..., m].mean(-1)
    return out


def epoch_features(x, ch_names, sfreq, tmin, task=(0.5, 3.5), base=(-1.0, 0.0)):
    """x: (n_ch, n_t). Returns dict of interpretable features (ERD in dB relative to baseline)."""
    t0 = lambda t: int(round((t - tmin) * sfreq))
    idx = {c: i for i, c in enumerate(ch_names)}
    chans = [c for c in KEY_CH + EXTRA_CH if c in idx]
    xt = x[:, t0(task[0]):t0(task[1])]
    xb = x[:, t0(base[0]):t0(base[1])]
    pt = _bandpower(xt, sfreq); pb = _bandpower(xb, sfreq)
    feats = {}
    for c in chans:
        i = idx[c]
        feats[c] = {b: float(10 * np.log10(pt[b][i] / pb[b][i])) for b in BANDS}
    # laterality: negative -> stronger ERD (more suppression) on left hemisphere (C3) => right-hand-like
    def lat(b):
        if "C3" in idx and "C4" in idx:
            return float(feats["C3"][b] - feats["C4"][b])
        return 0.0
    feats["laterality_C3_minus_C4"] = {b: lat(b) for b in ("mu", "beta_low")}
    if "Cz" in idx and "C3" in idx and "C4" in idx:
        feats["midline_vs_lateral_Cz_minus_mean(C3,C4)"] = {
            b: float(feats["Cz"][b] - 0.5 * (feats["C3"][b] + feats["C4"][b])) for b in ("mu", "beta_low")}
    # time course of mu ERD at C3/C4/Cz in 3 windows
    tc = {}
    for c in KEY_CH:
        if c not in idx: continue
        vals = []
        for (a, b_) in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)):
            seg = x[idx[c], t0(a):t0(b_)]
            p = _bandpower(seg[None], sfreq)["mu"][0]
            vals.append(float(10 * np.log10(p / pb["mu"][idx[c]])))
        tc[c] = vals
    feats["mu_ERD_dB_over_time_0-1s_1-2s_2-3s"] = tc
    # absolute log10 band power (uV^2/Hz) for every channel: carries the most decodable information
    feats["abs_log10_power_all_channels"] = {
        c: {b: float(np.log10(pt[b][idx[c]] * 1e12)) for b in BANDS} for c in ch_names}
    return feats


def round_feats(f, nd=1):
    if isinstance(f, dict):
        return {k: round_feats(v, nd) for k, v in f.items()}
    if isinstance(f, list):
        return [round_feats(v, nd) for v in f]
    if isinstance(f, float):
        return round(f, nd)
    return f


def feature_vector(feats):
    """Flatten the JSON features into a numeric vector (for the local baseline on the same info Jev sees)."""
    v = []
    def rec(o):
        if isinstance(o, dict):
            for k in sorted(o): rec(o[k])
        elif isinstance(o, list):
            for e in o: rec(e)
        else:
            v.append(float(o))
    rec(feats)
    return np.array(v, dtype=np.float32)
