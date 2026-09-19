"""Imagined-word ("inner speech") decoding feasibility on Nieto et al. 2022 (OpenNeuro ds003626).
4 words: up / down / right / left, inner-speech condition. Within-subject CV, all 128 ch and a Muse-like 4-ch proxy.
Also: the 'speed-dial' framing -> 1 word vs the other 3 + confidence gating.
"""
import sys, glob, numpy as np, mne
sys.path.insert(0, ".")
mne.set_log_level("ERROR")
from scipy.signal import butter, sosfiltfilt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

WORDS = {0: "up", 1: "down", 2: "right", 3: "left"}
subjects = sys.argv[1:] or ["01"]

def muse_proxy_picks(info):
    std = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    targets = {k: std[k] for k in ["AF7", "AF8", "TP9", "TP10"]}
    pos = {ch["ch_name"]: ch["loc"][:3] for ch in info["chs"]}
    names = [n for n in pos if np.isfinite(pos[n]).all() and np.abs(pos[n]).sum() > 0]
    picks = []
    for t, tp in targets.items():
        d = {n: np.linalg.norm(pos[n] - tp) for n in names}
        best = min(d, key=d.get); picks.append(best)
    return picks

for s in subjects:
    Xs, ys = [], []
    for ses in ["01", "02", "03"]:
        fe = f"data/innerspeech/sub-{s}_ses-{ses}_eeg-epo.fif"; fv = f"data/innerspeech/sub-{s}_ses-{ses}_events.dat"
        try:
            ep = mne.read_epochs(fe, preload=True, verbose=False)
            ep.set_montage(mne.channels.make_standard_montage("biosemi128"), on_missing="ignore")
            ev = np.load(fv, allow_pickle=True)
        except Exception as e:
            print("skip", fe, e); continue
        ev = np.asarray(ev)
        cls, cond = ev[:, 1].astype(int), ev[:, 2].astype(int)   # columns: [sample, class, condition, session]
        m = cond == 1  # inner speech
        Xs.append(ep.get_data(copy=True)[m]); ys.append(cls[m]); info = ep.info; sf = ep.info["sfreq"]; tmin = ep.tmin
    if not Xs: continue
    X = np.concatenate(Xs); y = np.array([WORDS[c] for c in np.concatenate(ys)])
    # action window 1.0-3.5 s after cue (per dataset paper)
    a, b = int((1.0 - tmin) * sf), int((3.5 - tmin) * sf)
    X = X[:, :, a:b]
    print(f"\nsub-{s}: inner-speech trials {X.shape} sfreq={sf} | {dict(zip(*np.unique(y, return_counts=True)))}")
    picks = muse_proxy_picks(info); print("Muse proxy channels (nearest to AF7, AF8, TP9, TP10):", picks)
    pidx = [info["ch_names"].index(p) for p in picks]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for band_name, (lo, hi) in {"broad 4-40": (4, 40), "theta+alpha 4-13": (4, 13), "beta+gamma 13-40": (13, 40)}.items():
        sos = butter(4, [lo, hi], btype="band", fs=sf, output="sos"); Xf = sosfiltfilt(sos, X, axis=-1)
        for chn, Xc in (("128ch", Xf), ("muse4", Xf[:, pidx])):
            pipe = make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=3000, C=0.1))
            pred = cross_val_predict(pipe, Xc, y, cv=cv)
            print(f"  {band_name:18s} {chn:6s} Riemannian 4-word acc = {(pred == y).mean():.3f} (chance 0.25)")
    # speed-dial framing: one target word vs everything else, with confidence gating
    sos = butter(4, [4, 40], btype="band", fs=sf, output="sos"); Xf = sosfiltfilt(sos, X, axis=-1)
    for target in ["up", "left"]:
        yb = np.where(y == target, target, "other")
        pipe = make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"))
        proba = cross_val_predict(pipe, Xf, yb, cv=cv, method="predict_proba"); pt = proba[:, list(np.unique(yb)).index(target)]
        for tau in (0.5, 0.7, 0.8):
            fire = pt >= tau; tp = (fire & (yb == target)).sum(); fp = (fire & (yb != target)).sum()
            print(f"  speed-dial '{target}' vs rest, tau={tau}: detect {tp/(yb==target).sum():.2f}, false-fire rate on other words {fp/(yb!=target).sum():.2f}")
