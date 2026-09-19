"""Decoders exactly as validated in experiments/two_layer.py and channel_ablation.py (no new training recipe).
Layer 1: mental arithmetic vs rest (Shin 2017B), 3 s windows.  Layer 2 (imagery confirm): yes/no/idle (Shin 2017A + B rest), 4 s crops."""
from __future__ import annotations
import os, numpy as np, joblib
from scipy.signal import butter, sosfiltfilt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

SF = 128; WIN_S, HOP_S = 3.0, 1.0; WIN, HOP = int(WIN_S * SF), int(HOP_S * SF); TRIAL_S = 10.0
CONFIRM_S = 4.0; DW = int(CONFIRM_S * SF)
_SOS = butter(4, [4, 30], btype="band", fs=SF, output="sos")


def load_subject(s: int):
    B = np.load(f"cache/shinB_s{s:02d}.npz", allow_pickle=True); A = np.load(f"cache/shinA_s{s:02d}.npz", allow_pickle=True)
    XB = sosfiltfilt(_SOS, B["X"].astype(np.float64), axis=-1); XA = sosfiltfilt(_SOS, A["X"].astype(np.float64), axis=-1)
    yB = np.array(B["y"]); yA = np.where(np.array(A["y"]) == "left_hand", "yes", "no")
    return XB, yB, XA, yA, list(B["ch"])


def windows(x, w=WIN, h=HOP):
    return np.stack([x[:, i:i + w] for i in range(0, x.shape[-1] - w + 1, h)])


def _clf():
    return make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=3000, C=0.5))


def folds(yB, yA):
    fB = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yB)), yB))
    fA = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yA)), yA))
    return fB, fA


class Decoders:
    def __init__(self, L1, L2, task_label, teB, teA):
        self.L1, self.L2, self.task, self.teB, self.teA = L1, L2, task_label, teB, teA
        self.ti = list(L1.classes_).index(task_label)

    def p_task(self, x_win):            # x_win: (ch, WIN)
        return float(self.L1.predict_proba(x_win[None])[0, self.ti])

    def confirm_imagery(self, x_crop, theta=0.6):   # x_crop: (ch, DW) -> yes | no | idle | undecided
        p = dict(zip(self.L2.classes_, self.L2.predict_proba(x_crop[None])[0]))
        if max(p, key=p.get) == "idle": return "idle", p
        py = p["yes"] / (p["yes"] + p["no"])
        return ("yes" if py >= theta else "no" if py <= 1 - theta else "undecided"), p


def fit_decoders(s: int, fold: int) -> Decoders:
    fn = f"cache/models/shin_s{s:02d}_fold{fold}.joblib"
    if os.path.exists(fn): return joblib.load(fn)
    XB, yB, XA, yA, _ = load_subject(s)
    fB, fA = folds(yB, yA); trB, teB = fB[fold]; trA, teA = fA[fold]
    task = [c for c in np.unique(yB) if c != "rest"][0]
    Xw = np.concatenate([windows(XB[i]) for i in trB]); yw = np.concatenate([[yB[i]] * 8 for i in trB])
    L1 = _clf().fit(Xw, yw)
    cropsA = np.concatenate([windows(XA[i], DW) for i in trA]); ycA = np.concatenate([[yA[i]] * windows(XA[i], DW).shape[0] for i in trA])
    rest_tr = [i for i in trB if yB[i] == "rest"]
    cropsR = np.concatenate([windows(XB[i], DW) for i in rest_tr]); ycR = np.array(["idle"] * len(cropsR))
    L2 = _clf().fit(np.concatenate([cropsA, cropsR]), np.concatenate([ycA, ycR]))
    d = Decoders(L1, L2, task, teB, teA); joblib.dump(d, fn); return d
