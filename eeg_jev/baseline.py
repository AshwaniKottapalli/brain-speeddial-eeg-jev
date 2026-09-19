"""Reference classifiers: CSP+LDA and Riemannian tangent space + LR on raw epochs, and LR on the exact
JSON features Jev sees (the information ceiling of the descriptor)."""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt


def crop_filter(X, sfreq, tmin, t0=0.5, t1=3.5, lo=8, hi=30):
    a = int(round((t0 - tmin) * sfreq)); b = int(round((t1 - tmin) * sfreq))
    sos = butter(4, [lo, hi], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X[:, :, a:b], axis=-1).astype(np.float64)


def within_subject_cv(X, y, g, sfreq, tmin, pipe_name="riemann", n_splits=5):
    accs = {}
    for s in np.unique(g):
        m = g == s
        Xs = crop_filter(X[m], sfreq, tmin); ys = y[m]
        if pipe_name == "csp":
            pipe = make_pipeline(CSP(n_components=6, log=True), LinearDiscriminantAnalysis())
        else:
            pipe = make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=2000, C=1.0))
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)
        accs[int(s)] = float(cross_val_score(pipe, Xs, ys, cv=cv).mean())
    return accs


def features_lr_cv(F, y, g, n_splits=5):
    accs = {}
    for s in np.unique(g):
        m = g == s
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)
        accs[int(s)] = float(cross_val_score(pipe, F[m], y[m], cv=cv).mean())
    return accs
