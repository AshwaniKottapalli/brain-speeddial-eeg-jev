"""Apples-to-apples channel ablation for the mental-arithmetic speed-dial (Shin 2017B):
full 30-channel EEG vs Muse-like 4-channel proxy. IDENTICAL preprocessing, folds, windows, model, trigger logic
and evaluation; the only difference is which channels are handed to the covariance estimator.

Trigger logic: P(task) >= tau on K consecutive 3 s windows (1 s hop); after a fire the streak resets, so several
fires per 10 s trial are possible and are all counted for the false-trigger rate.
Usage: channel_ablation.py <subjects...>
"""
import sys, os, json, numpy as np, mne
sys.path.insert(0, ".")
mne.set_log_level("ERROR")
from scipy.signal import butter, sosfiltfilt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

subjects = [int(s) for s in sys.argv[1:]] or list(range(1, 11))
SF = 128; WIN, HOP = 3 * SF, 1 * SF; TRIAL_S = 10.0
MUSE = ["AFF5h", "AFF6h", "P7", "P8"]
TAUS = np.round(np.arange(0.50, 0.96, 0.05), 2); KS = (1, 2, 3)


def load_subject(s):
    fn = f"cache/shin_s{s:02d}.npz"
    if os.path.exists(fn):
        d = np.load(fn, allow_pickle=True); return d["X"], d["y"], list(d["ch"])
    from moabb.datasets import Shin2017B
    data = Shin2017B(accept=True).get_data(subjects=[s])[s]
    Xs, ys = [], []
    for sess in data.values():
        for raw in sess.values():
            raw = raw.copy().load_data().pick("eeg"); raw.filter(1, 40, verbose=False)
            ev, ev_id = mne.events_from_annotations(raw, verbose=False)
            ep = mne.Epochs(raw, ev, ev_id, tmin=0.0, tmax=TRIAL_S, baseline=None, preload=True, verbose=False)
            ep.resample(SF, verbose=False)
            inv = {v: k for k, v in ev_id.items()}
            Xs.append(ep.get_data(copy=True)); ys.append(np.array([inv[e] for e in ep.events[:, 2]])); ch = ep.ch_names
    X = np.concatenate(Xs).astype(np.float32); y = np.concatenate(ys)
    np.savez_compressed(fn, X=X, y=y, ch=np.array(ch)); return X, y, ch


def windows(x):
    return np.stack([x[:, i:i + WIN] for i in range(0, x.shape[-1] - WIN + 1, HOP)])


def make_clf():
    return make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=3000, C=0.5))


def fires_for(P, tau, K):
    """indices of windows at which a trigger fires (streak of K windows >= tau, reset after fire)."""
    out, streak = [], 0
    for w, p in enumerate(P):
        streak = streak + 1 if p >= tau else 0
        if streak == K: out.append(w); streak = 0
    return out


def evaluate(Ps, is_task, tau, K):
    det, lat, fa_events, n_rest = 0, [], 0, 0
    for P, t in zip(Ps, is_task):
        f = fires_for(P, tau, K)
        if t:
            if f: det += 1; lat.append(WIN / SF + f[0] * HOP / SF)   # seconds after command onset
        else:
            n_rest += 1; fa_events += len(f)
    return {"detect": det / is_task.sum(), "fa_per_hour": fa_events / (n_rest * TRIAL_S) * 3600,
            "latency_median": float(np.median(lat)) if lat else None}


# ---------------------------------------------------------------- run identical pipeline for both channel sets
per_subject = {}
for s in subjects:
    X, y, ch = load_subject(s)
    sos = butter(4, [4, 30], btype="band", fs=SF, output="sos"); X = sosfiltfilt(sos, X, axis=-1)
    task = [c for c in np.unique(y) if c != "rest"][0]; is_task = y == task
    pidx = [ch.index(c) for c in MUSE]
    sets = {"all30": np.arange(X.shape[1]), "muse4": np.array(pidx)}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))   # SAME folds for both sets
    Ps = {k: [None] * len(y) for k in sets}
    for tr, te in folds:
        Xtr_w = np.concatenate([windows(X[i]) for i in tr]); ytr_w = np.concatenate([[y[i]] * len(windows(X[i])) for i in tr])
        for name, idx in sets.items():
            clf = make_clf().fit(Xtr_w[:, idx], ytr_w); ti = list(clf.classes_).index(task)
            for i in te:
                Ps[name][i] = clf.predict_proba(windows(X[i])[:, idx])[:, ti]
    per_subject[s] = {"is_task": is_task, "Ps": Ps}
    aucs = {k: roc_auc_score(np.repeat(is_task, len(Ps[k][0])), np.concatenate(Ps[k])) for k in sets}
    print(f"sub {s:2d}: trials={len(y)} | window-level AUC all30={aucs['all30']:.3f} muse4={aucs['muse4']:.3f}", flush=True)
    per_subject[s]["auc"] = aucs

# ---------------------------------------------------------------- curves (pooled over subjects, all folds)
curves = {}
for name in ("all30", "muse4"):
    for K in KS:
        pts = []
        for tau in TAUS:
            Ps = sum([per_subject[s]["Ps"][name] for s in subjects], []); it = np.concatenate([per_subject[s]["is_task"] for s in subjects])
            r = evaluate(Ps, it, tau, K); r["tau"] = float(tau); r["K"] = K; pts.append(r)
        curves[f"{name}_K{K}"] = pts

print("\n=== Detection vs false triggers (pooled over subjects) ===")
print(f"{'tau':>5} {'K':>2} | {'30ch detect':>11} {'FA/h':>6} {'lat s':>6} | {'Muse4 detect':>12} {'FA/h':>6} {'lat s':>6} | {'Δdetect':>8}")
for K in KS:
    for a, m in zip(curves[f"all30_K{K}"], curves[f"muse4_K{K}"]):
        la = a["latency_median"] or float("nan"); lm = m["latency_median"] or float("nan")
        print(f"{a['tau']:5.2f} {K:2d} | {100*a['detect']:10.1f}% {a['fa_per_hour']:6.1f} {la:6.1f} | {100*m['detect']:11.1f}% {m['fa_per_hour']:6.1f} {lm:6.1f} | {100*(a['detect']-m['detect']):+7.1f}")

# ---------------------------------------------------------------- matched false-trigger budgets
print("\n=== Best achievable detection at a false-trigger budget (choose tau,K per channel set) ===")
budgets = [60, 30, 12, 6, 2]
matched = {}
for b in budgets:
    row = {}
    for name in ("all30", "muse4"):
        best = max((p for K in KS for p in curves[f"{name}_K{K}"] if p["fa_per_hour"] <= b), key=lambda p: p["detect"], default=None)
        row[name] = best
    matched[b] = row
    fa = lambda p: f"{100*p['detect']:5.1f}% (tau={p['tau']:.2f},K={p['K']}, lat {p['latency_median']:.0f}s)" if p else "  n/a"
    if row["all30"] and row["muse4"]:
        print(f"FA/h <= {b:3d}: 30ch {fa(row['all30'])} | Muse4 {fa(row['muse4'])} | loss {100*(row['all30']['detect']-row['muse4']['detect']):+.1f} pts")
    else:
        print(f"FA/h <= {b:3d}: 30ch {fa(row['all30'])} | Muse4 {fa(row['muse4'])}")

# ---------------------------------------------------------------- per-subject at fixed operating points
print("\n=== Per subject at fixed operating points ===")
ops = [(0.7, 2), (0.7, 3), (0.8, 2)]
hdr = " | ".join([f"tau={t},K={k}: 30ch det/FAh  Muse4 det/FAh" for t, k in ops])
print(f"{'sub':>3} {'AUC30':>6} {'AUC4':>6} | {hdr}")
per_sub_rows = []
for s in subjects:
    d = per_subject[s]; cells = []; rec = {"subject": s, "auc_all30": d["auc"]["all30"], "auc_muse4": d["auc"]["muse4"]}
    for t, k in ops:
        a = evaluate(d["Ps"]["all30"], d["is_task"], t, k); m = evaluate(d["Ps"]["muse4"], d["is_task"], t, k)
        cells.append(f"{100*a['detect']:5.0f}%/{a['fa_per_hour']:4.0f}   {100*m['detect']:5.0f}%/{m['fa_per_hour']:4.0f}")
        rec[f"all30_t{t}_K{k}"] = a; rec[f"muse4_t{t}_K{k}"] = m
    per_sub_rows.append(rec)
    print(f"{s:3d} {d['auc']['all30']:6.3f} {d['auc']['muse4']:6.3f} | " + " | ".join(cells))
dauc = np.array([r["auc_all30"] - r["auc_muse4"] for r in per_sub_rows])
ddet = np.array([r["all30_t0.7_K2"]["detect"] - r["muse4_t0.7_K2"]["detect"] for r in per_sub_rows])
dfa = np.array([r["all30_t0.7_K2"]["fa_per_hour"] - r["muse4_t0.7_K2"]["fa_per_hour"] for r in per_sub_rows])
print(f"\nPaired 30ch minus Muse4 over {len(subjects)} subjects: AUC {dauc.mean():+.3f} ± {dauc.std():.3f} (30ch better in {(dauc>0).sum()}/{len(dauc)}); "
      f"detect@0.7/K2 {100*ddet.mean():+.1f} ± {100*ddet.std():.1f} pts (better in {(ddet>0).sum()}); FA/h@0.7/K2 {dfa.mean():+.1f} ± {dfa.std():.1f}")

# ---------------------------------------------------------------- plot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
COL = {"all30": "#2a78d6", "muse4": "#eb6834"}; LBL = {"all30": "30 channels", "muse4": "Muse proxy (4 ch)"}
for ax, K in zip(axes, KS):
    for name in ("all30", "muse4"):
        pts = curves[f"{name}_K{K}"]
        ax.plot([p["fa_per_hour"] for p in pts], [100 * p["detect"] for p in pts], "-o", color=COL[name], lw=2, ms=5, label=LBL[name])
        for p in pts:
            if p["tau"] in (0.5, 0.7, 0.9): ax.annotate(f"τ={p['tau']:.1f}", (p["fa_per_hour"], 100 * p["detect"]), fontsize=7, color="#555", xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=5); ax.set_xlabel("false triggers per idle hour"); ax.set_title(f"K = {K} consecutive windows", fontsize=10)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("command detection rate (%)"); axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Mental-arithmetic speed-dial, Shin 2017B: 30-channel EEG vs Muse-like 4-channel proxy (identical pipeline)", fontsize=11)
fig.tight_layout(); fig.savefig("results/channel_ablation.png", dpi=150)
json.dump({"subjects": subjects, "curves": curves, "matched_budget": matched, "per_subject": per_sub_rows},
          open("results/channel_ablation.json", "w"), indent=1)
print("\nsaved results/channel_ablation.json and results/channel_ablation.png")
