"""'Count backwards' as a speed-dial command: mental arithmetic vs rest (Shin 2017B, 29 subjects, 30 EEG ch, 10 s trials).
This is the most Muse-relevant public test: frontal task engagement, not motor cortex.
Within-subject CV on sliding windows; all channels vs Muse-like proxy (AFF5h/AFF6h ~ AF7/AF8, TP7/TP8 or P7/P8 ~ TP9/TP10).
"""
import sys, numpy as np, mne
sys.path.insert(0, ".")
mne.set_log_level("ERROR")
from moabb.datasets import Shin2017B
from scipy.signal import butter, sosfiltfilt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

subjects = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4, 5]
ds = Shin2017B(accept=True)
WIN_S = 3.0; HOP_S = 1.0
summary = []
for s in subjects:
    data = ds.get_data(subjects=[s])[s]
    Xs, ys = [], []
    for sess in data.values():
        for raw in sess.values():
            raw = raw.copy().load_data().pick("eeg")
            raw.filter(1, 40, verbose=False)
            ev, ev_id = mne.events_from_annotations(raw, verbose=False)
            ep = mne.Epochs(raw, ev, ev_id, tmin=0.0, tmax=10.0, baseline=None, preload=True, verbose=False)
            ep.resample(128, verbose=False)
            inv = {v: k for k, v in ev_id.items()}
            Xs.append(ep.get_data(copy=True)); ys.append(np.array([inv[e] for e in ep.events[:, 2]])); ch = ep.ch_names; sf = ep.info["sfreq"]
    X = np.concatenate(Xs); y = np.concatenate(ys)
    if s == subjects[0]: print("channels:", ch); print("classes:", dict(zip(*np.unique(y, return_counts=True))), "trial shape", X.shape)
    proxy = [c for c in ["AFF5h", "AFF6h", "TP7", "TP8", "P7", "P8"] if c in ch][:4]
    pidx = [ch.index(c) for c in proxy]
    sos = butter(4, [4, 30], btype="band", fs=sf, output="sos"); X = sosfiltfilt(sos, X, axis=-1)
    WIN, HOP = int(WIN_S * sf), int(HOP_S * sf)
    def win(x): return np.stack([x[:, i:i + WIN] for i in range(0, x.shape[-1] - WIN + 1, HOP)])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    res = {}
    for name, Xc in (("all30", X), ("muse4", X[:, pidx])):
        accs_w, det, fa, fa3, det3 = [], [], [], [], []
        for tr, te in cv.split(Xc, y):
            Xtr = np.concatenate([win(Xc[i]) for i in tr]); ytr = np.concatenate([[y[i]] * len(win(Xc[i])) for i in tr])
            clf = make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=3000, C=0.5)).fit(Xtr, ytr)
            task = [c for c in clf.classes_ if c != "rest"][0] if "rest" in clf.classes_ else clf.classes_[0]
            ti = list(clf.classes_).index(task)
            for i in te:
                P = clf.predict_proba(win(Xc[i]))[:, ti]
                accs_w.append(((P >= 0.5) == (y[i] == task)).mean())
                # speed-dial: fire if 2 consecutive windows >= 0.7
                fires = np.any((P[:-1] >= 0.7) & (P[1:] >= 0.7))
                fires3 = np.any((P[:-2] >= 0.7) & (P[1:-1] >= 0.7) & (P[2:] >= 0.7))
                (det if y[i] == task else fa).append(fires); (det3 if y[i] == task else fa3).append(fires3)
        res[name] = (np.mean(accs_w), np.mean(det), np.mean(fa), np.mean(det3), np.mean(fa3))
    print(f"sub {s:2d}: all30 acc={res['all30'][0]:.2f} K2: detect={res['all30'][1]:.2f} FA/h={res['all30'][2]*360:.0f} K3: detect={res['all30'][3]:.2f} FA/h={res['all30'][4]*360:.0f} | "
          f"muse4 acc={res['muse4'][0]:.2f} K2: detect={res['muse4'][1]:.2f} FA/h={res['muse4'][2]*360:.0f} K3: detect={res['muse4'][3]:.2f} FA/h={res['muse4'][4]*360:.0f}")
    summary.append((res["all30"], res["muse4"]))
a = np.array([r[0] for r in summary]); m = np.array([r[1] for r in summary])
print(f"\nMEAN over {len(subjects)} subjects (FA/h = false triggers per hour of idle; K = consecutive 3 s windows >= 0.7)")
print(f"  all30 : acc={a[:,0].mean():.2f} | K2 detect={a[:,1].mean():.2f} FA/h={a[:,2].mean()*360:.0f} | K3 detect={a[:,3].mean():.2f} FA/h={a[:,4].mean()*360:.0f}")
print(f"  muse4 : acc={m[:,0].mean():.2f} | K2 detect={m[:,1].mean():.2f} FA/h={m[:,2].mean()*360:.0f} | K3 detect={m[:,3].mean():.2f} FA/h={m[:,4].mean()*360:.0f}")
import json; json.dump({"subjects": subjects, "all30": a.tolist(), "muse4": m.tolist()}, open(f"results/mental_arith_shin2017b_s{subjects[0]}-{subjects[-1]}.json", "w"))
