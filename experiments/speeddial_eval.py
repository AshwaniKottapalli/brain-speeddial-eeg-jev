"""Brain speed-dial evaluation on public EEGMMIDB (10 subjects), imagery commands + REST.

Simulates the real product loop:
  1. calibration: N_CAL trials per command (+ rest) from the user  -> train per-user decoder
  2. live: a continuous stream of held-out command / rest segments; 2 s sliding window every 0.5 s
  3. decision: fire a command only when the calibrated probability exceeds tau for K consecutive windows
Metrics: command detection rate, false triggers per hour of idle, latency.
Usage: speeddial_eval.py <n_commands 2|3|4> <channels all|muse> [n_cal_per_class]
"""
import sys, json, numpy as np
sys.path.insert(0, ".")
from scipy.signal import butter, sosfiltfilt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

n_cmd = int(sys.argv[1]); chans = sys.argv[2]; n_cal = int(sys.argv[3]) if len(sys.argv) > 3 else 15
d = np.load("cache/speeddial_eegbci.npz", allow_pickle=True)
X, y, g, ch, sf = d["X"], d["y"], d["g"], list(d["ch"]), float(d["sfreq"])
CMDS = ["left_hand", "right_hand", "feet", "hands"][:n_cmd]
ACTIONS = {"left_hand": "book_uber_home", "right_hand": "order_favourite_food", "feet": "call_partner", "hands": "play_music"}
if chans == "muse":
    pick = [ch.index(c) for c in ["AF7", "AF8", "TP7", "TP8"]]; X = X[:, pick]
sos = butter(4, [8, 30], btype="band", fs=sf, output="sos")
X = sosfiltfilt(sos, X, axis=-1)
WIN = int(2.0 * sf); HOP = int(0.5 * sf); TRIAL = X.shape[-1]  # 4 s trials


def windows(x):  # x: (ch, T) -> (n_win, ch, WIN)
    return np.stack([x[:, i:i + WIN] for i in range(0, x.shape[-1] - WIN + 1, HOP)])


def make_clf():
    return make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=3000, C=0.5))

import os
REPEATS = int(os.environ.get("REPEATS", "3"))
per_subject = []
for rep in range(REPEATS):
    rng = np.random.RandomState(rep)
    for s in np.unique(g):
        ids = np.where(g == s)[0]
        keep = [i for i in ids if y[i] in CMDS + ["rest"]]
        cal, test = [], []
        for c in CMDS + ["rest"]:
            cc = rng.permutation([i for i in keep if y[i] == c])
            n = min(n_cal, len(cc) - 4) if c != "rest" else n_cal * 2
            cal += list(cc[:n]); test += list(cc[n:])
        Xtr = np.concatenate([windows(X[i]) for i in cal]); ytr = np.concatenate([[y[i]] * len(windows(X[i])) for i in cal])
        clf = make_clf().fit(Xtr, ytr); classes = list(clf.classes_)
        test = rng.permutation(test)
        stream = np.concatenate([X[i] for i in test], axis=-1); labels = np.concatenate([[y[i]] * TRIAL for i in test])
        W = np.stack([stream[:, i:i + WIN] for i in range(0, stream.shape[-1] - WIN + 1, HOP)])
        P = clf.predict_proba(W)
        t_end = np.arange(len(W)) * HOP + WIN
        per_subject.append({"subject": int(s) + 100 * rep, "P": P, "t_end": t_end, "labels": labels, "classes": classes})

# simpler exact scoring
def score(tau, K):
    tot_cmd, correct, wrong_cmd, fa, idle_s, lats = 0, 0, 0, 0, 0.0, []
    for r in per_subject:
        P, t_end, labels, classes = r["P"], r["t_end"], r["labels"], r["classes"]
        fires = []  # (sample, class)
        streak = 0; last = None
        for w in range(len(P)):
            j = P[w].argmax(); c = classes[j]
            if c != "rest" and P[w, j] >= tau:
                streak = streak + 1 if c == last else 1; last = c
            else:
                streak = 0; last = None
            if streak == K:
                fires.append((t_end[w] - 1, c)); streak = 0; last = None
        fires = np.array(fires, dtype=object)
        for k in range(0, len(labels), TRIAL):
            truth = labels[k]
            f = [c for (t, c) in fires if k <= t < k + TRIAL]
            if truth in CMDS:
                tot_cmd += 1
                if truth in f:
                    correct += 1; t0 = [t for (t, c) in fires if k <= t < k + TRIAL and c == truth][0]; lats.append((t0 - k) / sf)
                elif f: wrong_cmd += 1
            else:
                idle_s += TRIAL / sf; fa += len(f)
    return {"tau": tau, "K": K, "detect_rate": correct / tot_cmd, "wrong_command_rate": wrong_cmd / tot_cmd,
            "false_triggers_per_hour_idle": fa / idle_s * 3600, "median_latency_s": float(np.median(lats)) if lats else None,
            "n_cmd_trials": tot_cmd, "idle_hours": idle_s / 3600}

print(f"repeats={REPEATS} | EEGMMIDB | {n_cmd} commands + rest | channels={chans} ({X.shape[1]}) | calibration={n_cal} trials/command "
      f"(~{n_cal*4*(n_cmd+2)/60:.1f} min of recording) | 10 subjects")
print(f"{'tau':>5} {'K':>2} {'detect%':>8} {'wrongCmd%':>10} {'falseTrig/h':>12} {'latency s':>10}")
grid = []
for K in (1, 2, 3):
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        r = score(tau, K); grid.append(r)
        print(f"{tau:5.2f} {K:2d} {100*r['detect_rate']:8.1f} {100*r['wrong_command_rate']:10.1f} {r['false_triggers_per_hour_idle']:12.1f} {r['median_latency_s'] if r['median_latency_s'] else float('nan'):10.2f}")
# per-subject window accuracy for reference
accs = []
for r in per_subject:
    P, labels, classes, t_end = r["P"], r["labels"], r["classes"], r["t_end"]
    pred = np.array(classes)[P.argmax(1)]; truth = labels[np.minimum(t_end - 1, len(labels) - 1)]
    accs.append((pred == truth).mean())
print(f"per-window accuracy ({n_cmd+1} classes incl. rest, chance {1/(n_cmd+1):.2f}): mean={np.mean(accs):.3f} per-subject={np.round(accs,2).tolist()}")
json.dump({"grid": grid, "window_acc": accs, "n_cmd": n_cmd, "chans": chans, "n_cal": n_cal,
           "per_subject_P": [{"subject": r["subject"], "P": r["P"].tolist(), "t_end": r["t_end"].tolist(),
                              "labels": r["labels"].tolist(), "classes": r["classes"]} for r in per_subject]},
          open(f"results/speeddial_eegbci_{n_cmd}cmd_{chans}_cal{n_cal}.json", "w"))
