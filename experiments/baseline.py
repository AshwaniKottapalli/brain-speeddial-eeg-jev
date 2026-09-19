import sys, json, numpy as np
sys.path.insert(0, ".")
from eeg_jev.baseline import within_subject_cv, features_lr_cv
which = sys.argv[1] if len(sys.argv) > 1 else "eegbci"
name = {"eegbci": "eegbci_s1-10", "bnci": "bnci_s1-3"}[which]
d = np.load(f"cache/{name}.npz", allow_pickle=True)
X, y, g, sfreq, tmin = d["X"], d["y"], d["g"], float(d["sfreq"]), float(d["tmin"])
F = np.load(f"cache/{which}_F.npy")
res = {"chance": 1 / len(np.unique(y))}
res["riemann_raw"] = within_subject_cv(X, y, g, sfreq, tmin, "riemann")
res["csp_lda_raw"] = within_subject_cv(X, y, g, sfreq, tmin, "csp")
res["lr_on_jev_features"] = features_lr_cv(F, y, g)
for k, v in res.items():
    if isinstance(v, dict):
        print(f"{k:22s} mean={np.mean(list(v.values())):.3f}  per-subject={ {s: round(a, 2) for s, a in v.items()} }")
    else:
        print(f"{k:22s} {v:.3f}")
json.dump(res, open(f"results/baseline_{which}.json", "w"), indent=1)
