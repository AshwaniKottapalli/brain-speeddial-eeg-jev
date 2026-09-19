"""Experiment B: which public-dataset paradigm gives the strongest EEG YES/NO confirmation?

One protocol for every paradigm: within-subject 5-fold CV, a binary decision from T seconds of EEG, decision
accuracy, false-YES rate (user meant NO but system says YES), and (where the dataset has idle/rest) YES-on-idle.

Paradigms and how a yes/no question maps onto them:
  imagery   : left-hand imagery = YES, right-hand = NO.  Shin2017A (30 ch), BNCI2014_001 L/R (22 ch), Cho2017 (64 ch)
  p300      : two options flash alternately; the attended one evokes a P300. Score = mean(clf score of YES flashes)
              - mean(NO flashes) over r repetitions each. Target epochs play 'attended option', non-target 'other'.
              BNCI2014_009 / BNCI2014_008 / BNCI2015_003 (visual), Sosulski2019 (auditory oddball, eyes-free)
  ssvep     : YES and NO flicker at two frequencies; look at the one you want. Best pair of frequencies per dataset.
              Nakanishi2015, Kalunga2016 (has REST -> YES-on-idle measurable)
  cvep      : code-modulated VEP, Thielen2015: 2 codes.
  arithmetic: 'do mental arithmetic = YES, stay idle = NO' (Shin2017B) -- our repeat-the-command baseline.
"""
import sys, os, json, numpy as np, mne, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
from scipy.signal import butter, sosfiltfilt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
import moabb.datasets as D
from moabb.paradigms import P300, LeftRightImagery, SSVEP

which = sys.argv[1:] or ["imagery_shin", "arith_shin", "p300_009", "p300_008", "p300_015", "oddball_sos", "ssvep_nak", "ssvep_kal", "cvep_thielen", "imagery_bnci", "imagery_cho"]
RES = {}


def cv_scores(X, y, make, n=5):
    """returns held-out decision scores (higher = YES) aligned with y (bool YES)"""
    sc = np.zeros(len(y))
    for tr, te in StratifiedKFold(n, shuffle=True, random_state=0).split(X, y):
        clf = make().fit(X[tr], y[tr]); sc[te] = clf.decision_function(X[te]) if hasattr(clf, "decision_function") else clf.predict_proba(X[te])[:, 1]
    return sc


def report(name, per_subject, note=""):
    acc = np.array([r["acc"] for r in per_subject]); fy = np.array([r["false_yes"] for r in per_subject])
    RES[name] = {"per_subject": per_subject, "acc_mean": float(acc.mean()), "acc_min": float(acc.min()), "acc_max": float(acc.max()),
                 "false_yes_mean": float(fy.mean()), "n_subjects": len(per_subject), "decision_s": per_subject[0].get("decision_s"), "note": note,
                 "yes_on_idle": float(np.mean([r["yes_on_idle"] for r in per_subject if r.get("yes_on_idle") is not None])) if any(r.get("yes_on_idle") is not None for r in per_subject) else None}
    print(f"{name:14s} n={len(per_subject):2d} | yes/no acc {acc.mean():.3f} (min {acc.min():.2f}, max {acc.max():.2f}) | false-YES {fy.mean():.3f} | "
          f"decision {per_subject[0].get('decision_s')}s | yes-on-idle {RES[name]['yes_on_idle']} | {note}", flush=True)


# ---------------------------------------------------------------- imagery (Riemannian TS+LR on 4 s crop)
def imagery_eval(name, XA, yA_is_yes, sf, decision_s=4.0, idle=None):
    sos = butter(4, [4, 30], btype="band", fs=sf, output="sos"); X = sosfiltfilt(sos, X_crop(XA, sf, decision_s), axis=-1)
    make = lambda: make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=3000, C=0.5))
    sc = cv_scores(X, yA_is_yes, make); pred = sc > 0
    r = {"acc": float((pred == yA_is_yes).mean()), "false_yes": float(pred[~yA_is_yes].mean()), "decision_s": decision_s}
    if idle is not None:
        clf = make().fit(X, yA_is_yes); xi = sosfiltfilt(sos, X_crop(idle, sf, decision_s), axis=-1); r["yes_on_idle"] = float((clf.decision_function(xi) > 0).mean())
    return r


def X_crop(X, sf, secs, start=0.0):
    a = int(start * sf); return X[:, :, a:a + int(secs * sf)]


if "imagery_shin" in which or "arith_shin" in which:
    rows_im, rows_ar = [], []
    for s in range(1, 11):
        A = np.load(f"cache/shinA_s{s:02d}.npz", allow_pickle=True); B = np.load(f"cache/shinB_s{s:02d}.npz", allow_pickle=True)
        XA, yA = A["X"].astype(float), A["y"] == "left_hand"; XB, yB = B["X"].astype(float), B["y"] != "rest"
        rest = XB[~yB]
        if "imagery_shin" in which: rows_im.append({"subject": s, **imagery_eval("imagery_shin", XA, yA, 128, 4.0, idle=rest)})
        if "arith_shin" in which: rows_ar.append({"subject": s, **imagery_eval("arith_shin", XB, yB, 128, 3.0)})
    if rows_im: report("imagery_shin", rows_im, "left=YES right=NO, 30 ch, 4 s; yes-on-idle from rest trials")
    if rows_ar: report("arith_shin", rows_ar, "arithmetic=YES idle=NO, 30 ch, 3 s (asymmetric: NO = do nothing)")

if "imagery_bnci" in which:
    d = np.load("cache/bnci_s1-3.npz", allow_pickle=True); X, y, g = d["X"].astype(float), d["y"], d["g"]
    m = np.isin(y, ["left_hand", "right_hand"]); rows = []
    for s in np.unique(g):
        mm = m & (g == s); rows.append({"subject": int(s), **imagery_eval("imagery_bnci", X[mm][:, :, int(1.5 * 128):], y[mm] == "left_hand", 128, 4.0)})
    report("imagery_bnci", rows, "BCI IV 2a left=YES right=NO, 22 ch, 4 s")

if "imagery_cho" in which:
    try:
        ds = D.Cho2017(); par = LeftRightImagery(fmin=4, fmax=30, tmin=0, tmax=4, resample=128); rows = []
        for s in range(1, 11):
            X, y, _ = par.get_data(ds, subjects=[s]); rows.append({"subject": s, **imagery_eval("imagery_cho", X, y == "left_hand", 128, 4.0)})
        report("imagery_cho", rows, "Cho2017 left=YES right=NO, 64 ch, 4 s")
    except Exception as e: print("imagery_cho failed:", e)

# ---------------------------------------------------------------- P300 / oddball: two-option speller simulation
def p300_eval(name, ds, subjects, reps=(1, 2, 3, 5), soa=None, note=""):
    par = P300(fmin=1, fmax=20, tmin=0, tmax=0.8, resample=128)
    per = {r: [] for r in reps}
    for s in subjects:
        try: X, y, meta = par.get_data(ds, subjects=[s])
        except Exception as e: print(name, "subject", s, "failed", e); continue
        yt = (y == "Target")
        make = lambda: make_pipeline(XdawnCovariances(nfilter=4, estimator="oas"), TangentSpace(), LogisticRegression(max_iter=3000))
        sc = cv_scores(X, yt, make)                       # single-flash target-vs-nontarget score (higher = attended)
        rng = np.random.RandomState(0); T, NT = sc[yt], sc[~yt]
        for r in reps:
            # a yes/no question: YES option flashes r times, NO option flashes r times. If the user attends YES, YES flashes
            # are 'target' epochs and NO flashes are 'non-target'. Decision = option with the higher mean score.
            n = 2000; d_att = np.array([rng.choice(T, r).mean() - rng.choice(NT, r).mean() for _ in range(n)])   # user attends YES: score(YES)-score(NO)
            d_idle = np.array([rng.choice(NT, r).mean() - rng.choice(NT, r).mean() for _ in range(n)])            # user attends nothing
            acc = 0.5 * ((d_att > 0).mean() + (d_att > 0).mean())   # symmetric: attending NO mirrors attending YES
            per[r].append({"subject": s, "acc": float(acc), "false_yes": float((d_att < 0).mean()), "decision_s": round(2 * r * (soa or 0.25), 1), "single_flash_auc": float(_auc(sc, yt))})
            # abstain margin m: say YES only if score(YES)-score(NO) > m, with m set so an idle user yields YES <= 5% of the time
            m = np.quantile(d_idle, 0.95)
            per.setdefault(f"{r}_margin", []).append({"subject": s, "acc": float(0.5 * ((d_att > m).mean() + (1 - (-d_att > m).mean() * 0 - ((-d_att) > m).mean()))),
                "p_yes_given_yes": float((d_att > m).mean()), "p_yes_given_no": float(((-d_att) > m).mean()), "p_yes_given_idle": float((d_idle > m).mean()),
                "false_yes": float(((-d_att) > m).mean()), "decision_s": round(2 * r * (soa or 0.25), 1), "yes_on_idle": float((d_idle > m).mean())})
    for r in reps:
        if per[r]: report(f"{name}_r{r}", per[r], f"{note}; {r} flash(es) per option, SOA {soa or 0.25}s")
        if per.get(f"{r}_margin"): report(f"{name}_r{r}_margin", per[f"{r}_margin"], f"abstain margin calibrated to <=5% YES when not attending; P(YES|YES)={np.mean([x['p_yes_given_yes'] for x in per[f'{r}_margin']]):.3f}")


def _auc(sc, y):
    from sklearn.metrics import roc_auc_score; return roc_auc_score(y, sc)


for key, cls, subs, soa, note in [("p300_009", "BNCI2014_009", range(1, 11), 0.25, "visual P300 speller (BNCI2014_009), 16 ch"),
                                   ("p300_008", "BNCI2014_008", range(1, 9), 0.25, "visual P300, ALS patients (BNCI2014_008), 8 ch"),
                                   ("p300_015", "BNCI2015_003", range(1, 11), 0.25, "visual P300 (BNCI2015_003), 8 ch"),
                                   ("oddball_sos", "Sosulski2019", range(1, 6), 0.25, "AUDITORY oddball (Sosulski2019), 31 ch, eyes-free")]:
    if key in which:
        try: p300_eval(key, getattr(D, cls)(), list(subs), soa=soa, note=note)
        except Exception as e: print(key, "failed:", e)

# ---------------------------------------------------------------- SSVEP: best frequency pair, 1-4 s
def ssvep_eval(name, ds, subjects, secs_list=(1.0, 2.0, 4.0), note=""):
    for secs in secs_list:
        rows = []
        for s in subjects:
            try:
                par = SSVEP(fmin=3, fmax=40, tmin=0, tmax=secs, resample=256); X, y, _ = par.get_data(ds, subjects=[s])
            except Exception as e: print(name, s, "failed", e); continue
            classes = [c for c in np.unique(y) if c != "rest"]
            best = None
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    m = np.isin(y, [classes[i], classes[j]]); yy = y[m] == classes[i]
                    make = lambda: make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=3000, C=0.5))
                    sc = cv_scores(X[m], yy, make); pred = sc > 0; acc = (pred == yy).mean()
                    if best is None or acc > best[0]:
                        best = (acc, float(pred[~yy].mean()), classes[i], classes[j], m, yy, make)
            r = {"subject": s, "acc": float(best[0]), "false_yes": best[1], "decision_s": secs, "pair": [best[2], best[3]]}
            if "rest" in y:
                clf = best[6]().fit(X[best[4]], best[5]); r["yes_on_idle"] = float((clf.decision_function(X[y == "rest"]) > 0).mean())
            rows.append(r)
        if rows: report(f"{name}_{secs:.0f}s", rows, f"{note}; best 2 of {len(classes)} frequencies per subject")


if "ssvep_nak" in which:
    try: ssvep_eval("ssvep_nak", D.Nakanishi2015(), list(range(1, 10)), note="SSVEP Nakanishi2015, 8 occipital ch, needs flickering YES/NO on screen")
    except Exception as e: print("ssvep_nak failed:", e)
if "ssvep_kal" in which:
    try: ssvep_eval("ssvep_kal", D.Kalunga2016(), list(range(1, 13)), note="SSVEP Kalunga2016, 8 ch, has REST -> yes-on-idle measured")
    except Exception as e: print("ssvep_kal failed:", e)

# ---------------------------------------------------------------- c-VEP Thielen2015 (2 codes)
if "cvep_thielen" in which:
    try:
        from moabb.paradigms import CVEP
        ds = D.Thielen2015(); par = CVEP(fmin=1, fmax=40, tmin=0, tmax=2.1, resample=120); rows = []
        for s in range(1, 13):
            try: X, y, meta = par.get_data(ds, subjects=[s])
            except Exception as e: print("cvep", s, e); continue
            yy = y == "1.0"
            make = lambda: make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=3000, C=0.5))
            sc = cv_scores(X, yy, make); pred = sc > 0
            rows.append({"subject": s, "acc": float((pred == yy).mean()), "false_yes": float(pred[~yy].mean()), "decision_s": 2.1})
        if rows: report("cvep_thielen", rows, "c-VEP Thielen2015, code bit 1 vs 0 epochs (2.1 s)")
    except Exception as e: print("cvep failed:", e)

json.dump(RES, open("results/yesno_compare.json", "w"), indent=1, default=float)
print("saved results/yesno_compare.json")
