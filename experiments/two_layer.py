"""Two-layer all-EEG speed-dial, Shin 2017 datasets A+B (same 29 subjects, same 30-channel montage).

Layer 1 (command):      mental arithmetic vs rest, Shin 2017B. Sliding 3 s windows / 1 s hop, K-streak trigger.
Layer 2 (confirmation): left-hand imagery = YES, right-hand imagery = NO, Shin 2017A. Decoded from a single D-second
                        window that starts when Layer 1 fires. Variants:
                          A2  : 2-class yes/no decoder, decide YES if P(yes) >= theta, NO if <= 1-theta, else no-decision (abort)
                          A3  : 3-class yes/no/idle decoder; idle class trained on Shin 2017B rest trials of the training fold
                          B   : "repeat the command" confirmation: Layer-2 decoder = Layer-1 decoder (task vs rest) on the next window
Sequential held-out simulation per subject and fold:
  genuine command  : held-out task trial -> L1 fires? -> user performs YES (held-out left-MI trial) -> L2 decision
  idle, unattended : held-out rest trial  -> L1 false alarm? -> user is NOT responding; L2 sees the SAME rest recording
                     continuing after the fire (if >= D s remain) else the next held-out rest trial
  idle, attended NO: as above but the user notices the prompt and performs NO (held-out right-MI trial)
Accidental action = an action executed without a genuine command. Reported per hour of idle.
Usage: two_layer.py <subjects...>   (env: D=4 confirm window seconds, THETA=0.6)
"""
import sys, os, json, numpy as np, mne
sys.path.insert(0, ".")
mne.set_log_level("ERROR")
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

subjects = [int(s) for s in sys.argv[1:]] or list(range(1, 11))
SF = 128; WIN, HOP = 3 * SF, 1 * SF; TRIAL_S = 10.0
D = float(os.environ.get("D", "4")); DW = int(D * SF); THETA = float(os.environ.get("THETA", "0.6"))
OPS = [(0.7, 2), (0.8, 2), (0.9, 2), (0.75, 3)]          # Layer-1 operating points (tau, K)
sos = butter(4, [4, 30], btype="band", fs=SF, output="sos")


def load(s, which):
    fn = f"cache/shin{which}_s{s:02d}.npz"
    if os.path.exists(fn):
        d = np.load(fn, allow_pickle=True); return d["X"], d["y"], list(d["ch"])
    from moabb.datasets import Shin2017A, Shin2017B
    ds = (Shin2017A if which == "A" else Shin2017B)(accept=True)
    data = ds.get_data(subjects=[s])[s]; Xs, ys = [], []
    for sess in data.values():
        for raw in sess.values():
            raw = raw.copy().load_data().pick("eeg"); raw.filter(1, 40, verbose=False)
            ev, ev_id = mne.events_from_annotations(raw, verbose=False)
            ep = mne.Epochs(raw, ev, ev_id, tmin=0.0, tmax=TRIAL_S, baseline=None, preload=True, verbose=False)
            ep.resample(SF, verbose=False); inv = {v: k for k, v in ev_id.items()}
            Xs.append(ep.get_data(copy=True)); ys.append(np.array([inv[e] for e in ep.events[:, 2]])); ch = ep.ch_names
    X = np.concatenate(Xs).astype(np.float32); y = np.concatenate(ys)
    np.savez_compressed(fn, X=X, y=y, ch=np.array(ch)); return X, y, ch


def windows(x, w=WIN, h=HOP):
    return np.stack([x[:, i:i + w] for i in range(0, x.shape[-1] - w + 1, h)])


def clf():
    return make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=3000, C=0.5))


def l1_fire(P, tau, K):
    streak = 0
    for w, p in enumerate(P):
        streak = streak + 1 if p >= tau else 0
        if streak == K: return w
    return None


def l2_decide(probs, classes, theta):
    """probs over classes -> 'yes' | 'no' | 'idle' | 'undecided'"""
    p = dict(zip(classes, probs))
    if "idle" in p and max(p, key=p.get) == "idle": return "idle"
    py = p["yes"] / (p["yes"] + p["no"])
    if py >= theta: return "yes"
    if py <= 1 - theta: return "no"
    return "undecided"


all_rows = []; subj_stats = []
ROWS_FN = f"results/two_layer_rows_D{int(D)}_theta{THETA}.json"
if os.environ.get("ANALYZE_ONLY"):
    all_rows = json.load(open(ROWS_FN)); subjects = sorted(set(r["subject"] for r in all_rows))
for s in ([] if os.environ.get("ANALYZE_ONLY") else subjects):
    XB, yB, chB = load(s, "B"); XA, yA, chA = load(s, "A")
    assert chA == chB, f"montage mismatch subject {s}"
    if s == subjects[0]: print(f"channels identical A/B ({len(chA)}). A classes {dict(zip(*np.unique(yA, return_counts=True)))}, B classes {dict(zip(*np.unique(yB, return_counts=True)))}")
    XB = sosfiltfilt(sos, XB, axis=-1); XA = sosfiltfilt(sos, XA, axis=-1)
    task = [c for c in np.unique(yB) if c != "rest"][0]
    yA2 = np.where(yA == "left_hand", "yes", "no")
    foldsB = list(StratifiedKFold(5, shuffle=True, random_state=0).split(XB, yB))
    foldsA = list(StratifiedKFold(5, shuffle=True, random_state=0).split(XA, yA2))
    rng = np.random.RandomState(s)
    for k in range(5):
        trB, teB = foldsB[k]; trA, teA = foldsA[k]
        # ---- Layer 1 decoder (task vs rest) on 3 s windows
        Xw = np.concatenate([windows(XB[i]) for i in trB]); yw = np.concatenate([[yB[i]] * 8 for i in trB])
        L1 = clf().fit(Xw, yw); ti = list(L1.classes_).index(task)
        # ---- Layer 2 decoders on D-second crops (crops every 1 s for training augmentation)
        cropsA = np.concatenate([windows(XA[i], DW) for i in trA]); ycA = np.concatenate([[yA2[i]] * windows(XA[i], DW).shape[0] for i in trA])
        L2_2 = clf().fit(cropsA, ycA)
        rest_tr = [i for i in trB if yB[i] == "rest"]
        cropsR = np.concatenate([windows(XB[i], DW) for i in rest_tr]); ycR = np.array(["idle"] * len(cropsR))
        L2_3 = clf().fit(np.concatenate([cropsA, cropsR]), np.concatenate([ycA, ycR]))
        L2_B = L1  # "repeat the command" variant re-uses the Layer-1 decoder on a 3 s window
        # held-out confirmation pools
        yes_pool = list(rng.permutation([i for i in teA if yA2[i] == "yes"])); no_pool = list(rng.permutation([i for i in teA if yA2[i] == "no"]))
        rest_te = [i for i in teB if yB[i] == "rest"]
        yes_ptr = no_ptr = 0
        # standalone L2 quality on held-out MI trials (first D s crop), both decoders
        for i in teA:
            x = XA[i][:, :DW][None]
            for name, m in (("A2", L2_2), ("A3", L2_3)):
                dec = l2_decide(m.predict_proba(x)[0], list(m.classes_), THETA)
                all_rows.append({"subject": s, "fold": k, "kind": "l2_standalone", "variant": name, "truth": yA2[i], "decision": dec})
        # ---- idle-crop census: L2 behaviour on all idle EEG, tagged by whether L1 (tau 0.7, K 2) had fired earlier in the trial
        for i in rest_te:
            P = L1.predict_proba(windows(XB[i]))[:, ti]; w = l1_fire(P, 0.7, 2)
            t_fire = None if w is None else WIN + w * HOP
            for start in range(0, XB.shape[-1] - DW + 1, SF):
                x = XB[i][:, start:start + DW][None]
                tag = "no_l1_fire_in_trial" if t_fire is None else ("after_l1_fire" if start >= t_fire else "before_l1_fire")
                r_ = {"subject": s, "fold": k, "kind": "idle_census", "tag": tag, "l1_maxp_trial": float(P.max())}
                for name, m in (("A2", L2_2), ("A3", L2_3)):
                    pr = m.predict_proba(x)[0]; r_[f"{name}_decision"] = l2_decide(pr, list(m.classes_), THETA); r_[f"{name}_pyes"] = float(dict(zip(m.classes_, pr))["yes"])
                all_rows.append(r_)
        # ---- sequential simulation over held-out Layer-1 trials
        for i in teB:
            P = L1.predict_proba(windows(XB[i]))[:, ti]
            genuine = yB[i] == task
            for tau, K in OPS:
                w = l1_fire(P, tau, K)
                row = {"subject": s, "fold": k, "kind": "seq", "tau": tau, "K": K, "genuine": bool(genuine), "l1_fired": w is not None,
                       "l1_latency": None if w is None else WIN / SF + w * HOP / SF, "l1_maxp": float(P.max())}
                if w is not None:
                    t_fire = WIN + w * HOP  # sample index in the trial at which L1 fired
                    if genuine:
                        j = yes_pool[yes_ptr % len(yes_pool)]; yes_ptr += 1; x_conf = XA[j][:, :DW]; x_conf_B = XB[i][:, t_fire:t_fire + WIN] if t_fire + WIN <= XB.shape[-1] else None
                        row["conf_source"] = "yes_trial"
                    else:
                        # unattended: same rest recording continues after the fire, else next held-out rest trial
                        if t_fire + DW <= XB.shape[-1]:
                            x_conf = XB[i][:, t_fire:t_fire + DW]; row["conf_source"] = "same_rest_after_fire"
                        else:
                            j = rest_te[(rest_te.index(i) + 1) % len(rest_te)]; x_conf = XB[j][:, :DW]; row["conf_source"] = "next_rest_trial"
                        x_conf_B = x_conf[:, :WIN]
                        jn = no_pool[no_ptr % len(no_pool)]; no_ptr += 1; x_no = XA[jn][:, :DW]
                    for name, m in (("A2", L2_2), ("A3", L2_3)):
                        pr = m.predict_proba(x_conf[None])[0]; row[f"{name}_decision"] = l2_decide(pr, list(m.classes_), THETA)
                        row[f"{name}_pyes"] = float(dict(zip(m.classes_, pr))["yes"])
                        if not genuine:
                            row[f"{name}_attendedNO_decision"] = l2_decide(m.predict_proba(x_no[None])[0], list(m.classes_), THETA)
                    if x_conf_B is not None:
                        pB = L2_B.predict_proba(x_conf_B[None])[0][ti]; row["B_decision"] = "yes" if pB >= tau else "no"; row["B_ptask"] = float(pB)
                    else:
                        row["B_decision"] = "undecided"
                all_rows.append(row)
    print(f"subject {s} done", flush=True)

if not os.environ.get("ANALYZE_ONLY"): json.dump(all_rows, open(ROWS_FN, "w"))

# ============================================================== analysis
import pandas as pd
df = pd.DataFrame(all_rows)
st = df[df.kind == "l2_standalone"]
print(f"\n=== Layer 2 standalone on held-out imagery trials (first {D:.0f} s), theta={THETA} ===")
for v in ("A2", "A3"):
    d = st[st.variant == v]; dec = d.decision.values; tr = d.truth.values
    decided = np.isin(dec, ["yes", "no"])
    print(f"{v}: yes/no accuracy among decided = {(dec[decided] == tr[decided]).mean():.3f} | decided {decided.mean():.2f} | "
          f"P(YES|intended YES) = {(dec[tr=='yes']=='yes').mean():.3f} [missed confirmations {1-(dec[tr=='yes']=='yes').mean():.3f}] | "
          f"P(YES|intended NO) = {(dec[tr=='no']=='yes').mean():.3f} [false confirmations] | idle-called {(dec=='idle').mean():.2f}")
    per = d.groupby("subject").apply(lambda g: (g.decision[g.decision.isin(['yes','no'])] == g.truth[g.decision.isin(['yes','no'])]).mean())
    print("    per-subject yes/no acc:", {int(k): round(v_, 2) for k, v_ in per.items()})

seq = df[df.kind == "seq"].copy()
seq["genuine"] = seq.genuine.astype(bool); seq["l1_fired"] = seq.l1_fired.astype(bool)
idle_hours = lambda d: (~d.genuine).sum() * TRIAL_S / 3600
print(f"\n=== Sequential pipeline, held-out, {len(subjects)} subjects, confirm window D={D:.0f}s ===")
print(f"{'L1 op':>10} | {'L1 det':>6} {'L1 FA/h':>7} | {'variant':>7} {'action|genuine':>14} {'acc/h unattended':>16} {'acc/h attendedNO':>16} {'e2e lat s':>9} | reduction vs single layer")
summary = {}
for tau, K in OPS:
    d = seq[(seq.tau == tau) & (seq.K == K)]
    gen = d[d.genuine]; idl = d[~d.genuine]
    l1_det = gen.l1_fired.mean(); l1_fa_h = idl.l1_fired.sum() / idle_hours(idl)
    for v in ("A2", "A3", "B"):
        col = f"{v}_decision"
        act_gen = (gen.l1_fired & (gen[col] == "yes")).mean()
        acc_unatt = (idl.l1_fired & (idl[col] == "yes")).sum() / idle_hours(idl)
        acc_att = (idl.l1_fired & (idl.get(f"{v}_attendedNO_decision", pd.Series(index=idl.index, dtype=object)) == "yes")).sum() / idle_hours(idl) if v != "B" else float("nan")
        lat = gen.l1_latency[gen.l1_fired & (gen[col] == "yes")].mean() + (D if v != "B" else WIN / SF)
        red = 1 - acc_unatt / l1_fa_h if l1_fa_h > 0 else float("nan")
        summary[f"{tau}_{K}_{v}"] = {"l1_detect": l1_det, "l1_fa_h": l1_fa_h, "action_given_genuine": act_gen, "accidental_h_unattended": acc_unatt,
                                     "accidental_h_attendedNO": acc_att, "e2e_latency": lat, "reduction": red}
        print(f"tau{tau:.2f},K{K} | {100*l1_det:5.1f}% {l1_fa_h:7.1f} | {v:>7} {100*act_gen:13.1f}% {acc_unatt:16.1f} {acc_att:16.1f} {lat:9.1f} | {100*red:5.1f}% fewer accidental actions (unattended)")

# ---- independence of Layer-1 and Layer-2 errors
print("\n=== Are Layer-1 and Layer-2 errors independent? ===")
tau, K = 0.7, 2
d = seq[(seq.tau == tau) & (seq.K == K) & (~seq.genuine)]
for v in ("A2", "A3"):
    # trial level, idle path: does L2 say YES more often right after an L1 false alarm than on comparable idle EEG without one?
    # Compare P(yes) on same-rest-after-fire segments vs the standalone idle behaviour: use L1 max prob as the L1 'error strength'.
    fired = d[d.l1_fired]
    r, p = spearmanr(fired.l1_maxp, fired[f"{v}_pyes"]) if len(fired) > 5 else (np.nan, np.nan)
    print(f"{v}: among {len(fired)} L1 false alarms, Spearman(L1 confidence, L2 P(yes)) = {r:+.3f} (p={p:.3f})")
    # subject level: L1 false-alarm rate vs L2 yes-on-idle rate
    g = d.groupby("subject").apply(lambda x: pd.Series({"fa": x.l1_fired.mean(), "yes_idle": (x[x.l1_fired][f"{v}_decision"] == "yes").mean()}))
    r2, p2 = spearmanr(g.fa, g.yes_idle)
    obs = (d.l1_fired & (d[f"{v}_decision"] == "yes")).mean()
    pred_indep = d.l1_fired.mean() * (d[d.l1_fired][f"{v}_decision"] == "yes").mean()
    print(f"    subject level: Spearman(L1 FA rate, L2 yes|idle rate) = {r2:+.3f} (p={p2:.3f}); per-subject L2 yes|idle: {dict((int(a), round(b,2)) for a,b in g.yes_idle.items())}")
    # genuine path: L1 latency/confidence vs L2 success (trials come from separate recordings -> independent by construction)
cen = df[df.kind == "idle_census"]
print(f"\nIdle-EEG census ({len(cen)} {D:.0f}s crops of held-out rest trials): P(L2 says YES | idle EEG)")
for v in ("A2", "A3"):
    by = cen.groupby("tag").apply(lambda x: pd.Series({"n": len(x), "p_yes": (x[f"{v}_decision"] == "yes").mean(), "mean_pyes": x[f"{v}_pyes"].mean()}))
    print(f"  {v}: " + " | ".join(f"{t}: n={int(r_.n)}, P(yes)={r_.p_yes:.3f}, mean p_yes={r_.mean_pyes:.3f}" for t, r_ in by.iterrows()))
    r, p = spearmanr(cen.l1_maxp_trial, cen[f"{v}_pyes"])
    print(f"      Spearman(L1 max prob of the trial, L2 p_yes of its crops) = {r:+.3f} (p={p:.3g})")
    for tau_, K_ in OPS:
        d_ = seq[(seq.tau == tau_) & (seq.K == K_) & (~seq.genuine)]
        fa_h = d_.l1_fired.sum() / idle_hours(d_); p_all = (cen[f"{v}_decision"] == "yes").mean()
        obs = (d_.l1_fired & (d_[f"{v}_decision"] == "yes")).sum() / idle_hours(d_)
        print(f"      tau{tau_},K{K_}: accidental/h observed={obs:.1f} vs independence prediction FA/h x P(yes|idle)={fa_h:.1f} x {p_all:.3f} = {fa_h*p_all:.1f}")
gen = seq[(seq.tau == tau) & (seq.K == K) & seq.genuine & seq.l1_fired]
for v in ("A2", "A3"):
    r, p = spearmanr(gen.l1_maxp, gen[f"{v}_pyes"])
    print(f"{v}: genuine path, Spearman(L1 confidence, L2 P(yes)) = {r:+.3f} (p={p:.3f})  [YES trials come from a separate recording, so any correlation is subject-level]")
gsub = seq[(seq.tau == tau) & (seq.K == K)].groupby("subject").apply(lambda x: pd.Series({"l1_det": x[x.genuine].l1_fired.mean(), "l1_fa": x[~x.genuine].l1_fired.mean(),
    "l2_acc_A2": (st[(st.subject == x.name) & (st.variant == "A2")].pipe(lambda z: (z.decision == z.truth).mean()))}))
r, p = spearmanr(gsub.l1_det, gsub.l2_acc_A2); r3, p3 = spearmanr(gsub.l1_fa, gsub.l2_acc_A2)
print(f"subject level: Spearman(L1 detect, L2 yes/no acc) = {r:+.3f} (p={p:.3f}); Spearman(L1 FA rate, L2 yes/no acc) = {r3:+.3f} (p={p3:.3f})")
json.dump({"summary": summary, "D": D, "theta": THETA, "subjects": subjects}, open(f"results/two_layer_summary_D{int(D)}_theta{THETA}.json", "w"), indent=1)
