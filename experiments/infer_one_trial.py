"""Exact inference pipeline of the 'EEG -> descriptor -> Jev' result, on ONE held-out trial, with a LIVE Jev call.
Also measures how much information the 3-word descriptor carries about the class (subject 1, 5 folds).
"""
import sys, os, json, itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, ".")
from eeg_jev.jev import JevClient, choice_q
from eeg_jev.baseline import crop_filter
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
torch.manual_seed(0)

# ------------------------------------------------------------------ 1. data: BCI IV 2a, subject 1
d = np.load("cache/bnci_s1-3.npz", allow_pickle=True)
X, y, g = d["X"], d["y"], d["g"]; sfreq, tmin = float(d["sfreq"]), float(d["tmin"])
classes = sorted(set(y.tolist())); yi = np.array([classes.index(c) for c in y])
m = np.where(g == 1)[0]
Xraw = crop_filter(X[m], sfreq, tmin)          # 0.5-3.5 s after cue, 8-30 Hz bandpass, 22 ch -> (n, 22, 384)
yi = yi[m]

# ------------------------------------------------------------------ 2. descriptor vocabulary + Jev prompt (verbatim from adapter.py)
DIMS = ["left_sensorimotor_cortex_C3", "midline_Cz", "right_sensorimotor_cortex_C4"]
LEVELS = ["suppression", "no change", "increase"]
codes = list(itertools.product(range(3), repeat=3))
PRIMER = ("EEG motor imagery. Descriptor of mu/beta (8-30 Hz) power change during the trial relative to baseline "
          "over three sensorimotor sites. C3 (left cortex) controls the right hand, C4 (right cortex) controls the "
          "left hand, Cz (midline) represents the feet/both hands. Motor imagery suppresses mu/beta power over the "
          "cortex controlling the imagined body part. Which movement was imagined?")
PHYSIO = {"left_hand": "Imagined LEFT hand: suppression at C4 (right hemisphere) stronger than at C3.",
          "right_hand": "Imagined RIGHT hand: suppression at C3 (left hemisphere) stronger than at C4.",
          "feet": "Imagined FEET: suppression focused at midline Cz; lateral C3/C4 spared or increased.",
          "tongue": "Imagined TONGUE: bilateral suppression over lateral/inferior sensorimotor cortex (C5/C6 side), "
                    "often with C3/C4/Cz not suppressed or even increased (surround inhibition)."}
def state_for(code): return {"mu_beta_power_change": {dd: LEVELS[c] for dd, c in zip(DIMS, code)}}
question = choice_q(PRIMER, {c: PHYSIO[c] for c in classes})

# ------------------------------------------------------------------ 3. Jev lookup table: 27 strings -> Jev class probabilities (cached from live calls)
jev = JevClient()
T = np.array([[jev.ask(state_for(c), {"intent": question})["intent"]["probabilities"][k] for k in classes] for c in codes], dtype=np.float32)
T /= T.sum(1, keepdims=True); Tt = torch.tensor(T)

# ------------------------------------------------------------------ 4. encoder trained against the table (identical to adapter.py)
class Encoder(nn.Module):
    def __init__(self, d_in): super().__init__(); self.net = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 9))
    def forward(self, x): return self.net(x).view(-1, 3, 3)
code_idx = torch.tensor(np.array(codes))
def joint(logits, tau):
    p = Fn.softmax(logits / tau, -1); pj = torch.ones(logits.shape[0], 27)
    for i in range(3): pj = pj * p[:, i, code_idx[:, i]]
    return pj
def train(Ftr, ytr):
    enc = Encoder(Ftr.shape[1]); opt = torch.optim.AdamW(enc.parameters(), 3e-3, weight_decay=1e-2)
    Ftr, ytr = torch.tensor(Ftr), torch.tensor(ytr)
    for ep in range(300):
        enc.train(); opt.zero_grad(); tau = max(0.3, 2.0 * 0.985 ** ep)
        loss = Fn.nll_loss(torch.log(joint(enc(Ftr), tau) @ Tt + 1e-6), ytr); loss.backward(); opt.step()
    return enc.eval()

emitted, truth, preds = [], [], []
keep = None
for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xraw, yi)):
    ts = make_pipeline(Covariances("oas"), TangentSpace()); Ftr = ts.fit_transform(Xraw[tr]).astype(np.float32); Fte = ts.transform(Xraw[te]).astype(np.float32)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    enc = train((Ftr - mu) / sd, yi[tr])
    with torch.no_grad(): lv = enc(torch.tensor((Fte - mu) / sd)).argmax(-1).numpy()
    for i, row in zip(te, lv):
        code = tuple(int(v) for v in row); emitted.append(codes.index(code)); truth.append(int(yi[i])); preds.append(int(T[codes.index(code)].argmax()))
    if fold == 0: keep = (te[0], codes[emitted[0]])

emitted, truth, preds = map(np.array, (emitted, truth, preds))
print(f"subject 1, 5-fold CV, {len(truth)} held-out trials: accuracy of argmax(Jev table[emitted descriptor]) = {(preds == truth).mean():.3f}")
# information carried by the descriptor
joint_ct = np.zeros((27, 4))
for e, t in zip(emitted, truth): joint_ct[e, t] += 1
P = joint_ct / joint_ct.sum(); pe = P.sum(1, keepdims=True); pt = P.sum(0, keepdims=True)
nz = P > 0; MI = (P[nz] * np.log2(P[nz] / (pe @ pt)[nz])).sum()
He = -(pe[pe > 0] * np.log2(pe[pe > 0])).sum()
print(f"distinct descriptors used: {len(set(emitted.tolist()))}/27 | descriptor entropy {He:.2f} bits (max 4.75) | "
      f"mutual information descriptor<->class {MI:.2f} bits (class entropy 2.00; a perfect 4-class label would carry 2.00)")
print("descriptors by class in Jev's table: " + ", ".join(f"{classes[k]}={int((T.argmax(1)==k).sum())}" for k in range(4)))

# ------------------------------------------------------------------ 5. ONE held-out trial through the full pipeline with a LIVE Jev call
i, code = keep
print(f"\n=== single held-out trial #{int(m[i])} (subject 1, fold 0), true intent = {classes[yi[i]]} ===")
print(f"raw EEG crop shape: {Xraw[i].shape} (22 channels x 384 samples @128 Hz, 0.5-3.5 s post-cue, 8-30 Hz)")
print(f"encoder output (3 discrete levels): {code} -> descriptor text sent to Jev as `state`:")
state = state_for(code); print(json.dumps(state, indent=2))
print("question sent to Jev:"); print(json.dumps({"intent": question}, indent=2)[:900] + " ...")
live = JevClient(model="jev-1.13.0")            # different alias => cache miss => genuinely live call
ans = live.ask(state, {"intent": question})["intent"]
print("Jev live response:"); print(json.dumps({k: v for k, v in ans.items()}, indent=2))
print(f"cached table for this descriptor: {dict(zip(classes, np.round(T[codes.index(code)], 3)))}")
print(f"pipeline prediction: {ans['choice']} | truth: {classes[yi[i]]} | {'CORRECT' if ans['choice']==classes[yi[i]] else 'WRONG'}")
