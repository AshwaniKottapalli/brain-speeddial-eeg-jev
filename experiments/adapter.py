"""Black-box-trained neural adapter with Jev as the fixed decision core.

Architecture:  EEG epoch --> small encoder (trainable) --> discrete semantic descriptor (text) --> Jev --> intent probs

Trick that makes training possible WITHOUT gradients through Jev:
  the descriptor vocabulary is finite (L levels x D dims -> L**D strings), so we query Jev ONCE per string and
  store its class-probability vector in a lookup table T[code] (this is the only API cost).
  The encoder emits a categorical distribution over codes; expected Jev output = sum_code q(code) * T[code].
  That expectation is differentiable in the encoder parameters, so we train the encoder end-to-end against
  Jev's real outputs by cross-entropy, with straight-through/Gumbel sampling. At test time: hard argmax code
  -> the text string -> Jev's answer (identical to what the API returns for that string).

Usage: adapter.py <dataset> [semantic|opaque] [levels] [dims]
"""
import sys, json, itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, ".")
from eeg_jev.jev import JevClient, choice_q
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

which = sys.argv[1]; vocab = sys.argv[2] if len(sys.argv) > 2 else "semantic"
L = int(sys.argv[3]) if len(sys.argv) > 3 else 5
INPUT = sys.argv[4] if len(sys.argv) > 4 else "ts"        # ts = Riemannian tangent space of raw EEG | feats = JSON numbers
TABLE = sys.argv[5] if len(sys.argv) > 5 else "jev"       # jev = real Jev lookup | random = shuffled control
import os; NTRAIN = int(os.environ.get("NTRAIN", "0")); SEED = int(os.environ.get("SEED", "0"))     # if >0: subsample training fold to NTRAIN trials per class
D = json.load(open(f"cache/{which}_feats.json"))
y = np.array(D["y"]); g = np.array(D["g"]); classes = sorted(set(y.tolist())); C = len(classes)
F = np.load(f"cache/{which}_F.npy")
name = {"eegbci": "eegbci_s1-10", "bnci": "bnci_s1-3"}[which]
_d = np.load(f"cache/{name}.npz", allow_pickle=True)
from eeg_jev.baseline import crop_filter
Xraw = crop_filter(_d["X"], float(_d["sfreq"]), float(_d["tmin"]))
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
yi = np.array([classes.index(c) for c in y])

# ---- descriptor vocabulary -------------------------------------------------------------------------------
DIMS = ["left_sensorimotor_cortex_C3", "midline_Cz", "right_sensorimotor_cortex_C4"]
if vocab.endswith("4d"):
    DIMS = DIMS + ["bilateral_lateral_inferior_C5_C6_face_tongue_area"]
LEVELS5 = ["strong suppression", "mild suppression", "no change", "mild increase", "strong increase"]
LEVELS = LEVELS5 if L == 5 else ["suppression", "no change", "increase"]
codes = list(itertools.product(range(L), repeat=len(DIMS)))  # L**3 codes
PRIMER = ("EEG motor imagery. Descriptor of mu/beta (8-30 Hz) power change during the trial relative to baseline "
          "over three sensorimotor sites. C3 (left cortex) controls the right hand, C4 (right cortex) controls the "
          "left hand, Cz (midline) represents the feet/both hands. Motor imagery suppresses mu/beta power over the "
          "cortex controlling the imagined body part. Which movement was imagined?")

def code_to_state(code):
    if vocab.startswith("semantic"):
        return {"mu_beta_power_change": {d: LEVELS[c] for d, c in zip(DIMS, code)}}
    return {"neural_code": {f"feature_{i}": f"level_{c}" for i, c in enumerate(code)}}  # no physiology words

PHYSIO = {
    "left_hand": "Imagined LEFT hand: suppression at C4 (right hemisphere) stronger than at C3.",
    "right_hand": "Imagined RIGHT hand: suppression at C3 (left hemisphere) stronger than at C4.",
    "hands": "Imagined BOTH hands: bilateral, roughly symmetric suppression at C3 and C4.",
    "feet": "Imagined FEET: suppression focused at midline Cz; lateral C3/C4 spared or increased.",
    "tongue": "Imagined TONGUE: bilateral suppression over lateral/inferior sensorimotor cortex (C5/C6 side), "
              "often with C3/C4/Cz not suppressed or even increased (surround inhibition).",
}
crit = {c: (PHYSIO[c] if "crit" in vocab or vocab.endswith("4d") else None) for c in classes}
instr = PRIMER if vocab.startswith("semantic") else "Abstract neural code from an EEG trial. Which movement was imagined?"

# ---- build the Jev lookup table (the only API calls) ---------------------------------------------------
jev = JevClient()
T = np.zeros((len(codes), C), dtype=np.float32)
for ci, code in enumerate(tqdm(codes, desc="querying Jev for vocabulary")):
    ans = jev.ask(code_to_state(code), {"intent": choice_q(instr, crit)})["intent"]
    T[ci] = [ans["probabilities"].get(c, 0.0) for c in classes]
T = T / T.sum(1, keepdims=True)
if TABLE == "random":
    rs = np.random.RandomState(123); T = T[rs.permutation(len(T))]   # same marginal stats, semantics destroyed
print(f"vocab={vocab} L={L} codes={len(codes)} | Jev mock={jev.mock} | cost so far ${jev.cost_usd:.3f}")
print("Jev table: fraction of codes whose argmax is each class:",
      dict(zip(classes, np.bincount(T.argmax(1), minlength=C) / len(codes))))
print("Jev table: mean max-prob", T.max(1).mean().round(3), "| distinct argmax classes", len(set(T.argmax(1))))
Tt = torch.tensor(T)

# ---- encoder ---------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, d_in, n_dims, L, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden, n_dims * L))
        self.n_dims, self.L = n_dims, L
    def forward(self, x):
        return self.net(x).view(-1, self.n_dims, self.L)  # per-dim logits over levels

code_idx = torch.tensor(np.array(codes))  # (n_codes, n_dims)

def code_probs(logits, tau=1.0, hard=False):
    """joint distribution over the L**D codes from per-dim (independent) level distributions"""
    if hard:
        lv = logits.argmax(-1)  # (B, n_dims)
        onehot = torch.zeros(logits.shape[0], len(codes))
        flat = sum(lv[:, i] * (L ** (len(DIMS) - 1 - i)) for i in range(len(DIMS)))
        onehot[torch.arange(logits.shape[0]), flat] = 1.0
        return onehot
    p = Fn.softmax(logits / tau, -1)  # (B, n_dims, L)
    # joint prob of a code = prod over dims of p[dim, level]
    pj = torch.ones(logits.shape[0], len(codes))
    for i in range(len(DIMS)):
        pj = pj * p[:, i, code_idx[:, i]]
    return pj

def make_input(tr, te):
    if INPUT == "ts":
        ts = make_ts(); Ftr = ts.fit_transform(Xraw[tr]); Fte = ts.transform(Xraw[te])
        return Ftr.astype(np.float32), Fte.astype(np.float32)
    return F[tr], F[te]

def make_ts():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(Covariances("oas"), TangentSpace())

def train_eval(Xtr, ytr, Xte, yte, epochs=300, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = torch.tensor((Xtr - mu) / sd); Xte = torch.tensor((Xte - mu) / sd)
    ytr = torch.tensor(ytr); enc = Encoder(Xtr.shape[1], len(DIMS), L)
    opt = torch.optim.AdamW(enc.parameters(), 3e-3, weight_decay=1e-2)
    for ep in range(epochs):
        enc.train(); opt.zero_grad()
        tau = max(0.3, 2.0 * (0.985 ** ep))
        pj = code_probs(enc(Xtr), tau)
        p_class = pj @ Tt                       # expected Jev output (B, C)
        loss = Fn.nll_loss(torch.log(p_class + 1e-6), ytr)
        loss.backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        pj_hard = code_probs(enc(Xte), hard=True)
        p_class = pj_hard @ Tt                  # exactly what Jev returns for the emitted descriptor
        pred = p_class.argmax(1).numpy()
        codes_used = pj_hard.argmax(1).numpy()
    # control: identical encoder trunk with a direct softmax head (no Jev, no discrete bottleneck)
    torch.manual_seed(seed)
    direct = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, C))
    opt = torch.optim.AdamW(direct.parameters(), 3e-3, weight_decay=1e-2)
    for ep in range(epochs):
        direct.train(); opt.zero_grad(); Fn.cross_entropy(direct(Xtr), ytr).backward(); opt.step()
    direct.eval()
    with torch.no_grad():
        pred_direct = direct(Xte).argmax(1).numpy()
    return pred, codes_used, pred_direct

# ---- within-subject 5-fold CV -------------------------------------------------------------------------------
all_pred, all_true, all_codes, all_direct = [], [], [], []
for s in np.unique(g):
    m = np.where(g == s)[0]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    accs, daccs = [], []
    for tr, te in cv.split(F[m], yi[m]):
        if NTRAIN:
            rs = np.random.RandomState(7); keep = np.concatenate([rs.permutation(tr[yi[m][tr] == k_])[:NTRAIN] for k_ in range(C)]); tr = keep
        Ftr, Fte = make_input(m[tr], m[te])
        pred, cu, pd_ = train_eval(Ftr, yi[m][tr], Fte, yi[m][te], seed=SEED)
        accs.append((pred == yi[m][te]).mean()); all_pred += pred.tolist(); all_true += yi[m][te].tolist(); all_codes += cu.tolist()
        all_direct += pd_.tolist(); daccs.append((pd_ == yi[m][te]).mean())
    print(f"  subj {s}: EEG->adapter->Jev acc={np.mean(accs):.3f}   | same trunk, direct softmax (no Jev) acc={np.mean(daccs):.3f}")
all_pred, all_true = np.array(all_pred), np.array(all_true)
acc = (all_pred == all_true).mean(); bacc = balanced_accuracy_score(all_true, all_pred)
dacc = (np.array(all_direct) == all_true).mean()
print(f"direct-softmax control (no Jev bottleneck): acc={dacc:.3f}")
print(f"\n{which} adapter({vocab}, L={L}, input={INPUT}, table={TABLE}): acc={acc:.3f} bal_acc={bacc:.3f} chance={1/C:.3f} n={len(all_true)}")
print("confusion (order", classes, ")\n", confusion_matrix(all_true, all_pred))
uc, cnt = np.unique(all_codes, return_counts=True)
print(f"distinct descriptors emitted: {len(uc)} / {len(codes)}. Top 5:")
for ci in uc[np.argsort(-cnt)][:5]:
    print(f"   {cnt[list(uc).index(ci)]:4d}x  {code_to_state(codes[ci])}  -> Jev says {classes[T[ci].argmax()]} p={T[ci].max():.2f}")
print("descriptor most often emitted per TRUE class (does the adapter speak physiology?):")
all_codes = np.array(all_codes)
for k_, c in enumerate(classes):
    cc = all_codes[all_true == k_]; top = np.bincount(cc, minlength=len(codes)).argmax()
    print(f"   {c:11s} -> {(cc == top).mean():.0%} of trials: {code_to_state(codes[top])['mu_beta_power_change'] if vocab.startswith('semantic') else codes[top]}")
print(f"cumulative Jev cost ${jev.cost_usd:.3f}")
json.dump({"acc": float(acc), "bal_acc": float(bacc), "direct_acc": float(dacc), "vocab": vocab, "L": L, "table": T.tolist(),
           "codes": codes, "classes": classes}, open(f"results/adapter_{which}_{vocab}_L{L}_{INPUT}_{TABLE}{'_n'+str(NTRAIN) if NTRAIN else ''}.json", "w"))
