"""Zero-shot: EEG feature JSON -> Jev Choice on intent. No exemplars. Two prompt variants:
  physio : criteria describe the expected neural signature of each intent (tests whether Jev can apply rules)
  blind  : criteria are just the intent names (tests whether Jev has any prior for EEG->intent)
"""
import sys, json, numpy as np
sys.path.insert(0, ".")
from eeg_jev.jev import JevClient, choice_q
from tqdm import tqdm

which = sys.argv[1] if len(sys.argv) > 1 else "eegbci"
variant = sys.argv[2] if len(sys.argv) > 2 else "physio"
max_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 400

D = json.load(open(f"cache/{which}_feats.json"))
feats, y, g = D["feats"], np.array(D["y"]), np.array(D["g"])
classes = sorted(set(y.tolist()))

PRIMER = ("EEG motor-imagery trial. Values are event-related spectral changes in dB relative to the pre-cue "
          "baseline (negative = power suppression / ERD, positive = increase). C3 is over the LEFT sensorimotor "
          "cortex (controls the RIGHT hand), C4 is over the RIGHT sensorimotor cortex (controls the LEFT hand), "
          "Cz is the midline (feet / both hands). FC*/CP* are neighbours of C3/C4. Motor imagery suppresses mu "
          "(8-13 Hz) and beta (13-30 Hz) power over the cortex controlling the imagined body part.")
PHYSIO = {
    "left_hand": "Imagined LEFT hand: mu/beta suppression strongest at C4 (right hemisphere); laterality C3-C4 positive.",
    "right_hand": "Imagined RIGHT hand: mu/beta suppression strongest at C3 (left hemisphere); laterality C3-C4 negative.",
    "hands": "Imagined BOTH hands: bilateral suppression at C3 and C4, roughly symmetric; Cz less involved.",
    "feet": "Imagined FEET: suppression focused at midline Cz; C3/C4 relatively spared or even increased.",
    "tongue": "Imagined TONGUE: bilateral, weaker central suppression; less lateralised than hands.",
}
crit = {c: (PHYSIO[c] if variant == "physio" else None) for c in classes}
instr = (PRIMER + " Which movement was the participant imagining?") if variant == "physio" \
        else "This is an EEG recording during motor imagery. Which movement was the participant imagining?"

rng = np.random.RandomState(0)
idx = rng.permutation(len(y))[:max_trials]
jev = JevClient()
print("mock:", jev.mock, "| trials:", len(idx), "| variant:", variant)
rows = []
for i in tqdm(idx):
    state = {"eeg_trial_features": feats[i]}
    ans = jev.ask(state, {"intent": choice_q(instr, crit)})
    a = ans["intent"]
    rows.append({"i": int(i), "y": y[i], "g": int(g[i]), "pred": a["choice"], "p_true": a["probabilities"].get(y[i], 0),
                 "conf": a.get("confidence"), "probs": a["probabilities"], "tokens": ans["_usage"]["input_tokens"]})
pred = np.array([r["pred"] for r in rows]); yy = np.array([r["y"] for r in rows])
acc = (pred == yy).mean()
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
bacc = balanced_accuracy_score(yy, pred)
print(f"\n{which}/{variant}: acc={acc:.3f} bal_acc={bacc:.3f} chance={1/len(classes):.3f} "
      f"mean p(true)={np.mean([r['p_true'] for r in rows]):.3f} | pred dist={dict(zip(*np.unique(pred, return_counts=True)))}")
print("confusion (rows=true, cols=pred, order", classes, ")\n", confusion_matrix(yy, pred, labels=classes))
print(f"tokens this run: {sum(r['tokens'] for r in rows)} | cumulative cost ${jev.cost_usd:.3f} | cache hits {jev.cache_hits}")
json.dump({"acc": acc, "bal_acc": bacc, "rows": rows}, open(f"results/zero_shot_{which}_{variant}.json", "w"))
