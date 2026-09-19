"""Few-shot in-context: labeled EEG-feature exemplars from the SAME subject are packed into Jev's state next
to the query trial. Jev must do the EEG->intent mapping in-context. No external classifier.
Usage: few_shot.py <dataset> <k_per_class> <n_queries_per_subject> [compact|full]
"""
import sys, json, numpy as np
sys.path.insert(0, ".")
from eeg_jev.jev import JevClient, choice_q
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

which = sys.argv[1]; k = int(sys.argv[2]); nq = int(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else "compact"
D = json.load(open(f"cache/{which}_feats.json"))
feats, y, g = D["feats"], np.array(D["y"]), np.array(D["g"])
classes = sorted(set(y.tolist()))


def compact(f):
    """Keep only the most diagnostic numbers to fit many exemplars in context."""
    out = {c: {"mu": f[c]["mu"], "beta": round((f[c]["beta_low"] + f[c]["beta_high"]) / 2, 1)}
           for c in ("C3", "Cz", "C4") if c in f}
    out["lat_C3-C4_mu"] = f["laterality_C3_minus_C4"]["mu"]
    out["Cz-lateral_mu"] = f["midline_vs_lateral_Cz_minus_mean(C3,C4)"]["mu"]
    return out

enc = compact if mode == "compact" else (lambda f: f)
PRIMER = ("EEG motor-imagery decoding. Each example is one trial: spectral power change (dB vs pre-cue baseline; "
          "negative = suppression) at C3 (left cortex / right hand), Cz (midline / feet), C4 (right cortex / left hand). "
          "Labeled examples come from the same participant and session as the query. Infer the mapping from the "
          "examples and classify the query trial.")
rng = np.random.RandomState(1)
jev = JevClient()
rows = []
for s in np.unique(g):
    ids = np.where(g == s)[0]
    per = {c: rng.permutation(ids[y[ids] == c]) for c in classes}
    support = {c: per[c][:k] for c in classes}
    pool = np.concatenate([per[c][k:] for c in classes]); rng.shuffle(pool)
    queries = pool[:nq]
    examples = []
    for c in classes:
        for i in support[c]:
            examples.append({"trial": enc(feats[i]), "intent": c})
    rng.shuffle(examples)
    for qi in tqdm(queries, desc=f"subj {s}", leave=False):
        state = {"labeled_examples": examples, "query_trial": enc(feats[qi])}
        ans = jev.ask(state, {"intent": choice_q(PRIMER + " Which movement did the participant imagine in query_trial?",
                                                 {c: None for c in classes})})
        a = ans["intent"]
        rows.append({"i": int(qi), "g": int(s), "y": y[qi], "pred": a["choice"], "p_true": a["probabilities"].get(y[qi], 0),
                     "conf": a.get("confidence"), "tokens": ans["_usage"]["input_tokens"]})
pred = np.array([r["pred"] for r in rows]); yy = np.array([r["y"] for r in rows])
acc = (pred == yy).mean(); bacc = balanced_accuracy_score(yy, pred)
print(f"\n{which} few-shot k={k}/class mode={mode}: acc={acc:.3f} bal_acc={bacc:.3f} chance={1/len(classes):.3f} "
      f"mean p(true)={np.mean([r['p_true'] for r in rows]):.3f} n={len(rows)} tokens/call≈{np.mean([r['tokens'] for r in rows]):.0f}")
for s in np.unique(g):
    m = np.array([r["g"] == s for r in rows]); print(f"  subj {s}: acc={(pred[m]==yy[m]).mean():.3f}")
print(confusion_matrix(yy, pred, labels=classes))
print(f"cumulative cost ${jev.cost_usd:.3f} | cache hits {jev.cache_hits} | mock={jev.mock}")
json.dump({"acc": acc, "bal_acc": bacc, "k": k, "mode": mode, "rows": rows},
          open(f"results/few_shot_{which}_k{k}_{mode}.json", "w"))
