"""Jev gate v2: two atomic Noul questions (docs' composite-scoring pattern), combined in code.
  genuine : P(the decoder evidence reflects a genuine intended command rather than a false alarm)
  sensible: P(the proposed action is sensible in the current context)
Policy in code: fire if genuine>=g_hi and sensible>=s_hi; confirm if genuine>=g_lo; else ignore.
Reports the ROC-style quality of Jev's 'genuine' probability against ground truth, vs the decoder's own probability.
"""
import sys, json, numpy as np
sys.path.insert(0, ".")
from eeg_jev.jev import JevClient
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

rows0 = json.load(open(sys.argv[1]))  # jevgate_*.json from v1 (same candidates + contexts)
R = json.load(open(sys.argv[1].replace("jevgate_", "speeddial_")))
ACTIONS = {"left_hand": "book_uber_home", "right_hand": "order_favourite_food", "feet": "call_partner", "hands": "play_music"}
# rebuild recent-window evidence for each candidate
PS = {r["subject"]: (np.array(r["P"]), r["classes"]) for r in R["per_subject_P"]}
jev = JevClient()
out = []
for c in tqdm(rows0):
    P, classes = PS[c["subject"]]; w = c["w"]
    recent = np.round(P[max(0, w - 3):w + 1], 2).tolist()
    action = ACTIONS[c["cmd"]]
    evidence = {"class_order": classes, "recent_window_probabilities_oldest_first": recent, "top_probability_now": round(c["p"], 2),
                "margin_over_runner_up": round(c["margin"], 2), "consecutive_windows_agreeing": c["streak"],
                "this_users_calibration_accuracy": round(c["cal_acc"], 2), "chance_level": round(1 / len(classes), 2),
                "note": "false alarms typically show probability barely above 0.5, small margin, and flip between classes across windows"}
    qs = {"genuine": {"type": "noul", "instructions": "The EEG decoder evidence reflects a genuine intended mental command by the user, not a false alarm during idle.",
                      "criteria": {"true": "High, stable probability across consecutive windows with a clear margin", "false": "Marginal, unstable or flip-flopping evidence typical of idle noise"}},
          "sensible": {"type": "noul", "instructions": f"Performing '{action}' right now makes sense for the user given the context.",
                       "criteria": {"true": "The action is useful and not redundant in this situation", "false": "The action is redundant, already satisfied, or was just triggered"}}}
    a = jev.ask({"proposed_action": action, "decoder_evidence": evidence, "context": c["ctx"]}, qs)
    out.append({**c, "p_genuine": a["genuine"]["noul"], "p_sensible": a["sensible"]["noul"]})
real = np.array([r["truth"] == r["cmd"] for r in out]); impl = np.array([r["implausible"] for r in out])
pg = np.array([r["p_genuine"] for r in out]); ps = np.array([r["p_sensible"] for r in out]); pdec = np.array([r["p"] for r in out])
streak = np.array([r["streak"] for r in out]); margin = np.array([r["margin"] for r in out])
print(f"\nAUC for separating genuine commands from false alarms (n={len(out)}, {real.sum()} genuine):")
print(f"  Jev p_genuine            : {roc_auc_score(real, pg):.3f}")
print(f"  decoder top probability  : {roc_auc_score(real, pdec):.3f}")
print(f"  decoder margin           : {roc_auc_score(real, margin):.3f}")
print(f"  decoder streak length    : {roc_auc_score(real, streak):.3f}")
print(f"  simple code rule p*margin*streak : {roc_auc_score(real, pdec*margin*np.minimum(streak,3)):.3f}")
print(f"AUC for context plausibility (Jev p_sensible vs simulated ground truth): {roc_auc_score(~impl, ps):.3f}")
# operating point: fire if pg>=0.7 & ps>=0.6, confirm if pg>=0.4 else ignore
fire = (pg >= 0.7) & (ps >= 0.6); confirm = ~fire & (pg >= 0.4); ignore = ~(fire | confirm)
print(f"Composite policy: fire={fire.sum()} (on false alarms {(fire & ~real).sum()}, on real {(fire & real).sum()}), "
      f"confirm={confirm.sum()} (real {(confirm & real).sum()}), ignore={ignore.sum()} (real commands lost {(ignore & real).sum()}/{real.sum()})")
print(f"cost this run ${jev.cost_usd:.3f}")
json.dump(out, open(sys.argv[1].replace("jevgate_", "jevgate2_"), "w"))
