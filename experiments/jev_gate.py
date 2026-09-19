"""Jev as the decision gate of the brain speed-dial.

Input to Jev per candidate trigger (a window where the decoder's top class is a command):
  - decoder evidence: last 4 windows of class probabilities, streak length, margin, this user's calibration accuracy
  - context: local time, location, minutes since last trigger, whether the action is already satisfied
Jev answers Choice{fire, confirm, ignore} with probabilities + confidence.
Ground-truth policy for scoring:  fire    = real command AND context plausible
                                  confirm = real command AND context implausible (e.g. 'book uber home' while at home)
                                  ignore  = decoder false alarm (truth = rest or another command)
Compared against threshold-only gating at matched detection rate.
Usage: jev_gate.py <results/speeddial_*.json> [max_candidates]
"""
import sys, json, numpy as np
sys.path.insert(0, ".")
from eeg_jev.jev import JevClient, choice_q
from tqdm import tqdm

fn = sys.argv[1]; max_c = int(sys.argv[2]) if len(sys.argv) > 2 else 400
R = json.load(open(fn)); WIN_S, HOP_S, TRIAL_S = 2.0, 0.5, 4.0
ACTIONS = {"left_hand": "book_uber_home", "right_hand": "order_favourite_food", "feet": "call_partner", "hands": "play_music"}
rng = np.random.RandomState(0)

def sample_context(action):
    hour = int(rng.choice(range(24))); loc = str(rng.choice(["home", "office", "restaurant", "gym", "in_transit"]))
    since = int(rng.choice([1, 3, 10, 45, 180]))
    implausible = (action == "book_uber_home" and loc == "home") or (action == "order_favourite_food" and loc == "restaurant") \
                  or (since <= 3) or (action == "book_uber_home" and loc == "in_transit")
    return {"local_time": f"{hour:02d}:{int(rng.choice([0,15,30,45])):02d}", "location": loc,
            "minutes_since_last_trigger": since, "already_satisfied": bool(action == "book_uber_home" and loc == "home")}, implausible

cands = []
for r in R["per_subject_P"]:
    P = np.array(r["P"]); classes = r["classes"]; labels = np.array(r["labels"]); t_end = np.array(r["t_end"])
    sf = 128
    streak = 0; last = None
    for w in range(len(P)):
        j = P[w].argmax(); c = classes[j]
        if c != "rest": streak = streak + 1 if c == last else 1; last = c
        else: streak = 0; last = None
        if c == "rest" or P[w, j] < 0.5: continue
        truth = labels[min(t_end[w] - 1, len(labels) - 1)]
        cands.append({"subject": r["subject"], "w": w, "cmd": c, "p": float(P[w, j]), "streak": streak,
                      "margin": float(np.sort(P[w])[-1] - np.sort(P[w])[-2]),
                      "recent": np.round(P[max(0, w - 3):w + 1], 2).tolist(), "classes": classes, "truth": truth,
                      "cal_acc": float(R["window_acc"][[x["subject"] for x in R["per_subject_P"]].index(r["subject"])])})
rng.shuffle(cands); cands = cands[:max_c]
print(f"candidate triggers: {len(cands)} | true-command share: {np.mean([c['truth']==c['cmd'] for c in cands]):.2f}")

jev = JevClient()
INSTR = ("You are the safety gate of a brain-computer-interface 'speed dial'. A personal EEG decoder proposes an action "
         "whenever it thinks the user imagined the matching mental command. Decoders are noisy: during idle periods they "
         "produce false alarms. Decide whether to FIRE the action immediately, ask the user to CONFIRM (e.g. via a jaw clench), "
         "or IGNORE this proposal. Weigh the decoder evidence (probability, margin over the runner-up, how many consecutive "
         "windows agreed, this user's calibration accuracy) and whether the action makes sense in context.")
CRIT = {"fire": "Strong, consistent decoder evidence AND the action is sensible right now.",
        "confirm": "Evidence is decent but the action is odd in this context, or is irreversible/costly and evidence is only moderate.",
        "ignore": "Weak or inconsistent decoder evidence; probably a false alarm."}
rows = []
for c in tqdm(cands):
    action = ACTIONS[c["cmd"]]; ctx, implausible = sample_context(action)
    state = {"proposed_action": action, "decoder": {"class_order": c["classes"], "recent_window_probabilities": c["recent"],
             "top_probability": round(c["p"], 2), "margin_over_runner_up": round(c["margin"], 2), "consecutive_agreeing_windows": c["streak"],
             "this_users_calibration_accuracy": round(c["cal_acc"], 2), "chance_level": round(1 / len(c["classes"]), 2)},
             "context": ctx, "action_cost": "irreversible, costs money" if action in ("book_uber_home", "order_favourite_food") else "reversible"}
    a = jev.ask(state, {"decision": choice_q(INSTR, CRIT)})["decision"]
    real = c["truth"] == c["cmd"]
    gt = "ignore" if not real else ("confirm" if implausible else "fire")
    rows.append({**{k: v for k, v in c.items() if k not in ("recent", "classes")}, "ctx": ctx, "implausible": implausible,
                 "gt": gt, "jev": a["choice"], "p_fire": a["probabilities"]["fire"], "p_ignore": a["probabilities"]["ignore"], "conf": a["confidence"]})

gt = np.array([r["gt"] for r in rows]); jd = np.array([r["jev"] for r in rows]); real = np.array([r["truth"] == r["cmd"] for r in rows])
p = np.array([r["p"] for r in rows])
print(f"\nJev decision accuracy vs policy: {(gt == jd).mean():.3f}")
from sklearn.metrics import confusion_matrix
print("rows=policy, cols=Jev, order [fire, confirm, ignore]\n", confusion_matrix(gt, jd, labels=["fire", "confirm", "ignore"]))
# safety metrics: a 'fire' on a false alarm = bad; 'fire'/'confirm' on a real command = detection
fa_fire_jev = ((jd == "fire") & ~real).sum(); det_jev = ((jd != "ignore") & real).sum()
print(f"Jev gate  : fires on false alarms={fa_fire_jev}/{(~real).sum()}  | real commands passed (fire or confirm)={det_jev}/{real.sum()} "
      f"| fired outright on implausible-context real commands={((jd=='fire') & real & np.array([r['implausible'] for r in rows])).sum()}")
# threshold-only baseline matched to the same pass rate
target = det_jev / real.sum()
taus = np.linspace(0.5, 0.99, 50); best = None
for t in taus:
    passed = p >= t
    if (passed & real).sum() / real.sum() >= target: best = t
print(f"Threshold : at tau={best:.2f} giving the same real-command pass rate, fires on false alarms={((p >= best) & ~real).sum()}/{(~real).sum()} "
      f"(threshold has no notion of context, so all passes fire outright)")
print(f"cost this run ${jev.cost_usd:.3f}")
json.dump(rows, open(fn.replace("speeddial_", "jevgate_"), "w"), indent=0)
