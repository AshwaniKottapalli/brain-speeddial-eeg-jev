"""THINK benchmark, Experiment A (+C): a 20-intent agent interface driven by ONE reliable neural trigger.

Interface: switch-scanning. The 20 intents are highlighted one after another (dwell DWELL s each). The user stays
idle until the wanted intent is highlighted, then performs the mental command (mental arithmetic). Layer-1 fires
-> the highlighted item (latency-compensated) is selected. Optional hierarchical menu (4 categories x 5 items).
Optional confirmation (Experiment C): 'Do you want to <intent>?' answered with EEG YES/NO.

Everything neural is real held-out EEG (Shin 2017 A/B) replayed sample by sample through the same decoders and
state machine as the demo. The simulated user: idle = rest recordings; command = mental-subtraction recordings;
YES = left-hand imagery recordings; NO / not answering = rest recordings.

Metrics per subject and pooled: success (intended action executed), accidental actions (action executed that the
user did not ask for: false alarm selecting whatever was highlighted, or wrong item), duplicates, time to action,
commands until first error (run length), and the front-page number: out of N intentional attempts, how many did
the agent do correctly and how many things did it do unasked.

Usage: think_benchmark.py --subjects 1-10 --layout linear|hier --order zipf|fixed --confirm none|repeat|imagery [--dwell 5]
"""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, ".")
from collections import deque
from think_demo.decoder import load_subject, fit_decoders, SF, WIN, HOP, DW
from think_demo.intents import INTENTS, IDS
from think_demo.policy import PolicyGate

ap = argparse.ArgumentParser()
ap.add_argument("--subjects", default="1-10"); ap.add_argument("--layout", choices=["linear", "hier"], default="linear")
ap.add_argument("--order", choices=["zipf", "fixed"], default="zipf", help="zipf: intents ordered by usage and targets drawn from a Zipf prior")
ap.add_argument("--confirm", choices=["none", "repeat", "imagery", "p300sim"], default="none")
ap.add_argument("--dwell", type=float, default=5.0); ap.add_argument("--extend", type=float, default=3.0, help="hold the highlight up to this many extra seconds while P(task) >= 0.5")
ap.add_argument("--p300-key", default="p300_009_r2_margin", help="which measured P300 configuration to splice in (e.g. p300_009_r5_margin)")
ap.add_argument("--p300-acc", default="results/yesno_compare.json", help="per-subject P300 yes/no rates for the p300sim confirmation (statistical splice, other subjects)"); ap.add_argument("--tau", type=float, default=0.7); ap.add_argument("--K", type=int, default=2)
ap.add_argument("--attempts", type=int, default=20, help="intentional attempts per subject per fold-cycle (default: each intent once)")
ap.add_argument("--max-cycles", type=int, default=2, help="user gives up after this many full scan cycles without selection")
ap.add_argument("--cooldown", type=float, default=8.0); ap.add_argument("--theta", type=float, default=0.6)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-jev", action="store_true")
args = ap.parse_args()
a, b = args.subjects.split("-") if "-" in args.subjects else (args.subjects, args.subjects); subjects = list(range(int(a), int(b) + 1))
DWELL = int(args.dwell * SF); EXT = int(args.extend * SF); COOL = int(args.cooldown * SF); COMP = int(1.5 * SF)   # latency compensation: item highlighted 1.5 s before the fire (fires come 1.5-4 s after task onset)
rng = np.random.RandomState(args.seed)
zipf_p = 1 / np.arange(1, 21); zipf_p /= zipf_p.sum()
gate = PolicyGate(use_jev=not args.no_jev)
P300 = []
if args.confirm == "p300sim":
    R = json.load(open(args.p300_acc)); key = [k for k in R if k.startswith(args.p300_key)]
    P300 = [{"p_yes_given_yes": r["p_yes_given_yes"], "p_yes_given_idle": r["p_yes_given_idle"], "decision_s": r["decision_s"]} for r in R[key[0]]["per_subject"]]
    print(f"p300sim: using {key[0]} per-subject rates from {len(P300)} P300 subjects (different people than the EEG replay)")


class Pools:
    """held-out EEG pools for one subject/fold; each 10 s recording is consumed sequentially, cycling when exhausted."""
    def __init__(self, XB, yB, XA, yA, dec):
        self.rest = deque(rng.permutation([i for i in dec.teB if yB[i] == "rest"]).tolist())
        self.task = deque(rng.permutation([i for i in dec.teB if yB[i] == dec.task]).tolist())
        self.yes = deque(rng.permutation([i for i in dec.teA if yA[i] == "yes"]).tolist())
        self.XB, self.XA = XB, XA; self.cur = {}; self.consumed = {"rest": 0, "task": 0, "yes": 0}
        self.avail = {"rest": len(self.rest) * XB.shape[-1], "task": len(self.task) * XB.shape[-1], "yes": len(self.yes) * XA.shape[-1]}
    def sample(self, kind):
        """next single sample of the requested kind (rest/task/yes), streaming through whole recordings"""
        kind = "rest" if kind == "idle" else kind
        if kind not in self.cur or self.cur[kind][1] >= self.cur[kind][0].shape[-1]:
            pool = getattr(self, kind); i = pool[0]; pool.rotate(-1)
            self.cur[kind] = [(self.XA if kind == "yes" else self.XB)[i], 0]
        x, k = self.cur[kind]; self.cur[kind][1] += 1; self.consumed[kind] += 1
        return x[:, k]


def run_subject(s):
    XB, yB, XA, yA, _ = load_subject(s)
    out = []; idle_all = 0
    for fold in range(5):
        dec = fit_decoders(s, fold); pools = Pools(XB, yB, XA, yA, dec)
        order = list(range(20)) if args.order == "zipf" else rng.permutation(20).tolist()   # zipf: catalogue already usage-sorted
        targets = rng.choice(20, size=args.attempts, p=zipf_p) if args.order == "zipf" else rng.permutation(20)[:args.attempts]
        # ---- streaming state
        buf = deque(maxlen=WIN); t = 0; next_dec = WIN; streak = 0; cool_until = -1; last_p = 0.0
        highlight_t0 = 0; pos = 0            # scanning position (index into current menu)
        level, cat = 0, None                 # hierarchical menu state
        idle_time = 0                        # samples during which the user was NOT performing a command
        for target in targets:
            attempt = {"subject": s, "fold": fold, "target": IDS[target], "start": t, "actions": [], "cycles": 0}
            user_state = "idle"; cycles = 0; selected = None; done = False; started_scan_pos = pos; passes = 0
            max_len = int((args.max_cycles * 20 * args.dwell + 60) * SF)   # hard cap per attempt (user gives up)
            while not done:
                if t - attempt["start"] > max_len: attempt["outcome"] = "gave_up"; break
                # ---- what is highlighted now?
                if args.layout == "linear":
                    menu = order; want_idx = menu.index(target)
                else:
                    if level == 0: menu = [0, 1, 2, 3]; want_idx = order.index(target) // 5
                    else: menu = [order[cat * 5 + j] for j in range(5)]; want_idx = (order.index(target) % 5) if cat == order.index(target) // 5 else -1
                item = menu[pos]
                wants_now = (pos == want_idx) and t >= cool_until      # menu frozen + user idle during cooldown
                user_state = "task" if wants_now else "idle"
                # ---- one sample of EEG according to what the user is doing
                x = pools.sample(user_state); buf.append(x); t += 1
                if user_state == "idle": idle_time += 1
                # ---- advance highlight (frozen during cooldown)
                if t < cool_until: highlight_t0 = max(highlight_t0, t)
                dwell_now = DWELL + (EXT if (last_p >= 0.5 and t - highlight_t0 < DWELL + EXT) else 0)
                if t - highlight_t0 >= dwell_now:
                    if wants_now: attempt["missed_while_highlighted"] = attempt.get("missed_while_highlighted", 0) + 1
                    highlight_t0 = t; pos = (pos + 1) % len(menu)
                    if pos == 0:
                        if args.layout == "hier" and level == 1: level = 0
                        passes += 1
                        if passes >= args.max_cycles * (2 if args.layout == "hier" else 1):
                            attempt["outcome"] = "gave_up"; done = True; break
                # ---- decode every HOP
                if t >= next_dec and len(buf) == WIN:
                    next_dec += HOP
                    if t < cool_until: streak = 0; last_p = 0.0; continue
                    p = dec.p_task(np.stack(buf, 1)); last_p = p; streak = streak + 1 if p >= args.tau else 0
                    if streak >= args.K:
                        streak = 0
                        # latency-compensated selection: item highlighted COMP samples ago
                        back = t - COMP; sel_pos = pos if back >= highlight_t0 else (pos - 1) % len(menu)
                        sel_item = menu[sel_pos]; genuine_fire = (sel_pos == want_idx)
                        if os.environ.get("DEBUG"): print(f"  fire t={t/SF:7.1f} pos={pos} want={want_idx} sel={sel_pos} hl_t0={highlight_t0/SF:7.1f} back={back/SF:7.1f} user={user_state} p={p:.2f} genuine={genuine_fire}")
                        if args.layout == "hier" and level == 0:
                            cat = sel_item if sel_item < 4 else 0
                            if genuine_fire: level = 1; pos = 0; highlight_t0 = t
                            else:
                                # wrong category selected: user notices, stays idle through the sub-menu -> count as one wasted pass
                                attempt["actions"].append({"kind": "wrong_category", "t": t}); level = 1; pos = 0; highlight_t0 = t
                            cool_until = t + COOL; continue
                        # ---- leaf selection: policy gate + optional confirmation
                        intent_id, label, cost = INTENTS[sel_item]
                        gd = gate.decide_generic(label, cost)
                        confirmed = True
                        if args.confirm == "p300sim" and gd["decision"] != "ignore":
                            pr = P300[(s - 1) % len(P300)]     # statistical splice: P300 subject i paired with EEG subject s
                            t += int(pr["decision_s"] * SF)
                            confirmed = rng.rand() < (pr["p_yes_given_yes"] if genuine_fire else pr["p_yes_given_idle"])
                        elif args.confirm != "none" and gd["decision"] != "ignore":
                            # user answers YES only if this is what they wanted; otherwise they do nothing (rest)
                            need = WIN if args.confirm == "repeat" else DW
                            seg = np.stack([pools.sample(("task" if args.confirm == "repeat" else "yes") if genuine_fire else "rest") for _ in range(need)], 1)
                            t += need; idle_time += 0 if genuine_fire else need
                            if args.confirm == "repeat": confirmed = dec.p_task(seg[:, -WIN:]) >= args.tau
                            else: confirmed = dec.confirm_imagery(seg, args.theta)[0] == "yes"
                        if gd["decision"] == "ignore": confirmed = False
                        if confirmed:
                            attempt["actions"].append({"kind": "action", "intent": intent_id, "correct": sel_item == target, "t": t})
                            if sel_item == target and selected is None:
                                selected = t; attempt["time_to_action"] = (t - attempt["start"]) / SF
                        cool_until = t + COOL
                        level = 0; pos = 0; highlight_t0 = t + COOL      # menu restarts from the top after the cooldown
                        if sel_item == target and confirmed:
                            attempt["outcome"] = "success"; done = True
                        elif not genuine_fire:
                            pass  # accidental: scanning continues, user still waiting
                        else:
                            # genuine fire but confirmation failed: user waits for the next cycle
                            pass
            attempt["end"] = t; out.append(attempt)
        REUSE.append({k: pools.consumed[k] / pools.avail[k] for k in pools.consumed}); idle_all += idle_time
    return out, idle_all / SF


rows = []; idle_total = 0.0; REUSE = []
for s in subjects:
    r, idle_s = run_subject(s); rows += r; idle_total += idle_s
    ok = np.mean([x["outcome"] == "success" for x in r]); acc = sum(sum(a["kind"] == "action" and not a["correct"] for a in x["actions"]) for x in r)
    print(f"subject {s:2d}: attempts {len(r)} success {100*ok:5.1f}%  unasked actions {acc:3d}  median time-to-action {np.median([x['time_to_action'] for x in r if 'time_to_action' in x]) if ok else float('nan'):5.1f}s", flush=True)

# ---------------------------------------------------------------- pooled metrics
N = len(rows); succ = [x for x in rows if x["outcome"] == "success"]
unasked = sum(sum(a["kind"] == "action" and not a["correct"] for a in x["actions"]) for x in rows)
dups = sum(max(0, sum(a["kind"] == "action" and a["correct"] for a in x["actions"]) - 1) for x in rows)
tta = [x["time_to_action"] for x in succ]
total_h = sum(x["end"] - x["start"] for x in rows) / SF / 3600
# commands until first error: walk the attempt sequence per subject/fold, an error = failed attempt or any unasked action inside it
runs = []
for s in subjects:
    for f in range(5):
        seq = [x for x in rows if x["subject"] == s and x["fold"] == f]; n = 0
        for x in seq:
            bad = x["outcome"] != "success" or any(a["kind"] == "action" and not a["correct"] for a in x["actions"])
            if bad: runs.append(n); n = 0
            else: n += 1
        runs.append(n)
print(f"\nTHINK benchmark | layout={args.layout} order={args.order} confirm={args.confirm} dwell={args.dwell}s | {len(subjects)} subjects x 5 folds x {args.attempts} attempts")
print(f"  intentional attempts            : {N}")
print(f"  agent did the right thing       : {len(succ)} ({100*len(succ)/N:.1f}%)")
print(f"  agent did something unasked     : {unasked} actions ({unasked/N:.2f} per attempt, {unasked/total_h:.1f} per hour of use)")
print(f"  duplicate actions               : {dups}")
print(f"  time to action (median / p90)   : {np.median(tta):.1f}s / {np.percentile(tta, 90):.1f}s")
print(f"  commands before an error (mean) : {np.mean(runs):.1f}   (median {np.median(runs):.0f})")
print(f"  gave up (no selection in {args.max_cycles} cycles): {sum(x['outcome']=='gave_up' for x in rows)}   | times the wanted item was highlighted but not detected: {sum(x.get('missed_while_highlighted', 0) for x in rows)}")
print(f"  total simulated use             : {total_h*60:.0f} min, idle share {100*idle_total/ (total_h*3600):.0f}%")
print(f"  EEG reuse factor (consumed/available, mean over folds): rest x{np.mean([r['rest'] for r in REUSE]):.1f}, task x{np.mean([r['task'] for r in REUSE]):.1f}, yes x{np.mean([r['yes'] for r in REUSE]):.1f}")
print(f"  policy gate: {'live Jev' if gate.live else 'rule fallback'}")
json.dump({"args": vars(args), "N": N, "success": len(succ), "unasked": unasked, "duplicates": dups, "tta_median": float(np.median(tta)) if tta else None,
           "tta_p90": float(np.percentile(tta, 90)) if tta else None, "runs_mean": float(np.mean(runs)), "hours": total_h, "rows": rows},
          open(f"results/think_benchmark_{args.layout}_{args.order}_{args.confirm}{'_' + args.p300_key.split('_')[2] if args.confirm == 'p300sim' else ''}.json", "w"), default=int)
