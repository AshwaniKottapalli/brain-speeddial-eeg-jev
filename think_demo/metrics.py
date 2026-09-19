"""Headless end-to-end product metric: replay every held-out trial of every subject/fold through the FULL
demo pipeline (replay source -> sample-by-sample FSM -> policy gate -> confirmation -> agent) and count what a user
would experience: intentional commands completed, accidental actions during idle, latency."""
from __future__ import annotations
import json, numpy as np
from .decoder import load_subject, fit_decoders, SF, TRIAL_S
from .stream import ReplaySource
from .fsm import ThinkFSM
from .policy import PolicyGate
from .agent import MockAgent


def run_session(s, fold, confirm_mode, gate, tau=0.7, K=2, seed=0, context=None, script=None, cooldown_s=None):
    XB, yB, XA, yA, _ = load_subject(s); dec = fit_decoders(s, fold)
    rng = np.random.RandomState(seed)
    if script is None:   # all held-out trials of this fold, shuffled: the whole test fold becomes one session
        script = [("command" if yB[i] == dec.task else "rest", int(i)) for i in rng.permutation(dec.teB)]
    src = ReplaySource(XB, yB, XA, yA, dec, dec.task, confirm_mode=confirm_mode, script=script, seed=seed)
    agent = MockAgent(log_path="results/demo_actions_headless.jsonl")
    fsm = ThinkFSM(dec, src, gate, agent, tau=tau, K=K, confirm_mode=confirm_mode, context=context, cooldown_s=cooldown_s)
    for sample, truth, seg, t in src: fsm.step(sample, truth, seg, t)
    n_cmd = sum(k == "command" for k, _ in script); n_rest = sum(k == "rest" for k, _ in script)
    cmd_segs = {sid for sid, (k, _) in src.segment_onsets.items() if k == "command"}
    det_segs = {e["segment"] for e in fsm.events if e["kind"] == "detected" and e["truth"] == "command"}
    gen_actions = [a for a in fsm.actions if a["genuine"]]
    completed_segs = {a["segment"] for a in gen_actions}
    first_lat = {}
    for a in gen_actions: first_lat.setdefault(a["segment"], a["latency_s"])
    return {"subject": s, "fold": fold, "confirm": confirm_mode, "n_commands": n_cmd,
            "detected": len(det_segs & cmd_segs), "completed": len(completed_segs & cmd_segs),
            "duplicate_actions": len(gen_actions) - len(completed_segs),
            "accidental": sum(not a["genuine"] for a in fsm.actions), "idle_s": n_rest * TRIAL_S,
            "latencies": list(first_lat.values()),
            "l1_false_alarm_events": sum(e["kind"] == "detected" and e["truth"] != "command" for e in fsm.events),
            "cmd_windows_suppressed_by_cooldown": sum(e["kind"] == "suppressed_by_cooldown" and e["truth"] == "command" for e in fsm.events)}


def run_all(subjects, modes=("none", "repeat", "imagery"), use_jev=True, tau=0.7, K=2, cooldown_s=None):
    gate = PolicyGate(use_jev=use_jev)
    rows = [run_session(s, f, m, gate, tau, K, cooldown_s=cooldown_s) for m in modes for s in subjects for f in range(5)]
    out = {"gate_source": "jev-live" if gate.live else "rule-fallback", "tau": tau, "K": K, "cooldown_s": cooldown_s, "rows": rows, "summary": {}}
    print(f"\nEnd-to-end product metric | subjects {subjects} x 5 folds | L1 tau={tau} K={K} cooldown={cooldown_s}s | policy gate: {out['gate_source']}")
    print(f"{'confirmation':>12} | {'commands':>8} {'detected':>8} {'completed':>9} {'success%':>8} {'duplicates':>10} | {'idle h':>6} {'accidental':>10} {'accidental/h':>12} | {'median lat s':>12} | {'L1 FA/h':>7} {'cd-suppressed':>13}")
    for m in modes:
        r = [x for x in rows if x["confirm"] == m]
        n = sum(x["n_commands"] for x in r); det = sum(x["detected"] for x in r); c = sum(x["completed"] for x in r)
        dup = sum(x["duplicate_actions"] for x in r); acc = sum(x["accidental"] for x in r)
        idle_h = sum(x["idle_s"] for x in r) / 3600; lats = sum([x["latencies"] for x in r], [])
        l1fa = sum(x["l1_false_alarm_events"] for x in r) / idle_h; supp = sum(x["cmd_windows_suppressed_by_cooldown"] for x in r)
        out["summary"][m] = {"commands": n, "detected": det, "completed": c, "success": c / n, "duplicate_actions": dup, "idle_hours": idle_h,
                             "accidental": acc, "accidental_per_hour": acc / idle_h, "median_latency_s": float(np.median(lats)) if lats else None,
                             "l1_detect": det / n, "l1_fa_per_hour": l1fa, "cooldown_suppressed_windows": supp}
        print(f"{m:>12} | {n:8d} {det:8d} {c:9d} {100*c/n:7.1f}% {dup:10d} | {idle_h:6.2f} {acc:10d} {acc/idle_h:12.1f} | {np.median(lats) if lats else float('nan'):12.1f} | {l1fa:7.1f} {supp:13d}")
    json.dump(out, open(f"results/demo_end_to_end_metrics_cd{cooldown_s}.json", "w"), indent=1)
    return out
