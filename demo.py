#!/usr/bin/env python
"""THINK demo. Public-dataset EEG replay (Shin 2017 A/B), NOT live hardware.

  python demo.py                          live terminal demo (subject 3, 3 commands, repeat-confirm, 2x speed)
  python demo.py --confirm imagery        confirm with a separate YES/NO imagery task
  python demo.py --confirm none           single layer, no confirmation
  python demo.py --metrics                headless end-to-end metric over subjects 1-10 x 5 folds, all confirm modes
"""
import argparse, sys, time, json
sys.path.insert(0, ".")
from think_demo.decoder import load_subject, fit_decoders, SF
from think_demo.stream import ReplaySource
from think_demo.fsm import ThinkFSM
from think_demo.policy import PolicyGate
from think_demo.agent import MockAgent

ap = argparse.ArgumentParser()
ap.add_argument("--subject", type=int, default=3); ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--commands", type=int, default=3); ap.add_argument("--rests", type=int, default=5)
ap.add_argument("--confirm", choices=["none", "repeat", "imagery"], default="repeat")
ap.add_argument("--speed", type=float, default=2.0, help="replay speed multiplier (1 = real time)")
ap.add_argument("--tau", type=float, default=0.7); ap.add_argument("--K", type=int, default=2)
ap.add_argument("--no-jev", action="store_true", help="use the rule fallback instead of live Jev for the policy gate")
ap.add_argument("--context", default='{"local_time":"20:40","location":"home","minutes_since_last_same_action":240,"day":"Friday"}')
ap.add_argument("--metrics", action="store_true"); ap.add_argument("--subjects", default="1-10")
ap.add_argument("--headless", action="store_true", help="run one session without the UI, print events")
ap.add_argument("--cooldown", type=float, default=8.0, help="refractory seconds after an action or abort")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--screenshot", default="results/demo_screenshot.svg")
args = ap.parse_args()

if args.metrics:
    from think_demo.metrics import run_all
    a, b = args.subjects.split("-") if "-" in args.subjects else (args.subjects, args.subjects)
    run_all(list(range(int(a), int(b) + 1)), use_jev=not args.no_jev, tau=args.tau, K=args.K, cooldown_s=args.cooldown); sys.exit()

XB, yB, XA, yA, ch = load_subject(args.subject); dec = fit_decoders(args.subject, args.fold)
src = ReplaySource(XB, yB, XA, yA, dec, dec.task, n_commands=args.commands, n_rest=args.rests, seed=args.seed, confirm_mode=args.confirm)
gate = PolicyGate(use_jev=not args.no_jev); agent = MockAgent()
ctx = json.loads(args.context)

if args.headless:
    fsm = ThinkFSM(dec, src, gate, agent, tau=args.tau, K=args.K, confirm_mode=args.confirm, context=ctx, cooldown_s=args.cooldown,
                   on_event=lambda e: print(f"t={e['t']:6.1f}s  {e['kind']:18s} " + " ".join(f"{k}={v}" for k, v in e.items() if k not in ('t', 'kind', 'steps'))))
    print("script:", [(k, i) for k, i in src.script])
    for sample, truth, seg, t in src: fsm.step(sample, truth, seg, t)
    print("actions:", fsm.actions); sys.exit()

from think_demo.ui import run_ui
run_ui(dec, src, gate, agent, args, ctx, ch)
