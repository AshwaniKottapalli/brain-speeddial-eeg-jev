"""Terminal UI (rich). Shows the replayed EEG, decoder probability trace, state machine, Jev policy decision,
confirmation, and the mock agent's actions. Exports an SVG screenshot at the end."""
from __future__ import annotations
import time, numpy as np
from collections import deque
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from .fsm import ThinkFSM
from .decoder import SF
from .policy import ACTIONS

BARS = "▁▂▃▄▅▆▇█"
STATE_STYLE = {"IDLE": "dim", "GATE": "yellow", "CONFIRMING": "bold yellow", "COOLDOWN": "cyan", "DETECTED": "bold green"}


def spark(vals, n=58, lo=0.0, hi=1.0):
    vals = list(vals)[-n:]
    return "".join(BARS[min(7, int((v - lo) / (hi - lo + 1e-9) * 7.999))] for v in vals)


def run_ui(dec, src, gate, agent, args, ctx, ch):
    console = Console(record=True)
    eeg_hist = {c: deque(maxlen=58) for c in ("AFF5h", "Cz", "P7")}
    events = deque(maxlen=12); agent_lines = deque(maxlen=8); thought = Text("")
    truth_now = {"label": "rest"}

    def on_event(e):
        k = e["kind"]
        if k == "detected": msg = f"🧠 intent detected  P(task)={e.get('p')}  [{'genuine' if e.get('truth')=='command' else 'FALSE ALARM: user idle'}]"
        elif k == "gate": msg = f"⚖️  Jev policy ({e.get('source')}): {str(e.get('decision')).upper()}  p_sensible={e.get('p_sensible')}"
        elif k == "confirm_requested": msg = f"❓ confirmation requested ({e.get('mode')}) ..."
        elif k == "confirm_result": msg = f"{'✅' if e.get('answer')=='yes' else '❌'} confirmation: {str(e.get('answer')).upper()}  " + " ".join(f"{k2}={v}" for k2, v in e.items() if k2 not in ('t','kind','answer'))
        elif k == "action": msg = "🤖 ACTION EXECUTED " + ("(genuine, latency %.1fs)" % e["latency_s"] if e.get("latency_s") is not None else "(ACCIDENTAL: no command was intended)")
        elif k == "suppressed_by_cooldown": return
        else: msg = {"aborted": "↩️  aborted, back to idle", "ignored_by_policy": "🚫 ignored by policy", "idle": "… idle"}.get(k, k)
        events.append(f"[dim]{e['t']:6.1f}s[/] {msg}")
        if k == "action":
            for s_ in e["steps"]: agent_lines.append(f"  {ACTIONS[fsm.action]['emoji']} {s_}")

    fsm = ThinkFSM(dec, src, gate, agent, tau=args.tau, K=args.K, confirm_mode=args.confirm, context=ctx, on_event=on_event, cooldown_s=args.cooldown)
    cidx = {c: ch.index(c) for c in eeg_hist if c in ch}

    def render():
        lay = Layout()
        head = Text.assemble(("  THINK  ", "bold white on magenta"), ("  think a command → EEG → intent → Jev policy → confirm → agent acts   ", "bold"),
                             ("  ⚠ PUBLIC-DATASET EEG REPLAY (Shin 2017, subject %d) — NOT LIVE HARDWARE" % args.subject, "bold red"))
        # what the (simulated) user is doing right now
        doing = {"rest": "😐 idle (rest recording)", "command": "🧠 thinking the command: mental subtraction = ORDER FOOD", "yes_response": "🧠 answering YES (left-hand imagery)"}[truth_now["label"]]
        eeg = Table.grid(padding=(0, 1)); eeg.add_column(justify="right"); eeg.add_column()
        for c, h in eeg_hist.items():
            eeg.add_row(f"[dim]{c:>6}[/]", spark(h, lo=-40, hi=40) if h else "")
        eeg.add_row("[bold]P(task)[/]", "[green]" + spark(fsm.p_hist) + "[/]" if fsm.p_hist else "")
        eeg.add_row("", f"[dim]threshold τ={args.tau}  streak K={args.K}   now P(task)=[/][bold]{fsm.p_now:.2f}[/]  [dim]streak[/] {fsm.streak}")
        st = Text.assemble(("STATE  ", "dim"), (f" {fsm.state} ", STATE_STYLE.get(fsm.state, "bold")), ("     ", ""), (f"t = {fsm.t/SF:6.1f} s", "dim"),
                           ("     speed ×%.1f" % args.speed, "dim"), ("\nuser:  ", "dim"), (doing, ""))
        left = Group(Panel(eeg, title="EEG stream (30 ch, 128 Hz) → Layer-1 decoder: mental arithmetic vs rest", border_style="blue"),
                     Panel(st, title="state machine", border_style="magenta"))
        pol = gate.last or {}
        polt = Text.assemble(("Jev sees: action + cost + context. It never sees EEG or probabilities.\n", "dim"),
                             (f"context: {ctx}\n", ""), (f"last decision: {pol.get('decision','-').upper()}  p_sensible={pol.get('p_sensible','-')}  source={pol.get('source','-')}", "bold"))
        conf = fsm.last_confirm
        conft = Text(f"mode: {args.confirm}   last: {conf[0].upper() if conf else '-'} {conf[1] if conf else ''}")
        right = Group(Panel(polt, title="Jev policy / context gate (TypeSafe Jev, %s)" % ("LIVE API" if gate.live else "rule fallback"), border_style="yellow"),
                      Panel(conft, title="Layer-2 confirmation", border_style="green"),
                      Panel(Text("\n".join(agent_lines) or "  (no action yet)"), title="🤖 mock agent", border_style="cyan"))
        n_gen = len({a["segment"] for a in fsm.actions if a["genuine"]}); n_acc = sum(not a["genuine"] for a in fsm.actions)
        n_dup = sum(a["genuine"] for a in fsm.actions) - n_gen
        n_cmd_seen = sum(1 for sid, (k, on) in src.segment_onsets.items() if k == "command" and on <= fsm.t)
        foot = Text.assemble(("commands so far ", "dim"), (f"{n_cmd_seen}", "bold"), ("   completed ", "dim"), (f"{n_gen}", "bold green"), ("   accidental ", "dim"), (f"{n_acc}", "bold red"), ("   duplicate ", "dim"), (f"{n_dup}", "bold red"),
                             ("   median latency ", "dim"), (f"{np.median([a['latency_s'] for a in fsm.actions if a['latency_s']]) if n_gen else float('nan'):.1f}s", "bold"))
        lay.split_column(Layout(Panel(head), size=3), Layout(name="body"), Layout(Panel(Text.from_markup("\n".join(events)) if events else Text(""), title="event log"), size=14), Layout(Panel(foot), size=3))
        lay["body"].split_row(Layout(left, ratio=3), Layout(right, ratio=2))
        return lay

    t0 = time.time(); n = 0
    with Live(render(), console=console, refresh_per_second=8, screen=False) as live:
        for sample, truth, seg, t in src:
            truth_now["label"] = truth
            for c, i in cidx.items():
                if n % 16 == 0: eeg_hist[c].append(float(sample[i]) * 1e6)
            fsm.step(sample, truth, seg, t); n += 1
            if n % 16 == 0:
                live.update(render())
                target = t0 + n / SF / args.speed; dt = target - time.time()
                if dt > 0: time.sleep(dt)
        live.update(render()); time.sleep(0.5)
    console.save_svg(args.screenshot, title="THINK demo (public-dataset EEG replay)")
    print(f"\nsaved screenshot {args.screenshot}")
    print("actions:", fsm.actions)
