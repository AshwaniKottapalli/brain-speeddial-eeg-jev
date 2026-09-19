"""State machine: IDLE -> (L1 streak) DETECTED -> GATE (Jev policy) -> CONFIRMING -> EXECUTE -> COOLDOWN -> IDLE.
Operating point from the experiments: tau 0.7, K 2 (Layer 1); repeat-confirm on the next 3 s window; imagery
confirm on the next 4 s crop with theta 0.6 and an idle class."""
from __future__ import annotations
import numpy as np
from collections import deque
from .decoder import SF, WIN, HOP, DW, CONFIRM_S, WIN_S

COOLDOWN_S = 8.0   # refractory period after an action or abort (see README: 3 s causes duplicate orders)


class ThinkFSM:
    def __init__(self, dec, source, gate, agent, action="ORDER_FOOD", tau=0.7, K=2, confirm_mode="repeat", theta=0.6,
                 context=None, on_event=None, cooldown_s=None):
        self.cooldown_s = COOLDOWN_S if cooldown_s is None else cooldown_s
        self.dec, self.src, self.gate, self.agent = dec, source, gate, agent
        self.action, self.tau, self.K, self.confirm_mode, self.theta = action, tau, K, confirm_mode, theta
        self.context = context or {"local_time": "20:40", "location": "home", "minutes_since_last_same_action": 240, "day": "Friday"}
        self.on_event = on_event or (lambda *a, **k: None)
        self.buf = deque(maxlen=WIN); self.truth_buf = deque(maxlen=WIN); self.seg_buf = deque(maxlen=WIN)
        self.state = "IDLE"; self.streak = 0; self.p_hist = deque(maxlen=60); self.t = 0
        self.next_decode = WIN; self.fire_t = None; self.fire_seg = None; self.fire_truth = None
        self.cooldown_until = -1; self.confirm_start = None; self.confirm_buf = []
        self.events = []; self.actions = []; self.p_now = 0.0; self.last_confirm = None
        self.stats = {"windows": 0}

    def _log(self, kind, **kw):
        e = {"t": self.t / SF, "kind": kind, **kw}; self.events.append(e); self.on_event(e)

    def step(self, sample, truth, seg_id, t_in_seg):
        """feed ONE sample (30-ch vector). Returns state."""
        self.buf.append(sample); self.truth_buf.append(truth); self.seg_buf.append(seg_id); self.t += 1
        if self.state == "CONFIRMING":
            self.confirm_buf.append(sample)
            need = WIN if self.confirm_mode == "repeat" else DW
            if len(self.confirm_buf) >= need:
                self._resolve_confirmation(np.stack(self.confirm_buf, 1))
            return self.state
        if self.t >= self.next_decode and len(self.buf) == WIN:
            self.next_decode += HOP
            p = self.dec.p_task(np.stack(self.buf, 1)); self.p_now = p; self.p_hist.append(p); self.stats["windows"] += 1
            if self.state == "COOLDOWN":
                if p >= self.tau: self._log("suppressed_by_cooldown", truth=truth)
                if self.t >= self.cooldown_until: self.state = "IDLE"; self._log("idle")
                return self.state
            if self.state == "IDLE":
                self.streak = self.streak + 1 if p >= self.tau else 0
                if self.streak >= self.K:
                    self.streak = 0
                    # attribute the fire to the segment (and its label) that makes up the majority of the window
                    segs = list(self.seg_buf); self.fire_seg = max(set(segs), key=segs.count); self.fire_t = self.t
                    truths = list(self.truth_buf); self.fire_truth = max(set(truths), key=truths.count)
                    self._log("detected", p=round(p, 2), truth=self.fire_truth, segment=self.fire_seg)
                    self._gate()
        return self.state

    def _gate(self):
        self.state = "GATE"
        d = self.gate.decide(self.action, self.context)
        self._log("gate", **{k: v for k, v in d.items() if k != "context"})
        if d["decision"] == "ignore":
            self.state = "COOLDOWN"; self.cooldown_until = self.t + int(self.cooldown_s * SF); self._log("ignored_by_policy")
        elif d["decision"] == "fire" or self.confirm_mode == "none":
            self._execute(confirmed_by=None)
        else:
            self.state = "CONFIRMING"; self.confirm_buf = []; self.confirm_start = self.t
            self._log("confirm_requested", mode=self.confirm_mode)
            if self.fire_truth == "command": self.src.user_answers_yes(self.fire_seg)   # simulated user answers YES (imagery mode)

    def _resolve_confirmation(self, x):
        if self.confirm_mode == "repeat":
            p = self.dec.p_task(x[:, -WIN:]); ans = "yes" if p >= self.tau else "no"; detail = {"p_task": round(p, 2)}
        else:
            ans, probs = self.dec.confirm_imagery(x[:, :DW], self.theta); detail = {k: round(float(v), 2) for k, v in probs.items()}
        self.last_confirm = (ans, detail); self._log("confirm_result", answer=ans, **detail)
        if ans == "yes": self._execute(confirmed_by=self.confirm_mode)
        else:
            self.state = "COOLDOWN"; self.cooldown_until = self.t + int(self.cooldown_s * SF); self._log("aborted", answer=ans)

    def _execute(self, confirmed_by):
        kind, onset = self.src.segment_onsets.get(self.fire_seg, ("?", self.fire_t))
        genuine = self.fire_truth == "command"
        latency = (self.t - onset) / SF if genuine else None
        steps = self.agent.act(self.action, {"genuine": genuine, "confirmed_by": confirmed_by, "latency_s": latency, "segment": self.fire_seg})
        self.actions.append({"t": self.t / SF, "genuine": genuine, "latency_s": latency, "confirmed_by": confirmed_by, "segment": self.fire_seg})
        self._log("action", genuine=genuine, latency_s=None if latency is None else round(latency, 1), steps=steps)
        self.state = "COOLDOWN"; self.cooldown_until = self.t + int(self.cooldown_s * SF)
