"""Replay source: emits recorded EEG one sample at a time, as a headset would, following a session script.
A SimulatedUser decides what the person is 'doing' (idle / thinking the command / answering YES) and the
source draws the matching held-out recording. Every sample carries ground truth for scoring."""
from __future__ import annotations
import numpy as np
from collections import deque
from .decoder import SF, DW, TRIAL_S


class ReplaySource:
    def __init__(self, XB, yB, XA, yA, dec, task_label, n_commands=3, n_rest=6, seed=0, confirm_mode="repeat", script=None):
        rng = np.random.RandomState(seed)
        self.XB, self.XA = XB, XA
        cmd_pool = [i for i in dec.teB if yB[i] == task_label]; rest_pool = [i for i in dec.teB if yB[i] == "rest"]
        self.yes_pool = deque(rng.permutation([i for i in dec.teA if yA[i] == "yes"]).tolist())
        if script is None:
            cmds = rng.permutation(cmd_pool)[:n_commands].tolist(); rests = rng.permutation(rest_pool)[:n_rest].tolist()
            script = []
            # interleave: rest, rest, command, rest, command, ... ; always start and end idle
            ci, ri = 0, 0
            pattern = ["rest"] + sum([["command", "rest"] for _ in range(n_commands)], []) + ["rest"] * max(0, n_rest - n_commands - 1)
            for kind in pattern:
                if kind == "command" and ci < len(cmds): script.append(("command", cmds[ci])); ci += 1
                elif kind == "rest" and ri < len(rests): script.append(("rest", rests[ri])); ri += 1
        self.script = script
        self.confirm_mode = confirm_mode
        self.queue = deque()          # (sample_vector, truth_label, segment_id, t_in_segment)
        self.seg_id = -1; self.seg_ptr = 0
        self.t = 0                    # global sample counter
        self.n_channels = XB.shape[1]
        self.segment_onsets = {}      # seg_id -> (kind, onset_sample)
        self._load_next_segment()

    def _load_next_segment(self):
        self.seg_ptr_idx = getattr(self, "seg_ptr_idx", -1) + 1
        if self.seg_ptr_idx >= len(self.script): self.done = True; return
        self.done = False
        kind, idx = self.script[self.seg_ptr_idx]; self.seg_id += 1
        self.segment_onsets[self.seg_id] = (kind, self.t + len(self.queue))
        x = self.XB[idx]
        for k in range(x.shape[-1]):
            self.queue.append((x[:, k], kind, self.seg_id, k / SF))

    def user_answers_yes(self, seg_id):
        """Called by the FSM when it asks for confirmation during a GENUINE command (imagery mode only):
        the simulated user performs the YES imagery for CONFIRM_S seconds. Spliced in at the head of the stream."""
        if self.confirm_mode != "imagery": return
        j = self.yes_pool[0]; self.yes_pool.rotate(-1)
        x = self.XA[j][:, :DW]
        for k in reversed(range(DW)):
            self.queue.appendleft((x[:, k], "yes_response", seg_id, k / SF))

    def __iter__(self): return self

    def __next__(self):
        if not self.queue:
            self._load_next_segment()
            if self.done: raise StopIteration
        s = self.queue.popleft(); self.t += 1
        return s

    def total_samples(self):
        return int(len(self.script) * TRIAL_S * SF)
