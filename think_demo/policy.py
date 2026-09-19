"""Jev as the CONTEXT / POLICY gate only. The experiments (SPEEDDIAL_REPORT.md s5) showed Jev is worse than a
threshold at judging decoder evidence but useful (AUC 0.70) at judging whether an action is sensible in context.
So Jev never sees EEG or probabilities here. It sees: the proposed action, its cost, and the situation."""
from __future__ import annotations
import os, sys
sys.path.insert(0, ".")
from eeg_jev.jev import JevClient

ACTIONS = {
    "ORDER_FOOD": {"label": "order your favourite food", "cost": "irreversible, costs money", "emoji": "🍜"},
    "UBER_HOME": {"label": "book an Uber home", "cost": "irreversible, costs money", "emoji": "🚗"},
    "PLAY_MUSIC": {"label": "play your evening playlist", "cost": "reversible, free", "emoji": "🎵"},
}


class PolicyGate:
    def __init__(self, use_jev=True):
        self.jev = JevClient() if use_jev else None
        self.live = bool(self.jev and not self.jev.mock)
        self.last = None

    def decide(self, action: str, context: dict) -> dict:
        """returns {'decision': fire|confirm|ignore, 'p_sensible': float|None, 'source': 'jev'|'rule'}"""
        meta = ACTIONS[action]
        if self.live:
            state = {"proposed_action": meta["label"], "action_cost": meta["cost"], "context": context}
            q = {"sensible": {"type": "noul",
                              "instructions": f"Performing '{meta['label']}' right now makes sense for the user given the context.",
                              "criteria": {"true": "The action is useful and not redundant in this situation",
                                           "false": "The action is redundant, already satisfied, or was just triggered"}}}
            p = self.jev.ask(state, q)["sensible"]["noul"]; src = "jev"
        else:
            # rule fallback (clearly labelled in the UI) mirroring the same policy
            p = 0.15 if context.get("minutes_since_last_same_action", 999) < 20 else 0.85; src = "rule"
        if p < 0.3: dec = "ignore"
        elif "money" in meta["cost"]: dec = "confirm"
        else: dec = "fire"
        self.last = {"decision": dec, "p_sensible": round(float(p), 2), "source": src, "action": action, "context": context}
        return self.last
