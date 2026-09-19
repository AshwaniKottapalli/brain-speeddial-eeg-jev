"""Mock agent: executes the visible action. Nothing leaves the machine."""
from __future__ import annotations
import json, time, random
from .policy import ACTIONS

SCRIPTS = {
    "ORDER_FOOD": ["Opening food app", "Finding your favourite: Hyderabadi biryani, Paradise", "Applying saved address + payment",
                   "Order #{n} placed. ETA {eta} min"],
    "UBER_HOME": ["Opening Uber", "Pickup: current location -> Home", "UberGo, 4 min away", "Ride #{n} confirmed"],
    "PLAY_MUSIC": ["Opening Spotify", "Evening playlist", "Playing"],
}


class MockAgent:
    def __init__(self, log_path="results/demo_actions.jsonl"):
        self.log_path = log_path; self.history = []

    def act(self, action: str, meta: dict):
        n = random.randint(1000, 9999); eta = random.randint(25, 40)
        steps = [s.format(n=n, eta=eta) for s in SCRIPTS[action]]
        rec = {"t": time.time(), "action": action, "steps": steps, **meta}
        self.history.append(rec)
        with open(self.log_path, "a") as f: f.write(json.dumps(rec) + "\n")
        return steps
