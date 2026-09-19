"""Thin wrapper around the TypeSafe SDK: disk cache, token/cost accounting, mock mode, live-key check."""
from __future__ import annotations
import hashlib, json, os, time
from dotenv import load_dotenv

load_dotenv()
PRICE_PER_M_TOKENS = 0.042  # USD, input tokens only (output free)
CACHE_DIR = "cache/jev"
os.makedirs(CACHE_DIR, exist_ok=True)


class JevClient:
    def __init__(self, mock: bool | None = None, budget_usd: float = 4.5, model: str | None = None):
        model = model or os.environ.get("JEV_MODEL", "jev-latest")
        key = os.environ.get("TYPESAFE_API_KEY")
        self.mock = (not key) if mock is None else mock
        self.model = model
        self.budget_usd = budget_usd
        self.tokens = 0
        self.calls = 0
        self.cache_hits = 0
        if not self.mock:
            from typesafe_sdk import TypeSafeClient, RetryPolicy
            self.client = TypeSafeClient(model=model, timeout=60.0,
                                         retry=RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=20.0))
        # load persisted usage
        self._usage_fn = os.path.join(CACHE_DIR, "usage.json")
        if os.path.exists(self._usage_fn):
            u = json.load(open(self._usage_fn)); self.tokens_total = u.get("tokens", 0)
        else:
            self.tokens_total = 0

    @property
    def cost_usd(self):
        return self.tokens_total / 1e6 * PRICE_PER_M_TOKENS

    def _key(self, state, questions):
        blob = json.dumps({"m": self.model, "s": state, "q": questions}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def ask(self, state, questions: dict):
        """questions: name -> dict(type=..., instructions=..., criteria=...). Returns dict name -> answer dict."""
        k = self._key(state, questions)
        fn = os.path.join(CACHE_DIR, k + ".json")
        if os.path.exists(fn):
            self.cache_hits += 1
            return json.load(open(fn))
        if self.mock:
            return self._mock(questions)
        if self.cost_usd > self.budget_usd:
            raise RuntimeError(f"Budget exceeded: ${self.cost_usd:.3f} > ${self.budget_usd}")
        from typesafe_sdk import Choice, Noul, Score
        qobjs = {}
        for name, q in questions.items():
            t = q["type"]
            if t == "choice":
                qobjs[name] = Choice(instructions=q.get("instructions"), criteria=q["criteria"])
            elif t == "noul":
                qobjs[name] = Noul(instructions=q.get("instructions"), criteria=q.get("criteria"))
            elif t == "score":
                qobjs[name] = Score(instructions=q.get("instructions"), criteria=q["criteria"])
        resp = self.client.system_one(state=state, questions=qobjs)
        out = {}
        for name, a in resp.answers.items():
            d = a.model_dump() if hasattr(a, "model_dump") else dict(a)
            out[name] = d
        usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
        out["_usage"] = usage
        self.tokens += usage.get("input_tokens", 0); self.tokens_total += usage.get("input_tokens", 0)
        self.calls += 1
        json.dump({"tokens": self.tokens_total}, open(self._usage_fn, "w"))
        json.dump(out, open(fn, "w"))
        return out

    def _mock(self, questions):
        import random
        out = {}
        for name, q in questions.items():
            if q["type"] == "choice":
                opts = list(q["criteria"]); ps = [random.random() for _ in opts]; s = sum(ps)
                probs = {o: p / s for o, p in zip(opts, ps)}
                out[name] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs,
                             "confidence": random.random()}
            elif q["type"] == "noul":
                out[name] = {"type": "noul", "noul": random.random()}
        out["_usage"] = {"input_tokens": len(json.dumps(questions)) // 4, "output_tokens": 0}
        return out


def choice_q(instructions, criteria: dict):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}
