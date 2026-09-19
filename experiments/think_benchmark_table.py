"""Aggregate results/think_benchmark_*.json into the THINK benchmark table."""
import glob, json, numpy as np
rows = []
for fn in sorted(glob.glob("results/think_benchmark_*.json")):
    r = json.load(open(fn)); a = r["args"]; N = r["N"]
    tag = f"{a['layout']:6s} {a['order']:5s} {a['confirm']:8s}" + (f" {fn.split('_')[-1][:-5]}" if a["confirm"] == "p300sim" else "")
    gave = sum(x["outcome"] == "gave_up" for x in r["rows"])
    rows.append((tag, N, r["success"], 100 * r["success"] / N, r["unasked"], r["unasked"] / N * 100, r["unasked"] / r["hours"], r["duplicates"], r["tta_median"], r["tta_p90"], r["runs_mean"], gave))
print(f"{'layout order confirm':28s} | {'N':>4} {'correct':>7} {'%':>6} | {'unasked':>7} {'/100':>5} {'/h':>6} | {'dups':>4} | {'t50 s':>5} {'t90 s':>5} | {'cmds/err':>8} {'gaveup':>6}")
for t in rows:
    print(f"{t[0]:28s} | {t[1]:4d} {t[2]:7d} {t[3]:6.1f} | {t[4]:7d} {t[5]:5.0f} {t[6]:6.1f} | {t[7]:4d} | {t[8] if t[8] else float('nan'):5.0f} {t[9] if t[9] else float('nan'):5.0f} | {t[10]:8.1f} {t[11]:6d}")
