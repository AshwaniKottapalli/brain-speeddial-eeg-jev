# EEG -> intent -> agent: what public data says about a "brain speed-dial"

Offline experiments on public EEG datasets testing whether a few trained mental commands can drive an AI agent,
and whether TypeSafe's Jev model can sit in that loop. No hardware was used; a 4-channel "Muse-like" setup is
simulated by picking the nearest electrodes from research-grade recordings. Nothing here runs live yet.

## Results in one table

| Question | Answer | Where |
|---|---|---|
| Can Jev be fine-tuned or fed EEG directly? | No. Hosted, text-only, no weights, no LoRA. | `REPORT.md` s1 |
| Zero-shot / few-shot: EEG feature JSON -> Jev -> intent | Chance (0.26 to 0.33 vs 0.25) | `REPORT.md` s4a-b |
| Trained encoder -> 3-word physiology descriptor -> Jev | **0.737** on BCI IV 2a (best classical 0.796; random-table control 0.714) | `REPORT.md` s4c |
| Imagined words as commands (Nieto 2022, 128 ch) | Chance (0.26 to 0.31 for 4 words) | `SPEEDDIAL_REPORT.md` s2 |
| Motor imagery as commands, EEGMMIDB | Weak; unreadable from Muse-like channels | s3 |
| **Mental arithmetic vs rest as a command** (Shin 2017B, 10 subj, 30 ch) | **96% detection at 28 false triggers per idle hour**; 82% at 5 per hour | s4, s4b |
| Same, Muse-like 4 channels | 77% at 47 per hour; collapses to 39% at a 12-per-hour budget | s4b |
| Jev as the trigger's evidence gate | Worse than a plain threshold (AUC 0.58 vs 0.64); useful only for context/policy | s5 |
| Second all-EEG confirmation layer (left/right imagery as YES/NO) | Accidental actions down 75 to 96% (to about 1 per idle hour) but genuine completions fall from 96% to about 50% | s7 |

Read the two reports before quoting numbers: every headline figure comes with a false-trigger rate, and the
Muse numbers are a proxy, not a headband.

## Layout
- `eeg_jev/` data loading, feature extraction, Jev client (cached, budgeted, mock mode), baselines
- `experiments/` one script per experiment, see the reports for which produced what
- `results/` JSON outputs and the channel-ablation figure
- `REPORT.md` EEG -> Jev -> intent (can Jev learn EEG?)
- `SPEEDDIAL_REPORT.md` brain speed-dial: command detection, channel ablation, Jev gate, two-layer confirmation

## Reproduce
```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
echo TYPESAFE_API_KEY=... > .env            # only needed for the Jev experiments; scripts run in mock mode without it
.venv/bin/python experiments/prep.py bnci && .venv/bin/python experiments/baseline.py bnci
.venv/bin/python experiments/adapter.py bnci semantic_crit 3 ts jev        # EEG -> descriptor -> Jev
.venv/bin/python experiments/channel_ablation.py 1 2 3 4 5 6 7 8 9 10      # 30 ch vs Muse-4
.venv/bin/python experiments/two_layer.py 1 2 3 4 5 6 7 8 9 10             # command + EEG confirmation
```
Datasets download automatically (MNE / MOABB / OpenNeuro): PhysioNet EEGMMIDB, BCI Competition IV 2a,
Shin 2017 A and B, Nieto 2022 inner speech. Total Jev API spend for everything in the reports: about $0.35.

## License
MIT.
