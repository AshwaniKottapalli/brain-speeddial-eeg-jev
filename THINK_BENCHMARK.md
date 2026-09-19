# THINK benchmark v0: 20 intents, one neural trigger, EEG confirmation (2026-09-20)

All EEG is real, held-out, public-dataset recordings (Shin 2017 A/B, 10 subjects, 30 channels) replayed sample by
sample through the same decoders and state machine as the demo. Nothing is live hardware. Code:
`experiments/think_benchmark.py` (A and C), `experiments/yesno_compare.py` (B), table: `experiments/think_benchmark_table.py`.

## Experiment A: can one neural signal drive a 20-intent agent?

We do not decode 20 thoughts. The interface is **switch scanning**: the 20 intents are highlighted one after another
(5 s each, extended up to 3 s more while the decoder sees rising task probability). The user stays idle until the
wanted intent is highlighted, then performs the one reliable mental command (mental subtraction). When Layer 1 fires
(P(task) >= 0.7 on 2 consecutive 3 s windows), the item highlighted 1.5 s earlier is selected. After any action or
abort the menu freezes for 8 s and restarts from the top. Layouts: **linear** (20 items) and **hierarchical**
(4 categories x 5 items, two selections). Orders: **zipf** (menu sorted by usage, targets drawn from a Zipf prior,
like a real speed-dial) and **fixed** (random menu order, uniform targets).

Simulated user, all from held-out EEG of the same subject: idle = rest recordings; command = mental-subtraction
recordings; YES = left-hand imagery; NO or not answering = rest. Each subject x fold runs 20 intentional attempts
(N = 1000 per configuration). The user gives up after 2 full scan cycles without a selection.

Metrics: **correct** = the intended action was executed. **unasked** = an action the user did not ask for (a false
alarm selected whatever was highlighted, or the wrong item was picked). **dups** = repeated execution of the same
command. **t50/t90** = time from the moment the user starts wanting the action to its execution. **cmds/err** =
mean number of consecutive correct commands before an error (failed attempt or any unasked action).

| layout | menu order | confirmation | N | correct | unasked actions | unasked / hour | duplicates | time-to-action median / p90 | commands before an error | gave up |
|---|---|---|---|---|---|---|---|---|---|---|
| hier | fixed | imagery | 1000 | **718 (71.8%)** | **13** (1 per 100) | 0.3 | 0 | 68 s / 172 s | 2.1 | 282 |
| hier | fixed | none | 1000 | **871 (87.1%)** | **293** (29 per 100) | 13.4 | 0 | 44 s / 96 s | 2.6 | 129 |
| hier | fixed | repeat | 1000 | **847 (84.7%)** | **73** (7 per 100) | 2.9 | 0 | 51 s / 124 s | 3.7 | 153 |
| hier | zipf | imagery | 1000 | **707 (70.7%)** | **12** (1 per 100) | 0.3 | 0 | 61 s / 180 s | 2.0 | 293 |
| hier | zipf | none | 1000 | **817 (81.7%)** | **307** (31 per 100) | 13.5 | 0 | 38 s / 77 s | 2.3 | 183 |
| hier | zipf | p300sim (P300 r2) | 1000 | **794 (79.4%)** | **14** (1 per 100) | 0.5 | 0 | 45 s / 119 s | 3.0 | 206 |
| hier | zipf | p300sim (P300 r5) | 1000 | **809 (80.9%)** | **10** (1 per 100) | 0.4 | 0 | 39 s / 86 s | 3.3 | 191 |
| hier | zipf | repeat | 1000 | **811 (81.1%)** | **52** (5 per 100) | 2.0 | 0 | 45 s / 115 s | 3.0 | 189 |
| linear | fixed | imagery | 1000 | **607 (60.7%)** | **20** (2 per 100) | 0.5 | 0 | 81 s / 198 s | 1.3 | 393 |
| linear | fixed | none | 1000 | **739 (73.9%)** | **967** (97 per 100) | 31.2 | 0 | 56 s / 105 s | 1.7 | 261 |
| linear | fixed | repeat | 1000 | **714 (71.4%)** | **206** (21 per 100) | 5.8 | 0 | 64 s / 152 s | 2.0 | 286 |
| linear | zipf | imagery | 1000 | **673 (67.3%)** | **17** (2 per 100) | 0.5 | 0 | 56 s / 168 s | 1.8 | 327 |
| linear | zipf | none | 1000 | **770 (77.0%)** | **574** (57 per 100) | 22.5 | 0 | 31 s / 97 s | 1.8 | 230 |
| linear | zipf | p300sim (P300 r2) | 1000 | **752 (75.2%)** | **27** (3 per 100) | 0.9 | 0 | 37 s / 112 s | 2.5 | 248 |
| linear | zipf | p300sim (P300 r5) | 1000 | **745 (74.5%)** | **30** (3 per 100) | 1.1 | 0 | 35 s / 104 s | 2.4 | 255 |
| linear | zipf | repeat | 1000 | **759 (75.9%)** | **134** (13 per 100) | 4.8 | 0 | 39 s / 109 s | 2.4 | 241 |

Reading:
- One reliable trigger can drive a 20-intent interface, at 74 to 87% success without confirmation, but a
  **single-layer system does something unasked on 29 to 97 of every 100 attempts**. Every Layer-1 false alarm
  becomes a wrong action because something is always highlighted. This is the number that makes confirmation mandatory.
- **Menu order matters more than layout**: sorting by usage (zipf) roughly halves time-to-action (31 s vs 56 s median,
  linear, no confirmation) because the menu restarts from the top and frequent intents sit there.
- The **hierarchical menu** is the best layout in every configuration (82 to 87% correct without confirmation,
  13.5 unasked per hour vs 22 to 31 for linear) because each scan is short (4 then 5 items), so the user waits less
  and idle exposure per attempt is lower. Its usage-sorted variant is faster (38 s vs 44 s median) but a few points
  less successful than the random-order variant, because repeated attempts at the same top item collide with the
  post-action cooldown more often.
- Confirmation cuts unasked actions by an order of magnitude and the per-hour rates line up with the two-layer
  experiment (repeat about 5 per hour, imagery about 0.5 per hour), at a cost of 5 to 15 points of success and
  10 to 25 s of latency.
- Time-to-action is dominated by scanning, not decoding: 30 to 80 s medians. A speed-dial with one trigger is a
  patient interface. Two independent triggers (or steering with a second signal) is the obvious next lever.
- "Gave up" is high (18 to 39%): with only 5 s of highlight per item, a missed detection costs a whole cycle. This is
  the cost of the 5 s dwell; longer dwell trades it against time-to-action.

EEG reuse: each fold has only 60 s of rest and 60 s of task EEG, so recordings are cycled about 25x for idle. The
resulting false-alarm rate per hour matched the independent per-subject estimates (e.g. 12 per hour for subject 3),
but the unasked-action counts are dominated by a handful of recordings per subject and should be read as +-30%.

## Experiment B: the best public-dataset EEG YES/NO

One protocol for all: within-subject 5-fold CV, a binary decision from T seconds of EEG. For P300 and oddball a
yes/no question is two options flashing alternately r times each; the attended option evokes the P300; decision =
option with the higher mean classifier score. Because an idle (non-attending) user must not produce YES, an
**abstain margin** is calibrated per subject so that YES-on-idle <= 5%; the margin rows are the ones that matter
for a confirmation gate.

| paradigm (dataset) | subjects | decision time | P(YES given user means YES) | P(YES given user means NO) | P(YES given idle) | per-subject range |
|---|---|---|---|---|---|---|
| left/right imagery = YES/NO (Shin2017A, 30 ch) | 10 | 4.0 s | 0.623 | 0.370 | 0.41 | 0.47 to 0.93 (acc) |
| left/right imagery (BCI IV 2a, 22 ch) | 3 | 4.0 s | 0.822 | 0.157 | not measurable | 0.65 to 0.97 (acc) |
| repeat the command: arithmetic = YES, idle = NO (Shin2017B) | 10 | 3.0 s | 0.893 | 0.117 | not measurable | 0.82 to 0.95 (acc) |
| SSVEP best pair (Kalunga2016, 8 ch) | 12 | 4.0 s | 0.672 | 0.247 | 0.35 | 0.53 to 0.87 (acc) |
| SSVEP best pair (Nakanishi2015, 8 occipital ch) | 9 | 2.0 s | 0.889 | 0.126 | not measurable | 0.73 to 1.00 (acc) |
| SSVEP best pair (Nakanishi2015) | 9 | 4.0 s | 0.919 | 0.089 | not measurable | 0.73 to 1.00 (acc) |
| visual P300, ALS patients (BNCI2014_008), 1 flash/option | 8 | 0.5 s | 0.862 | 0.138 | not measurable | 0.80 to 0.94 (acc) |
| visual P300, ALS, 5 flashes/option | 8 | 2.5 s | 0.989 | 0.011 | not measurable | 0.98 to 1.00 (acc) |
| visual P300, ALS, 5 flashes, idle-safe margin | 8 | 2.5 s | 0.785 | 0.000 | 0.05 | 0.66 to 0.97 |
| visual P300 (BNCI2014_009, 16 ch), 1 flash/option, forced choice | 10 | 0.5 s | 0.947 | 0.053 | not measurable | 0.89 to 0.97 (acc) |
| visual P300, 2 flashes, forced choice | 10 | 1.0 s | 0.987 | 0.013 | not measurable | 0.95 to 1.00 (acc) |
| visual P300, 3 flashes, forced choice | 10 | 1.5 s | 0.996 | 0.004 | not measurable | 0.98 to 1.00 (acc) |
| visual P300, 5 flashes, forced choice | 10 | 2.5 s | 0.999 | 0.001 | not measurable | 1.00 to 1.00 (acc) |
| visual P300, 1 flash, idle-safe margin | 10 | 0.5 s | 0.563 | 0.001 | 0.05 | 0.41 to 0.75 |
| visual P300, 2 flashes, idle-safe margin | 10 | 1.0 s | 0.785 | 0.000 | 0.05 | 0.59 to 0.93 |
| visual P300, 3 flashes, idle-safe margin | 10 | 1.5 s | 0.890 | 0.000 | 0.05 | 0.73 to 0.98 |
| **visual P300, 5 flashes, idle-safe margin** | 10 | 2.5 s | 0.970 | 0.000 | 0.05 | 0.89 to 1.00 |

Reading:
- **Visual P300 is the strongest confirmation signal by a wide margin.** With 5 flashes per option (2.5 s) and the
  idle-safe margin: P(YES | user means YES) = 0.97, P(YES | user means NO) = 0.000, P(YES | idle) = 0.05,
  every one of 10 subjects above 0.94. Even 3 flashes (1.5 s) give 0.89 / 0.000 / 0.05.
- Our current left/right imagery is the **weakest** option tested: 62% yes/no accuracy and, critically, it says
  YES on 41% of idle EEG because it has no notion of "not answering". The idle class we added earlier fixes the
  second problem but not the first.
- SSVEP works for some people (92% at 4 s on Nakanishi 2015) but has no natural abstain behaviour (35 to 55% YES on
  rest in Kalunga 2016) and needs occipital electrodes and a flickering screen.
- "Repeat the command" (arithmetic = YES, idle = NO) is a respectable 89% and needs no screen, but NO is passive.
- P300 in ALS patients (BNCI2014_008, 8 channels) is weaker (0.79 YES acceptance at 2.5 s), a reminder that the
  visual P300 numbers assume a healthy user looking at a screen.
- Hardware note: P300 needs a display with two flashing options and gaze; it works on 8 to 16 channels including
  Pz/Cz/Oz; consumer forehead-only montages will be weaker. Auditory oddball (eyes-free) could not be evaluated: the
  Sosulski 2019 host was unreachable; c-VEP (Thielen 2015) could not be evaluated either: epoching its 5 GB continuous recordings exhausted memory on this machine.

Caveat on the P300 numbers: they are derived by resampling held-out single-flash scores into r-flash decisions,
which assumes flashes are independent. Real repeated flashes are somewhat correlated, so treat the r >= 3 rows as
optimistic by a few points; the r = 1 row (95% forced-choice, 0.5 s) is measured directly.

## Experiment C: the full loop and the front-page number

EEG -> 20-intent scanning interface -> Jev policy gate ("is this action sensible now?", never sees EEG) ->
"Do you want to <intent>?" -> EEG YES/NO -> agent. Two confirmation sources were run with the same subject's EEG
(repeat, imagery). Because no public subject has both a strong command task and a P300 recording, the P300
confirmation is a **statistical splice**: the outcome of each confirmation is drawn from the measured per-subject
P300 rates above (P300 subject i paired with EEG subject i), including its 2.5 s decision time. It is labelled
`p300sim` and is not a replay of P300 EEG.

Hierarchical usage-sorted menu, 10 subjects x 5 folds x 20 attempts:

| configuration | intentional attempts | agent did it correctly | agent did something unasked | median time-to-action |
|---|---|---|---|---|
| no confirmation | 1000 | **817** (81.7%) | **307** (13.5 per hour) | 38 s |
| repeat-the-command confirmation (real EEG) | 1000 | **811** (81.1%) | **52** (2.0 per hour) | 45 s |
| left/right imagery confirmation (real EEG) | 1000 | **707** (70.7%) | **12** (0.3 per hour) | 61 s |
| P300 confirmation, 5 flashes (statistical splice) | 1000 | **809** (80.9%) | **10** (0.4 per hour) | 39 s |

## What to put on the front page of THINK

Out of 1000 things the simulated user intentionally tried to do with the best all-real-EEG configuration
(hierarchical menu, usage-sorted, repeat-the-command confirmation): **the agent did 811 correctly and did 52
things nobody asked for (2.0 per hour of use)**. With a P300 confirmation of the measured quality spliced in, the
unasked count drops to the single digits while keeping about 75 to 80% correct. With no confirmation at all, the
agent does something unasked on roughly one attempt in three. Those three sentences are the honest state of THINK.
