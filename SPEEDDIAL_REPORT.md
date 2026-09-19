# Brain speed-dial: proof of value on public EEG data (2026-09-19)

Goal: a few trained mental commands -> detect when the user "thinks" one -> confidence -> trigger an action
(book Uber home, order food). No hardware available, so every number below is from public datasets, with a
4-channel "Muse proxy" (electrodes nearest AF7 / AF8 / TP9 / TP10) simulated from research-grade recordings.
Code: `experiments/speeddial_eval.py`, `mental_arith_eval.py`, `innerspeech_eval.py`, `jev_gate*.py`.

## 1. Headline

| Command type | Dataset | Channels | Detect rate | False triggers per idle hour | Verdict |
|---|---|---|---|---|---|
| **Mental arithmetic ("count backwards") vs rest** | Shin 2017B, 10 subj | 30 | **0.92 to 0.96** | 19 to 40 | works |
| same | same | **Muse proxy 4** | **0.66 to 0.77** | 18 to 47 | works, degraded; at a strict 12 FA/h budget 91% vs 39% |
| Motor imagery, 2 commands + rest | EEGMMIDB, 10 subj | 64 | 0.18 (best user 0.67) | 25 (best user 22) | weak, user-dependent |
| same | same | Muse proxy 4 | 0.01 | 2 | does not work |
| **Imagined words**, 4 words | Nieto 2022, 3 subj | 128 | 4-class acc 0.26 to 0.31 (chance 0.25) | | at chance |
| same, single word as trigger | same | 128 | 0.00 to 0.14 | false-fire on other words 0.01 to 0.09 | unusable |
| same | same | Muse proxy 4 | 4-class acc 0.23 to 0.36 | | at chance |

Detection = a command trial produced a correct trigger; a trigger requires K consecutive sliding windows above
a probability threshold. Latency 2 to 3 s from command onset in every configuration.

**Bottom line.** "Think a word and it happens" is not possible on any EEG, let alone a Muse. "Perform a
deliberate mental task and it happens" is possible and reaches product-grade detection on research EEG; on
Muse-like channels it still detects 3 in 4 commands, with roughly one false trigger every 2 to 5 minutes of
idle before any confirm step. The speed-dial is viable if the commands are mental tasks, the number of
commands is small (2 to 3), and irreversible actions go through a cheap confirm gesture.

## 2. Imagined words (inner speech): the honest negative result

Nieto et al. 2022 (OpenNeuro ds003626): 128-channel Biosemi, 4 Spanish words (up/down/right/left),
inner-speech condition, 180 to 240 trials per subject. Riemannian tangent-space classifier, 5-fold CV,
1.0 to 3.5 s action window.

| Subject | 128 ch, 4-40 Hz | 128 ch, 4-13 Hz | 128 ch, 13-40 Hz | Muse proxy, best band |
|---|---|---|---|---|
| 01 | 0.305 | 0.290 | 0.260 | 0.360 |
| 02 | 0.263 | 0.200 | 0.237 | 0.267 |
| 03 | 0.261 | 0.278 | 0.222 | 0.250 |

Chance is 0.25. The 0.36 is one of 18 tests and does not replicate. This matches the dataset's own paper
(about 0.29 for 4 classes). Single-word speed-dial framing ("up" vs everything) detects 0 to 14% of trials
while false-firing on 1 to 9% of other-word trials. Not a foundation for anything.

## 3. Motor imagery as commands (EEGMMIDB, 10 subjects, 64 ch, with explicit rest)

Simulated product loop: N calibration trials per command, a per-user Riemannian decoder trained on 2 s
windows, then a continuous held-out stream (commands and idle interleaved, 4 s each) decoded every 0.5 s.
Trigger = K consecutive windows with the same command above probability tau. 3 random calibration splits pooled.

2 commands + rest, 64 channels, 18 calibration trials per command (about 5 minutes of recording):

| tau | K | detect % | wrong-command % | false triggers / idle hour | latency s |
|---|---|---|---|---|---|
| 0.6 | 1 | 39.7 | 14.2 | 211 | 2.0 |
| 0.7 | 2 | 18.1 | 3.5 | 25 | 2.6 |
| 0.8 | 2 | 9.2 | 1.1 | 7 | 2.6 |
| 0.9 | 2 | 1.8 | 0.0 | 0.6 | 2.8 |

Per user at tau 0.7, K 2: detect ranges from 0% (users 8, 9) to 67% (user 7); the best 3 users average 41%
detect at 28 false triggers per idle hour. More commands made it worse (3 commands: 23% detect at 65 FA/h;
4 commands: 20% at 66 FA/h). Less calibration (8 trials) made it worse (9% at 22 FA/h). Muse proxy channels
(AF7/AF8/TP7/TP8, no central electrodes): detect 1 to 5% at any usable threshold, i.e. motor imagery is
physically not readable from Muse positions.

## 4. Mental arithmetic as a command (Shin 2017B, "count backwards" vs rest)

30 EEG channels, 10 s trials, 30 task + 30 rest trials per subject over 3 sessions. Per-user 5-fold CV,
3 s windows every 1 s, trigger = K consecutive windows with P(task) >= 0.7. FA/h = false triggers per hour of rest.

| Subject | 30 ch acc | 30 ch K2 detect / FA/h | 30 ch K3 detect / FA/h | Muse-4 acc | Muse-4 K2 detect / FA/h | Muse-4 K3 detect / FA/h |
|---|---|---|---|---|---|---|
| 1 | 0.93 | 0.90 / 24 | 0.90 / 24 | 0.73 | 0.63 / 36 | 0.57 / 24 |
| 2 | 0.89 | 0.90 / 12 | 0.83 / 0 | 0.79 | 0.77 / 36 | 0.47 / 0 |
| 3 | 0.97 | 1.00 / 12 | 1.00 / 12 | 0.91 | 0.90 / 12 | 0.83 / 0 |
| 4 | 0.93 | 1.00 / 24 | 0.97 / 0 | 0.79 | 0.90 / 36 | 0.77 / 24 |
| 5 | 0.86 | 1.00 / 48 | 0.90 / 36 | 0.68 | 0.50 / 24 | 0.43 / 12 |
| **mean** | **0.92** | **0.96 / 24** | **0.92 / 14** | **0.78** | **0.74 / 29** | **0.61 / 12** |

Muse proxy here = AFF5h, AFF6h, P7, P8 (nearest available to AF7, AF8, TP9, TP10). Every subject works.
Caveat: "rest" in this dataset is quiet fixation; real idle life (talking, walking, reading) will produce more
false alarms than this, which is why a confirm step for costly actions is not optional.

### 4b. Channel ablation, 10 subjects: 30 channels vs Muse-4, identical pipeline (`experiments/channel_ablation.py`)

Same 1-40 Hz load filter, 4-30 Hz bandpass, 128 Hz, 10 s trials, 3 s windows every 1 s, Riemannian tangent
space + logistic regression, the same 5 stratified folds per subject for both channel sets, the same K-streak
trigger (streak resets after each fire, so every false event is counted). Only the channel indices differ.
Curves: `results/channel_ablation.png`; all numbers: `results/channel_ablation.json`.

Window-level AUC (how separable "count backwards" is from rest, before any threshold):

| Subject | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 ch | 0.966 | 0.949 | 0.996 | 0.979 | 0.943 | 0.974 | 0.988 | 0.990 | 0.944 | 0.991 | 0.972 |
| Muse-4 | 0.794 | 0.872 | 0.965 | 0.882 | 0.753 | 0.816 | 0.918 | 0.961 | 0.813 | 0.930 | 0.870 |

30 ch beats Muse-4 in 10 of 10 subjects, paired difference +0.10 (sd 0.06).

Best achievable detection at a fixed false-trigger budget (tau and K chosen per channel set, pooled over subjects):

| False triggers per idle hour allowed | 30 ch detect (tau, K, latency) | Muse-4 detect (tau, K, latency) | loss |
|---|---|---|---|
| 60 (one per minute) | 97.7% (0.60, 2, 4 s) | 83.0% (0.60, 3, 5 s) | 15 pts |
| 30 | 95.3% (0.60, 3, 5 s) | 66.0% (0.70, 3, 5 s) | 29 pts |
| 12 (one per 5 min) | 90.7% (0.75, 3, 5 s) | 39.0% (0.85, 3, 6 s) | 52 pts |
| 6 | 82.0% (0.90, 2, 5 s) | 22.7% (0.90, 3, 6 s) | 59 pts |
| 2 | 65.7% (0.95, 2, 5 s) | 9.7% (0.95, 3, 7 s) | 56 pts |

Fixed operating point tau 0.7, K 2, per subject (detect / false triggers per idle hour):

| Subject | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 ch | 90/60 | 90/12 | 100/12 | 100/24 | 100/72 | 100/48 | 97/24 | 100/24 | 90/72 | 93/48 |
| Muse-4 | 63/72 | 77/36 | 90/12 | 90/48 | 50/48 | 70/60 | 83/12 | 90/24 | 60/48 | 93/108 |

Paired at this point: detection +19 pts (sd 14) for 30 ch, better in 9 of 10 subjects; false triggers not
significantly different (-7 per hour, sd 24). Latency: 30 ch fires one window (1 s) earlier at the same tau
because its probabilities cross the threshold sooner; median 4 s vs 5 s at tau 0.7, K 2.

Reading: the two curves are not a shift, they diverge. At loose thresholds the 4-channel proxy keeps most of
the detection (loses 15 pts at 60 false triggers/hour). As you demand fewer false triggers the gap opens to
50 to 60 points, because 4 channels rarely produce the confident, sustained probabilities that a strict
threshold needs. With 30 channels a strict, quiet speed-dial (about 1 false trigger per 10 idle minutes) still
detects 8 to 9 commands in 10; with Muse-like channels the same quietness leaves 2 to 4 in 10. Subjects 3, 8
and 10 lose little; subjects 1, 5, 6 and 9 lose most. The mean loss on 10 subjects at 5-subject headline
settings (K 2, tau 0.7) is 96% vs 77% detection, consistent with the 5-subject table above.

## 5. Jev as the decision gate (live API, 600 calls, about $0.02; whole project to date $0.33)

Setup: 300 candidate triggers from the motor-imagery decoder (99 genuine, 201 false alarms), each with the
decoder's last 4 probability windows, margin, streak, the user's calibration accuracy, and a simulated context
(time, location, minutes since last trigger).

- **v1, one Choice {fire, confirm, ignore}:** Jev answered "confirm" on 286 of 300. It fired on 3 false alarms
  (a threshold matched to the same pass rate fires on 170), but also fired on only 1 of 38 clearly-legitimate
  commands. It is a safe but non-discriminating gate.
- **v2, two atomic Noul questions combined in code** (the pattern the docs recommend): AUC for separating
  genuine commands from false alarms was 0.58 for Jev's "genuine" probability vs 0.64 for the decoder's own
  top probability. Jev's "is this action sensible now" probability reached AUC 0.70 against the simulated
  context rule.

Conclusion: do not let Jev judge decoder evidence; a calibrated threshold plus streak rule in code is better and
free. Jev's defensible job is the context and policy layer: is this action redundant, already satisfied, too
soon after the last one, or costly enough to demand a confirm. That is a useful place for it, but it is a
policy convenience, not the source of reliability.

## 6. What to build (when hardware arrives)

1. Commands = 2 to 3 deliberate mental tasks (mental subtraction proven here; inner singing, mental rotation
   and vivid imagery are the usual other candidates and need a per-user bake-off during calibration).
2. Calibration = about 10 minutes: 30 x 10 s of each task and of rest.
3. Decoder = the exact pipeline used here: 4 to 30 Hz bandpass, 3 s windows every second, Riemannian
   tangent space + logistic regression, per user.
4. Trigger = 2 or 3 consecutive windows >= 0.7. Expect roughly 1 false trigger per 2 to 5 idle minutes on Muse,
   so add an artifact-based confirm (jaw clench, double blink, both near 100% on Muse) for money actions.
5. Jev = context and policy gate producing fire / confirm / ignore from action cost, redundancy, and timing.
6. Skip imagined words entirely; skip motor imagery on Muse.

## 7. Two-layer all-EEG interaction: command + EEG confirmation (`experiments/two_layer.py`)

**Confirmation dataset.** Shin 2017 dataset B (mental arithmetic vs rest, used for Layer 1) contains no yes/no
task. The same 29 subjects were recorded with the same 30-channel montage in Shin 2017 dataset A doing left-hand
vs right-hand motor imagery. Layer 2 uses that as a binary confirmation code: **left-hand imagery = YES,
right-hand imagery = NO**. This is a mapping I chose, not a native yes/no task; the labels and recordings are
untouched. An imagined-speech "yes/no" was ruled out because imagined words decode at chance (section 2).
Channel names were asserted identical between A and B for every subject.

**Pipeline (held out, 10 subjects, 5 folds each, folds aligned across the two datasets).**
Layer 1 = the section-4 decoder (3 s windows, 1 s hop, K-streak trigger). Layer 2 fires only after Layer 1
fires and decodes one D-second window with a Riemannian tangent-space + logistic-regression model trained on
D-second crops of the training-fold imagery trials. Decision: YES if P(yes | yes or no) >= theta, NO if
<= 1 - theta, else undecided (action aborted). Three variants:

- **A2** two-class yes/no decoder.
- **A3** three-class yes/no/idle decoder; the idle class is trained on the training fold's rest trials from
  dataset B, so Layer 2 can recognise "the user is not answering".
- **B** "repeat the command": Layer 2 re-runs the Layer-1 decoder on the next 3 s window (no dataset A needed).

Sequential simulation: a genuine command is a held-out task trial; if Layer 1 fires, the user answers YES
with a held-out left-imagery trial. An idle period is a held-out rest trial; if Layer 1 false-alarms, two
scenarios are scored: *unattended* (the user is not responding, Layer 2 sees the same rest recording continuing
after the fire) and *attended NO* (the user notices and answers NO with a held-out right-imagery trial).
Accidental action = action executed with no genuine command, per hour of idle.

### 7a. Layer 2 on its own (held-out imagery trials, first D seconds)

| D, theta | variant | yes/no accuracy (decided) | decided | P(YES given YES) | missed confirmations | P(YES given NO) false confirmations | called idle |
|---|---|---|---|---|---|---|---|
| 4 s, 0.6 | A2 | 0.700 | 0.77 | 0.503 | 0.497 | 0.220 | 0 |
| 4 s, 0.6 | A3 | 0.693 | 0.73 | 0.487 | 0.513 | 0.203 | 0.12 |
| 4 s, 0.5 | A2 | 0.657 | 1.00 | 0.637 | 0.363 | 0.323 | 0 |
| 8 s, 0.6 | A2 | 0.713 | 0.72 | 0.497 | 0.503 | 0.200 | 0 |
| 8 s, 0.5 | A2 | 0.687 | 1.00 | 0.673 | 0.327 | 0.300 | 0 |

Per-subject yes/no accuracy (4 s, 0.6, A2): 0.90, 0.80, 0.72, 0.47, 0.56, 0.58, 0.63, 0.72, 0.95, 0.54.
Left-vs-right imagery on this dataset is a weak confirmation signal for most people: two subjects are above
0.9, four are near chance. Confirmation is the bottleneck of the two-layer system, not command detection.

### 7b. Sequential pipeline (D = 4 s, theta = 0.6)

| Layer-1 point | L1 detect | L1 false alarms per idle hour | variant | genuine command completed | accidental actions per idle hour, unattended | accidental per idle hour, attended NO | end-to-end latency | reduction in accidental actions vs single layer |
|---|---|---|---|---|---|---|---|---|
| tau 0.7, K 2 | 96.0% | 27.6 | A2 | 49.0% | 6.0 | 7.2 | 8.6 s | 78% |
| | | | A3 | 47.0% | 1.2 | 8.4 | 8.6 s | 96% |
| | | | B | 83.3% | 4.8 | n/a | 7.4 s | 83% |
| tau 0.8, K 2 | 93.3% | 14.4 | A2 | 43.3% | 4.8 | 1.2 | 9.0 s | 67% |
| | | | A3 | 42.0% | 1.2 | 0.0 | 9.0 s | 92% |
| | | | B | 73.7% | 1.2 | n/a | 7.6 s | 92% |
| tau 0.9, K 2 | 82.0% | 4.8 | A2 | 39.0% | 1.2 | 1.2 | 9.5 s | 75% |
| | | | A3 | 37.3% | 1.2 | 1.2 | 9.4 s | 75% |
| | | | B | 53.7% | 0.0 | n/a | 7.8 s | 100% |

Event counts behind those rates (300 genuine and 300 idle trials, 0.83 idle hours): at tau 0.7, K 2 there
were 23 Layer-1 false alarms; A2 let 5 through as actions, A3 let 1 through, B let 4 through. At tau 0.9 there
were 4 false alarms and 0 to 1 got through. One event equals 1.2 per hour, so the accidental-action rates carry
about +-1 event of uncertainty; the census-based estimate below is more precise.

Sensitivity: forced choice (theta 0.5) raises completion to 62% (A2) / 57% (A3) at tau 0.7 but lets A2's
accidental actions rise to 12 per hour (57% reduction) while A3 stays at 1.2 (96%). An 8 s confirm window
(theta 0.5) gives 67% / 64% completion with A3 still at 1.2 per hour, at 12.5 s end-to-end latency.

### 7c. Independence of Layer-1 and Layer-2 errors

Idle-EEG census: every 4 s crop of every held-out rest trial (2100 crops) was run through Layer 2 and tagged by
whether Layer 1 had fired earlier in that trial.

| variant | P(YES) on idle crops with no L1 fire in trial | before an L1 fire | after an L1 fire | Spearman(L1 max prob of trial, L2 p_yes) |
|---|---|---|---|---|
| A2 | 0.380 (n 1939) | 0.408 (n 120) | 0.244 (n 41) | +0.03 (p 0.24) |
| A3 | 0.060 (n 1939) | 0.100 (n 120) | 0.073 (n 41) | +0.16 (p 3e-14) |

Observed accidental actions vs the independence prediction (L1 false alarms per hour x P(YES given idle)):

| L1 point | A2 observed / predicted | A3 observed / predicted |
|---|---|---|
| tau 0.7, K 2 | 6.0 / 10.5 | 1.2 / 1.7 |
| tau 0.8, K 2 | 4.8 / 5.5 | 1.2 / 0.9 |
| tau 0.9, K 2 | 1.2 / 1.8 | 1.2 / 0.3 |

- Trial level, A2: errors are independent or slightly anti-correlated; the idle EEG that fools the arithmetic
  detector is not the idle EEG that looks like left-hand imagery.
- Trial level, A3: a weak but real positive correlation (rho 0.16). Its idle class is trained on rest EEG, so
  rest segments that look "active" to Layer 1 also look less idle to Layer 2. The observed rates still sit near
  the independence prediction within the +-1 event resolution.
- Variant B is correlated by construction: the same decoder on the continuation of the same recording. It
  still cut accidental actions 83 to 100% because most false alarms did not persist for another 3 s window.
- Subject level: Spearman(L1 detection rate, L2 yes/no accuracy) = -0.79 (p 0.007, n 10). The people the
  arithmetic detector works best for are the people left/right imagery works worst for. With 10 subjects this
  is suggestive, not established, but it argues against assuming one good user is good at both layers.
- Genuine path: YES trials come from a separate recording, so trial-level correlation cannot be measured there
  with this data; the small subject-level correlation (+0.10 to +0.14) reflects subject quality.

### 7d. What the confirmation layer buys and costs

- **Accidental actions.** Single layer: 27.6 per idle hour at the loose point, 4.8 at the strict point.
  Adding an EEG confirmation with an idle class (A3) brings both to about 1 per hour (1 event in 0.83 idle
  hours), a 75 to 96% reduction. A two-class yes/no decoder without an idle class (A2) is much weaker
  protection (57 to 78%) because it answers YES on 38% of idle EEG.
- **Cost.** Genuine command completion falls from 96% (single layer) to about 47 to 64% with imagery
  confirmation, because left/right imagery is decoded correctly only about 70% of the time and half of YES
  attempts are missed at theta 0.6. Latency grows from 4 s to 8.6 to 12.5 s.
- **Repeat-the-command (B)** is the pragmatic middle: 83% completion, 83 to 92% fewer accidental actions,
  7.4 s latency, no second task to learn, no second dataset needed. Its errors are correlated with Layer 1,
  so its protection is weaker against sustained artefacts (e.g. talking) than an independent task would be.
- Recommendation for the product: Layer 1 at tau 0.8, K 2 plus repeat-the-command confirmation for cheap
  actions (74% completion, about 1 accidental action per idle hour); for money actions add an idle-aware
  independent confirmation (A3-style) or the artefact-based jaw clench from section 6, accepting the
  completion cost. Do not deploy a two-class yes/no confirmation without an idle class.
