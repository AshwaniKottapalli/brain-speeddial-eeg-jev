# Can Jev learn EEG -> intent? Experimental report (2026-09-19)

## 1. What Jev actually exposes (verified from docs.typesafe.ai on 2026-09-19)

| Property | Fact |
|---|---|
| Access | Hosted only. `POST https://api.typesafe.ai/v1/systemone`, Python SDK `typesafe-sdk` (`TypeSafeClient`, `Choice`, `Score`, `Noul`) |
| Models | `jev-1.13.0` (= `jev-latest`, `jev-preview`) |
| Weights | Not available |
| Fine-tuning / LoRA / adapters | Explicitly not offered: "Jev is not fine-tuned or LoRA-adapted with customer data" |
| Gradients / logits | None. Output is calibrated class probabilities + confidence per question |
| Input | **Text only**: string, JSON object, or array of text. "No image, audio, or video input." 64k tokens/request, 32k for state |
| Output | Choice (option probabilities), Score (ordinal), Noul (P(yes)) |
| Price | $0.042 per million input tokens; output free |

Consequence: "fine-tune Jev on EEG" and "feed neural embeddings into Jev" are impossible by design.
The only way EEG can enter Jev is as text/JSON. Jev can be the fixed *decision core* of a pipeline, but any
learning must happen in a component in front of it.

## 2. Data and preprocessing

| Dataset | Subjects used | Channels | Classes | Trials |
|---|---|---|---|---|
| BCI Competition IV 2a (BNCI2014_001, via MOABB) | 1-3 | 22 | left hand, right hand, feet, tongue | 1728 |
| PhysioNet EEGMMIDB (via MNE) | 1-10 | 64 | left hand, right hand, both hands, feet | 900 |

Pipeline: 1-40 Hz FIR bandpass, average reference (EEGMMIDB), epochs -1 to +4 s around cue, resample 128 Hz.
Features handed to Jev as JSON: event-related mu/beta power change in dB vs. pre-cue baseline at C3/Cz/C4 and
6 neighbours, laterality (C3-C4), midline-vs-lateral contrast, mu time course, and absolute log band power for
every channel. Code: `eeg_jev/data.py`, `eeg_jev/features.py`.

## 3. Reference ceilings (within-subject 5-fold CV, accuracy, chance = 0.25)

| | BCI IV 2a | EEGMMIDB |
|---|---|---|
| Riemannian tangent space + LR on raw EEG | **0.796** | **0.603** |
| CSP + LDA on raw EEG | 0.740 | 0.511 |
| Logistic regression on the *exact JSON numbers Jev sees* | 0.519 | 0.487 |

The last row is the information ceiling of the "text door": serialising EEG to interpretable numbers already
throws away a third of the decodable signal before Jev sees anything.

## 4. Experiments with live Jev (total spend: 1653 calls, 7.4M tokens, $0.31)

### 4a. Zero-shot: feature JSON -> Jev Choice (no examples)

| Dataset | Prompt | n | Accuracy | Behaviour |
|---|---|---|---|---|
| BCI IV 2a | physiology-primed criteria | 400 | 0.292 | predicts only left/right hand (feet 4x, tongue 0x) |
| BCI IV 2a | blind (class names only) | 400 | 0.263 | predicts only left/right hand |
| EEGMMIDB | physiology-primed criteria | 200 | 0.255 | predicts only left/right hand |

Verdict: chance or marginally above. Jev cannot do the numeric aggregation (compare dozens of dB values,
decide laterality) needed to turn raw feature numbers into an intent. It collapses onto the two classes its
priors know best.

### 4b. Few-shot in-context: labeled same-subject exemplars in state -> Jev Choice

| Exemplars / class | Descriptor | tokens/call | Jev accuracy | Matched LR on same numbers & same k |
|---|---|---|---|---|
| 5 | compact (8 numbers) | 2.8k | 0.325 | 0.289 |
| 15 | compact (8 numbers) | 7.4k | 0.300 | 0.314 |
| 5 | full (137 numbers) | 31.9k | 0.267 | 0.328 |

Verdict: Jev in-context learning on numeric EEG features is at the level of a logistic regression trained on
the same handful of numbers, i.e. barely above chance. More numbers per exemplar made it worse, not better
(context filled with digits). In-context learning is not a viable route to EEG decoding with Jev.

### 4c. Trained neural adapter -> discrete physiology descriptor -> Jev (the main result)

Architecture (`experiments/adapter.py`):

```
raw EEG epoch (22 ch x 3 s, 8-30 Hz)
  -> Riemannian covariance -> tangent-space vector (253-d)             [fixed, unsupervised]
  -> MLP 253 -> 64 -> 3 heads x 3 levels                              [TRAINED]
  -> descriptor text, e.g. {"C3": "suppression", "Cz": "no change", "C4": "increase"}
  -> Jev Choice with physiology primer + per-class signature criteria  [FIXED, live API]
  -> intent probabilities
```

Training trick: the descriptor vocabulary is finite (3 levels ^ 3 sites = 27 strings), so Jev is queried once
per string (27 live calls, ~$0.002) to build a lookup table T[descriptor] = Jev's class probabilities. The
encoder emits a distribution over descriptors; expected Jev output = sum q(d) T[d], which is differentiable in
the encoder. Cross-entropy is minimised against Jev's *real* answers. At test time the hard argmax descriptor
is used and the prediction is literally Jev's answer for that string. A live re-query of 9 held-out
descriptors matched the cached table within 0.07 probability (max), so the pipeline is genuinely EEG -> text ->
Jev -> intent.

**BCI IV 2a results (1728 trials, within-subject 5-fold):**

| Variant | Jev table | Accuracy | Note |
|---|---|---|---|
| semantic, 3 levels, no class criteria | live | 0.593 | Jev never answers "tongue" for any of the 27 strings -> 3-class ceiling |
| semantic, 5 levels, no class criteria | live | 0.591 | same tongue hole with 125 strings |
| **semantic + class-signature criteria, 3 levels** | **live** | **0.737** | all 4 classes reachable; per-subject 0.80 / 0.55 / 0.86 |
| semantic + criteria, 4 sites (adds lateral C5/C6), 3 levels | live | 0.742 | 81 strings |
| opaque codes (`feature_0: level_2`), no physiology words | live | 0.389 | Jev collapses to 2 classes without semantics |
| semantic + criteria, **random-permuted table** (control) | shuffled | 0.714 | |
| same encoder trunk, direct softmax head, no Jev (control) | - | 0.734 | |
| Riemannian + LR (reference) | - | 0.796 | |

Multi-seed repeat (4 seeds): Jev table 0.730 +- 0.008 vs random table 0.712 +- 0.008. Jev beat random in
every seed. Low-data regime: at 5 training trials/class Jev 0.541 vs random 0.499; at 15/class 0.616 vs 0.601.

Does the adapter speak physiology? Most frequent descriptor per true class (semantic+criteria run):

| True intent | Descriptor emitted (C3 / Cz / C4) | share |
|---|---|---|
| right hand | suppression / no change / no change | 69% |
| left hand | increase / no change / suppression | 34% |
| feet | increase / suppression / increase | 62% |
| tongue | increase / increase / increase | 66% |

These are textbook contralateral ERD patterns, learned purely from Jev's feedback. Jev's own zero-shot
descriptor->intent table agrees with a hand-written physiology rule oracle on 20/24 unambiguous descriptors.

**EEGMMIDB results (900 trials, 10 subjects, 4 classes):** adapter+Jev 0.490, random-table control 0.490,
direct softmax 0.553, Riemannian 0.603. Harder dataset (about 72 training trials per fold) and the Jev-table
advantage vanished.

## 5. Answers to the questions asked

**Can Jev be adapted/fine-tuned to understand EEG directly?** No. There are no weights, no fine-tuning, no
adapter API, no gradient or embedding input. Text/JSON is the only input.

**Can raw EEG (or EEG numbers) enter Jev and be decoded?** Effectively no. Zero-shot on feature JSON is at
chance (0.26-0.29 vs 0.25). Few-shot in-context is at chance (0.27-0.33) and no better than a logistic
regression given the same numbers.

**What is the minimum neural adapter that keeps Jev as the core, and does it work?** A fixed Riemannian
tangent-space embedding plus a ~17k-parameter MLP that emits a 3-word descriptor. It works: 0.737 on BCI IV 2a,
vs 0.796 for the best conventional classifier and 0.734 for the identical MLP with a plain softmax head. The
whole EEG->text->Jev->intent loop runs live and the descriptors are physiologically meaningful. Training cost
against Jev was 27 API calls (about $0.002) because the descriptor space is finite.

**What does Jev itself contribute?** Modestly but consistently positive: +1.8 points over a randomly permuted
lookup table across 4 seeds on BCI IV 2a, +4 points in the 5-trials-per-class regime, and zero on EEGMMIDB.
Most of the decoding power lives in the trained encoder. Jev's value is (a) a fixed, calibrated, physiologically
sensible decision function that all 4 classes can reach when given class-signature criteria, and (b) an
interpretable text bottleneck. It is not a source of EEG knowledge that a classifier lacks.

## 6. What it would take to make "EEG -> Jev -> intent" genuinely work

1. **Accept the architecture found here.** Encoder -> discrete semantic descriptor -> Jev is the only form
   Jev's API permits. The descriptor vocabulary must be enumerable so Jev can be tabulated and the encoder
   trained against the table; with a few hundred strings the cost is cents.
2. **Widen the text door.** Add more sites/bands/time windows to the vocabulary (4 sites already helped
   slightly: 0.742). The information ceiling is set by what the descriptor can express, not by Jev.
3. **Give Jev class signatures.** Without per-class criteria Jev never emitted "tongue" for any descriptor.
   With them all classes were reachable and accuracy jumped from 0.59 to 0.74. This is the single most
   important prompt-engineering fact.
4. **Use physiology words, not codes.** Opaque codes drop Jev to 0.39; Jev contributes nothing without
   semantics it recognises.
5. **Where Jev adds real value:** turning the decoded motor intent into downstream *structured decisions*
   (e.g. "given these device states and this descriptor, which action, with what confidence?") using
   confidence-gated routing, which is what the model is built for. Raw signal decoding should stay in the
   encoder.
6. **To go beyond 0.74** you need a better encoder (EEGNet / larger tangent-space MLP), subject-specific
   training data (the poorest subject was 0.55 for every method), and a richer descriptor. None of these
   involve Jev.

## 7. Reproduce

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
echo TYPESAFE_API_KEY=... > .env
.venv/bin/python experiments/prep.py bnci && .venv/bin/python experiments/baseline.py bnci
.venv/bin/python experiments/zero_shot.py bnci physio 400
.venv/bin/python experiments/few_shot.py bnci 5 40 compact
.venv/bin/python experiments/adapter.py bnci semantic_crit 3 ts jev      # main result
.venv/bin/python experiments/adapter.py bnci semantic_crit 3 ts random   # control
```
All Jev responses are cached in `cache/jev/`; raw per-trial outputs are in `results/*.json`.
