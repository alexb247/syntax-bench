# Targeted-Syntax Acceptability Benchmark for LLMs

**The question.** Do today's language models actually judge English grammaticality,
or do they lean on surface cues? This benchmark probes that on three syntactic
phenomena using minimal pairs, where each item differs by exactly the feature
under test.

> Status: scaffold. Baselines run end to end. Real models and the deployed
> leaderboard are in progress.

## Why minimal pairs

Each item is one grammatical sentence and a minimally different ungrammatical one:

| phenomenon | grammatical | ungrammatical | what it isolates |
|---|---|---|---|
| agreement attraction | The key to the cabinets **is** on the table. | The key to the cabinets **are** on the table. | subject-verb agreement across a plural distractor |
| NPI licensing | **Nobody** has any money left. | **Somebody** has any money left. | whether a negative-polarity item is licensed |
| filler-gap | Which book did the student read __ yesterday? | Which book did the student read **the magazine** yesterday? | a wh-filler needs an empty gap |

Because the two sentences differ in only one place, getting the item right
requires sensitivity to that one feature and nothing else.

## Method

Each model is shown both sentences and picks the acceptable one. Every pair is
run in **both orders** (grammatical sentence first, then second) and the item
score is the average. This cancels position bias in the headline number and lets
us measure it directly as a separate column. Scoring is forced choice; a
probability- or surprisal-based variant is a planned v2.

## Running it

```bash
python run.py
```

Runs with no API keys and nothing to install: the included baselines exist to
prove the pipeline and act as honest reference points. To add real models, fill
in the stubs in `syntaxbench/models.py` and register them in `build_models()`
inside `run.py`.

## Results

_Populated once real models are wired in. Drop the leaderboard here, plus one
chart of accuracy by phenomenon._

A finding already visible from the baselines: the shorter-sentence heuristic
beats chance on agreement and filler-gap, which means those seed items carry a
length confound. See limitations.

## Limitations and next steps

- **Length confound in seed items.** The grammatical sentence is often the
  shorter string, so a length heuristic scores above chance. Rebalance items so
  the two sentences are length-matched. (First task.)
- **Scale.** Currently 15 hand-written pairs. Target is 50 to 100 per
  phenomenon, drawing additional items from the BLiMP minimal-pairs dataset and
  hand-writing the hardest cases.
- **Single scoring method.** Add probability or surprisal scoring as a second
  view on the same items.

## Layout

```
data/minimal_pairs.jsonl   the items
syntaxbench/data.py        loading and validation
syntaxbench/models.py      model interface, baselines, real-model stubs
syntaxbench/harness.py     two-order evaluation
syntaxbench/report.py      leaderboard table and JSON export
run.py                     entry point
```
