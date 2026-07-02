# Targeted-Syntax Acceptability Benchmark for LLMs

A small, controlled benchmark that asks not just whether frontier language models
can judge English grammaticality, but **how stable those judgments are**. It tests
three subtle syntactic phenomena with minimal pairs, scores several models against
honest baselines, and reports where they break.

**Live leaderboard:** https://alexb247.github.io/syntax-bench/

## Findings

On these phenomena, grammatical competence is largely a solved problem: all three
Claude models judge the constructions at or near ceiling. The interesting signal
is not accuracy but **robustness**, and it only shows up because every pair is
tested in both presentation orders.

- **The real weakness is order sensitivity, not grammar.** Almost every model
  error is an *order flip*: the model judges a pair correctly one way around and
  reverses when the same two sentences are swapped. It is not that the models
  cannot tell the sentences apart; it is that the judgment is not stable.
- **Position robustness is not monotonic with model size.** The mid-sized model
  (Sonnet) is the *least* order-robust of the three, with a position bias of 12.4
  versus about 2 for both the smaller Haiku and the larger Opus. This held across
  five independently generated item banks (Sonnet 12.4 ± 0.7), so it is a stable
  property, not an artifact of one sample. Bigger is not reliably steadier.
- **The NPI weakness is specific and explainable.** Where Sonnet dips, the misses
  concentrate on idiomatic *minimizer* NPIs ("lift a finger," "in the slightest"),
  not on negative polarity in general. Plain "ever" and "at all" are handled
  cleanly. So the gap is about low-frequency idioms, not the underlying licensing.
- **Measured without the confound, islands are real but not uniform (v2).** A
  factorial surprisal study with a local open model shows clear super-additive
  island effects for subject, adjunct, and complex-NP islands, and an effect
  indistinguishable from zero for the *wh*-island, the weak, gradient case in
  the human literature. See the v2 section below.

The headline order-sensitivity result is stable across five seeds; the per-phenomenon
subscores carry more variance and are read as suggestive (see Limitations).

## Method

Each item is a **minimal pair**: one grammatical sentence and a minimally different
ungrammatical one, designed so that getting it right requires sensitivity to a
single structural feature and nothing else. A model is shown both sentences and
picks the acceptable one (forced choice).

Every pair is scored in **both orders** (grammatical sentence first, then second),
and the item score is the average. This cancels position preference out of the
headline accuracy and, just as importantly, lets us *measure* it directly as a
separate column. That two-order control is what surfaced the main finding.

**Baselines.** Alongside the real models we run three reference models that carry
no grammatical knowledge: a random baseline, an always-pick-first baseline (which
exposes raw position bias), and a shorter-sentence heuristic (which exposes length
confounds). A real model that cannot clearly beat these has learned nothing.

## The phenomena

All items are controlled: length and surface cues are held constant where possible,
so only the structural feature varies. The starred member is ungrammatical.

- **That-trace.** Object extraction across overt *that* is fine; subject extraction
  is barred. Both members keep *that* and are the same length.
  `Who do you believe that the board hired?` vs `*Who do you believe that hired the board?`
- **NPI in the restrictor.** A negative-polarity item is licensed in the
  downward-entailing restrictor of *every* but not in its upward-entailing scope.
  Both members use the same words; only the NPI moves.
  `Every author who had ever missed a deadline was dropped.` vs
  `*Every author who was dropped had ever missed a deadline.`
- **Islands.** Extraction is barred out of islands (adjunct, complex noun phrase,
  *wh*, and subject). Tested across four subtypes. See Limitations: island items
  carry an unavoidable length confound in a forced-choice format.

## Item construction

Items are generated from templates with a fixed random seed (`generate_items.py`),
so the bank is fully reproducible, in the spirit of BLiMP. The current bank is 152
items: 50 that-trace, 50 NPI, and 52 islands across four subtypes.

Generation scales, but it produces some marginal items, so the bank goes through a
**validation pass**: model disagreement flags an item for review, and it is cut or
fixed on linguistic grounds, not on whether models passed it. Two systematic
template defects were found and fixed this way: adjunct items whose matrix verb
could take an object (giving the gap a grammatical attachment), and that-trace
items with weak bridge verbs that muddied the grammatical member.

## Limitations

- **Islands are length-confounded in forced choice.** The island-creating structure
  is inherently longer than its control, so the shorter-sentence heuristic reaches
  up to 100% on the complex-NP subtype on length alone. Forced-choice island numbers
  must therefore be read against that baseline, not at face value. The clean,
  confound-free phenomena are that-trace and NPI, where the heuristic sits at chance.
  Measuring islands properly requires a factorial, probability-based design that
  subtracts a complexity baseline; **v2 below does exactly that**, and the
  forced-choice island column stays only for comparison.
- **Item counts.** Tens of items per phenomenon means individual subscores carry
  real variance (NPI ranges about ± 3 across seeds). The headline position-robustness
  result, however, is stable across five independently generated seeds, so the
  model-to-model gap there can be stated with confidence (`multiseed.py`).
- **Single prompt, single decoding.** Results reflect one prompt template and one
  forced-choice elicitation. A robustness check across prompts is future work.

## Results

Accuracy is the share of pairs judged correctly, averaged over both presentation
orders. Position bias is the gap between the two orders (lower is steadier).

_152-item validated bank, three Claude models, run June 2026._

| model | overall | that-trace | NPI | island* | position bias |
| --- | --- | --- | --- | --- | --- |
| claude-opus-4-8 | 99.0 | 100.0 | 98.0 | 99.0 | 2.0 |
| claude-haiku-4-5 | 96.1 | 98.0 | 98.0 | 92.3 | 2.6 |
| claude-sonnet-4-6 | 93.8 | 100.0 | 88.0 | 93.3 | 12.5 |
| shorter-sentence (baseline) | 54.3 | 48.0 | 40.0 | 74.0 | 3.3 |
| always-first (baseline) | 50.0 | 50.0 | 50.0 | 50.0 | 100.0 |
| random (baseline) | 44.4 | 48.0 | 40.0 | 45.2 | 4.6 |

\* Island accuracy is length-confounded; see Limitations and the v2 study below.

## v2: The factorial island study

Forced choice cannot measure islands cleanly, because the island structure is
inherently longer and rarer than its control (see Limitations). v2 measures
islands the way the experimental syntax literature does. Each item is a matched
**quadruple** crossing [island vs non-island structure] x [extraction vs
no-extraction], with lexical content held constant, and the island effect is the
interaction, a difference in differences on total surprisal:

```
DiD = (island+extraction - island control) - (non-island+extraction - non-island control)
```

The structure's own cost appears in both of its conditions, and the extraction's
own cost appears on both sides, so both subtract out. A positive DiD means
extraction costs more inside the island than the parts predict on their own
(super-additivity), which is the island effect itself, with no length confound.

Surprisal needs token-level probabilities, which frontier APIs do not expose, so
v2 scores with a local open model (gpt2) and runs free with no key. Each study
uses the instrument its models allow: forced choice for API models, surprisal
for open ones.

Results: 12 quadruples per subtype, total surprisal in nats, percentile
bootstrap 95% CI over items.

| island subtype | interaction (DiD) | 95% CI |
| --- | --- | --- |
| subject | +8.66 | [+6.38, +11.05] |
| adjunct | +2.96 | [+1.83, +4.07] |
| complex NP | +2.91 | [+1.61, +4.25] |
| wh | +0.59 | [-0.09, +1.22] |

Three of the four subtypes show a clearly positive interaction: gpt2 penalizes
extraction out of subject, adjunct, and complex-NP islands beyond the additive
costs of the structure and the extraction. The *wh*-island interval straddles
zero, and that is a coherent result rather than a failure: *wh*-islands are the
weak, gradient case in the human acceptability literature, and that is how they
come out once the length confound is subtracted away. Extending v2 beyond gpt2
to larger open models is the natural next step.

## Running it

```bash
python generate_items.py     # (re)build the item bank, reproducible
python run.py                # run baselines; add Claude with a key (below)
python test_factorial.py     # v2 tests, stdlib only
python generate_factorial.py # (re)build the factorial quadruples
python factorial_islands.py  # v2 island study, local gpt2, free, no key
```

The baselines run with no key and nothing to install. To add the Claude models,
install the SDK and set your key, then rerun:

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
python run.py
```

## Layout

```
generate_items.py          template item generator (fixed seed)
data/minimal_pairs.jsonl   the generated item bank
generate_factorial.py      v2: matched 2x2 island quadruples (fixed seed)
data/factorial_islands.jsonl  the generated factorial bank
syntaxbench/data.py        loading and validation
syntaxbench/models.py      model interface, baselines, Claude/OpenAI adapters
syntaxbench/harness.py     two-order evaluation
syntaxbench/report.py      leaderboard table and JSON export
syntaxbench/factorial.py   v2: quadruple loading, DiD and bootstrap math
syntaxbench/surprisal.py   v2: local-model surprisal (the only torch module)
factorial_islands.py       v2 entry point, writes factorial_results.json
test_factorial.py          v2 tests (no torch needed)
run.py                     entry point, writes results.json
multiseed.py               multi-seed stability run (mean +/- sd across seeds)
index.html                 the deployed leaderboard
```
