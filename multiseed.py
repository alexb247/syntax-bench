"""Run the benchmark across several item-generation seeds and report mean +/- sd.

The single-run board tells you the scores for one particular bank of 152 items.
This asks a harder question: are the findings stable when the items themselves
are resampled? For each seed it regenerates a fresh bank from the same templates,
reruns every model, and then reports each metric as a mean and a standard
deviation across seeds. A finding that survives this (for example, Sonnet's
higher position bias) can be stated without the small-sample hedge; one that
swings wildly across seeds cannot.

    python multiseed.py --seeds 5          # baselines only, free
    export ANTHROPIC_API_KEY="sk-ant-..."  # add Claude (costs scale with seeds)
    python multiseed.py --seeds 5

Cost note: each seed is a full run (about 900 Claude calls). Five seeds is
roughly five times the cost of one board. Use --seeds 3 to spend less.
"""

import argparse
import json
import os
import statistics
import tempfile

import generate_items
from run import build_models
from syntaxbench.data import load_pairs
from syntaxbench.harness import evaluate


def bank_for_seed(seed: int):
    """Regenerate the item bank under a given seed, then load it through the
    normal loader so validation is identical to a real run."""
    generate_items.rng.seed(seed)          # reseed the generator's RNG
    items = generate_items.build()
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
        path = f.name
    pairs = load_pairs(path)
    os.unlink(path)
    return pairs


def metrics_for_run(model, pairs) -> dict[str, float]:
    res = evaluate(model, pairs)
    row = {"overall": res.accuracy(), "pos_bias": res.position_bias()}
    for ph in res.phenomena:
        row[ph] = res.accuracy(ph)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed stability run.")
    ap.add_argument("--seeds", type=int, default=5, help="how many seeds to run")
    ap.add_argument("--start", type=int, default=42, help="first seed")
    ap.add_argument("--out", default="multiseed_results.json")
    args = ap.parse_args()

    seeds = list(range(args.start, args.start + args.seeds))
    models = build_models()                # built once, reused across seeds
    print(f"Running {len(seeds)} seeds {seeds[0]}..{seeds[-1]} "
          f"({len(seeds)} full evaluations per model).\n")

    # accumulate: model -> metric -> [value per seed]
    acc: dict[str, dict[str, list[float]]] = {}
    phenomena: list[str] = []
    for s in seeds:
        pairs = bank_for_seed(s)
        if not phenomena:
            phenomena = sorted({p.phenomenon for p in pairs})
        for model in models:
            row = metrics_for_run(model, pairs)
            md = acc.setdefault(model.name, {})
            for metric, val in row.items():
                md.setdefault(metric, []).append(val)
        print(f"  seed {s} done")

    metrics = ["overall"] + phenomena + ["pos_bias"]
    print(f"\nMean +/- sd across {len(seeds)} seeds (percent):\n")
    header = f"{'model':<32}" + "".join(f"{m:>14}" for m in metrics)
    print(header)
    print("-" * len(header))

    summary: dict[str, dict] = {}
    for model_name, md in acc.items():
        cells = []
        summary[model_name] = {}
        for metric in metrics:
            vals = [v * 100 for v in md.get(metric, [])]
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary[model_name][metric] = {"mean": round(mean, 2),
                                           "sd": round(sd, 2),
                                           "values": [round(v, 1) for v in vals]}
            cells.append(f"{mean:5.1f}+-{sd:4.1f}")
        print(f"{model_name:<32}" + "".join(f"{c:>14}" for c in cells))

    with open(args.out, "w") as f:
        json.dump({"seeds": seeds, "summary": summary}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
