"""v2 factorial island study: score matched 2x2 quadruples by local surprisal.

    python3 generate_factorial.py    # (re)build data/factorial_islands.jsonl
    python3 factorial_islands.py     # score with gpt2 locally; free, no API key

For each island subtype the items are quadruples crossing [island vs
non-island structure] x [extraction vs no-extraction] with lexical content
held constant. The island effect for an item is the difference in differences
on total surprisal:

    DiD = (S[island_extraction] - S[island_noextraction])
        - (S[nonisland_extraction] - S[nonisland_noextraction])

Positive means extraction costs more inside the island structure than the
structure and extraction costs alone predict -- the factorial subtraction the
forced-choice v1 format could not make (README, Limitations). Per subtype we
report the mean DiD with a percentile bootstrap 95% CI over items.

Needs the optional extra (the core benchmark never does):

    pip3 install transformers torch
"""

import argparse
import json

from syntaxbench.factorial import CONDITIONS, bootstrap_ci, did, load_quadruples

_SHORT = {
    "nonisland_noextraction": "-isl/-ext",
    "nonisland_extraction": "-isl/+ext",
    "island_noextraction": "+isl/-ext",
    "island_extraction": "+isl/+ext",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Factorial island study scored by local-model surprisal.")
    ap.add_argument("--data", default="data/factorial_islands.jsonl")
    ap.add_argument("--model", default="gpt2", help="Hugging Face causal LM id")
    ap.add_argument("--out", default="factorial_results.json")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    args = ap.parse_args()

    quads = load_quadruples(args.data)
    subtypes = sorted({q.subtype for q in quads})
    print(f"Loaded {len(quads)} quadruples across {len(subtypes)} subtypes: {', '.join(subtypes)}")

    try:
        from syntaxbench.surprisal import SurprisalScorer
    except ImportError as e:
        raise SystemExit(
            f"Missing optional surprisal dependency: {e}.\n"
            "Install with:  pip3 install transformers torch\n"
            "(The core benchmark, run.py and multiseed.py, never needs these.)"
        )
    print(f"Scoring with {args.model} on CPU ({len(quads) * len(CONDITIONS)} sentences)...\n")
    scorer = SurprisalScorer(args.model)

    scored = []
    for i, q in enumerate(quads, start=1):
        surprisal = {c: scorer.score(q.sentences[c]) for c in CONDITIONS}
        totals = {c: s["total"] for c, s in surprisal.items()}
        scored.append({"id": q.id, "subtype": q.subtype,
                       "surprisal": surprisal, "did": did(totals)})
        if i % 12 == 0:
            print(f"  {i}/{len(quads)} quadruples scored")

    results = {}
    for st in subtypes:
        items = [s for s in scored if s["subtype"] == st]
        dids = [it["did"] for it in items]
        lo, hi = bootstrap_ci(dids, n_boot=args.boot)
        results[st] = {
            "n_items": len(items),
            "condition_means_total": {
                c: sum(it["surprisal"][c]["total"] for it in items) / len(items)
                for c in CONDITIONS
            },
            "interaction": {"mean": sum(dids) / len(dids), "ci95": [lo, hi]},
            "items": [{"id": it["id"], "did": it["did"],
                       "surprisal": it["surprisal"]} for it in items],
        }

    print(f"\nTotal surprisal (nats) with {args.model}, mean over items per condition.")
    print("DiD = (+isl/+ext - +isl/-ext) - (-isl/+ext - -isl/-ext); positive = island effect,")
    print("a 95% CI excluding zero = island effect the item noise does not explain away.\n")
    header = (f"{'subtype':<12}{'n':>4}"
              + "".join(f"{_SHORT[c]:>12}" for c in CONDITIONS)
              + f"{'DiD':>9}{'95% CI':>20}")
    print(header)
    print("-" * len(header))
    for st in subtypes:
        r = results[st]
        cm = r["condition_means_total"]
        ci = r["interaction"]["ci95"]
        print(f"{st:<12}{r['n_items']:>4}"
              + "".join(f"{cm[c]:>12.1f}" for c in CONDITIONS)
              + f"{r['interaction']['mean']:>+9.2f}"
              + f"  [{ci[0]:+6.2f}, {ci[1]:+6.2f}]")

    payload = {
        "model": args.model,
        "measure": "total surprisal (nats)",
        "design": "2x2 [island structure] x [extraction], interaction as difference in differences",
        "n_boot": args.boot,
        "subtypes": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
