"""Run the targeted-syntax benchmark end to end.

    python run.py

By default it runs the no-API-key baselines so the pipeline works out of the
box. Register real models in build_models() to populate the leaderboard.
"""

import argparse

from syntaxbench.data import load_pairs
from syntaxbench.harness import evaluate
from syntaxbench.models import (
    AlwaysFirst,
    Model,
    RandomBaseline,
    ShorterSentence,
    # OpenAIModel,      # uncomment once you have a key wired in
    # AnthropicModel,
    # HuggingFaceModel,
)
from syntaxbench.report import leaderboard_table, write_json


def build_models() -> list[Model]:
    """The models to evaluate. Add real ones here."""
    return [
        RandomBaseline(),
        AlwaysFirst(),
        ShorterSentence(),
        # OpenAIModel("gpt-4o-mini"),
        # AnthropicModel("claude-haiku-4-5-20251001"),
        # HuggingFaceModel("meta-llama/Llama-3.2-1B-Instruct"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted-syntax acceptability benchmark.")
    parser.add_argument("--data", default="data/minimal_pairs.jsonl")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    pairs = load_pairs(args.data)
    phenomena = sorted({p.phenomenon for p in pairs})
    print(f"Loaded {len(pairs)} minimal pairs across {len(phenomena)} phenomena: {', '.join(phenomena)}\n")

    results = [evaluate(model, pairs) for model in build_models()]

    print(leaderboard_table(results, phenomena))
    write_json(results, phenomena, args.out)
    print(f"\nWrote {args.out} (the leaderboard data source for the web page).")


if __name__ == "__main__":
    main()
