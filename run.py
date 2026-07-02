"""Run the targeted-syntax benchmark end to end.

    python run.py

Baselines always run, with no API key and nothing to install. If the
ANTHROPIC_API_KEY environment variable is set, three sizes of Claude are added
to the board automatically, so you can see whether a larger model judges
grammar better.
"""

import argparse
import os

from syntaxbench.data import load_pairs
from syntaxbench.harness import evaluate
from syntaxbench.models import (
    AlwaysFirst,
    AnthropicModel,
    Model,
    RandomBaseline,
    ShorterSentence,
    # OpenAIModel,        # set OPENAI_API_KEY and register below to add GPT
    # HuggingFaceModel,   # add an open model for an open-vs-closed comparison
)
from syntaxbench.report import leaderboard_table, write_json


def build_models() -> list[Model]:
    """The models to evaluate. Baselines always run; real models are added when
    their API key is present, so the script never crashes for missing keys."""
    models: list[Model] = [RandomBaseline(), AlwaysFirst(), ShorterSentence()]

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            models += [
                AnthropicModel("claude-haiku-4-5-20251001"),
                AnthropicModel("claude-sonnet-4-6"),
                AnthropicModel("claude-opus-4-8"),
            ]
            print("Added Claude: haiku, sonnet, opus.\n")
        except Exception as e:
            print(f"ANTHROPIC_API_KEY is set but Claude could not start: {e}")
            print("If this mentions a missing module, run: pip3 install anthropic\n")
    else:
        print("No ANTHROPIC_API_KEY found, so the board shows baselines only.")
        print('Add Claude with:  export ANTHROPIC_API_KEY="sk-ant-..."\n')

    return models


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
