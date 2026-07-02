"""The v2 factorial island study: quadruple loading and the interaction math.

Standard library only. This module must stay importable without torch or
transformers: the tests and the generator use it directly, and only the
factorial_islands.py entry point pairs it with syntaxbench.surprisal.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

# The 2x2: [island vs non-island structure] x [extraction vs no-extraction].
CONDITIONS = (
    "nonisland_noextraction",
    "nonisland_extraction",
    "island_noextraction",
    "island_extraction",
)


@dataclass(frozen=True)
class Quadruple:
    """One factorial item: four sentences with shared lexical content."""
    id: str
    subtype: str
    sentences: dict                              # condition name -> sentence
    lexicon: dict = field(default_factory=dict)  # the shared lexical fillers
    note: str = ""


def load_quadruples(path: str) -> list[Quadruple]:
    """Read a .jsonl file of factorial quadruples, validating as we go."""
    quads: list[Quadruple] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        sentences = record["sentences"]
        missing = [c for c in CONDITIONS if not sentences.get(c)]
        if missing:
            raise ValueError(f"Item {record.get('id')!r} (line {line_no}) is missing conditions: {missing}")
        if len({sentences[c] for c in CONDITIONS}) != len(CONDITIONS):
            raise ValueError(f"Item {record.get('id')!r} (line {line_no}) repeats a sentence across conditions")
        quad = Quadruple(
            id=record["id"],
            subtype=record["subtype"],
            sentences={c: sentences[c] for c in CONDITIONS},
            lexicon=record.get("lexicon", {}),
            note=record.get("note", ""),
        )
        if quad.id in seen_ids:
            raise ValueError(f"Duplicate item id {quad.id!r} on line {line_no}")
        seen_ids.add(quad.id)
        quads.append(quad)
    return quads


def did(totals: dict) -> float:
    """Difference in differences for one item's four condition scores.

        (island_extraction - island_noextraction)
      - (nonisland_extraction - nonisland_noextraction)

    Positive: extraction costs more inside the island structure than outside,
    beyond what the structure and the extraction cost on their own.
    """
    return ((totals["island_extraction"] - totals["island_noextraction"])
            - (totals["nonisland_extraction"] - totals["nonisland_noextraction"]))


def bootstrap_ci(values: list[float], n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`, resampling items with
    replacement. Deterministic for a given seed."""
    if not values:
        raise ValueError("Cannot bootstrap an empty list.")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = means[int(n_boot * (alpha / 2))]
    hi = means[int(n_boot * (1 - alpha / 2)) - 1]
    return lo, hi
