"""Loading and validating minimal-pair items."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MinimalPair:
    """One acceptability item: a grammatical sentence and a minimally different ungrammatical one."""
    id: str
    phenomenon: str
    good: str  # the grammatical sentence
    bad: str   # the minimally different ungrammatical sentence
    note: str = ""


def load_pairs(path: str) -> list[MinimalPair]:
    """Read a .jsonl file of minimal pairs into MinimalPair objects."""
    pairs: list[MinimalPair] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        pair = MinimalPair(
            id=record["id"],
            phenomenon=record["phenomenon"],
            good=record["good"],
            bad=record["bad"],
            note=record.get("note", ""),
        )
        if pair.id in seen_ids:
            raise ValueError(f"Duplicate item id {pair.id!r} on line {line_no}")
        seen_ids.add(pair.id)
        pairs.append(pair)
    return pairs
