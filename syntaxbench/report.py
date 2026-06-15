"""Turn ModelResults into a leaderboard table and a JSON file for the web page."""

import json
from .harness import ModelResult


def _fmt(value: float) -> str:
    return "  n/a" if value != value else f"{value * 100:5.1f}"  # value != value catches nan


def leaderboard_table(results: list[ModelResult], phenomena: list[str]) -> str:
    """A plain-text leaderboard, sorted by overall accuracy."""
    ranked = sorted(results, key=lambda r: (r.accuracy() != r.accuracy(), -r.accuracy()))

    headers = ["model", "overall"] + phenomena + ["pos-bias"]
    name_w = max(len("model"), max(len(r.model_name) for r in ranked))
    col_w = 10

    def row(cells: list[str]) -> str:
        first, rest = cells[0], cells[1:]
        return first.ljust(name_w) + "  " + "".join(c.rjust(col_w) for c in rest)

    lines = [row(headers), row(["-" * name_w] + ["-" * col_w for _ in headers[1:]])]
    for r in ranked:
        cells = [r.model_name, _fmt(r.accuracy())]
        cells += [_fmt(r.accuracy(p)) for p in phenomena]
        cells.append(_fmt(r.position_bias()))
        lines.append(row(cells))
    return "\n".join(lines)


def to_json(results: list[ModelResult], phenomena: list[str]) -> dict:
    """Structured results, ready to drop into the front end as the leaderboard data source."""
    return {
        "phenomena": phenomena,
        "models": [
            {
                "name": r.model_name,
                "overall": r.accuracy(),
                "by_phenomenon": {p: r.accuracy(p) for p in phenomena},
                "position_bias": r.position_bias(),
                "n_items": len(r.items),
            }
            for r in sorted(results, key=lambda r: (r.accuracy() != r.accuracy(), -r.accuracy()))
        ],
    }


def write_json(results: list[ModelResult], phenomena: list[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json(results, phenomena), f, indent=2)
