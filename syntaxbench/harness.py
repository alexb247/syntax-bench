"""The evaluation harness.

For each minimal pair we run the model twice:
  - order 1: the grammatical sentence is option A
  - order 2: the grammatical sentence is option B

A model that genuinely judges grammar should be right in both orders. A model
that just favors a position will be right in one and wrong in the other. The
per-item score is the average of the two (0.0, 0.5, or 1.0), which cancels out
position bias in the headline number while still letting us measure it.
"""

from dataclasses import dataclass, field

from .data import MinimalPair
from .models import Model


@dataclass
class ItemResult:
    id: str
    phenomenon: str
    correct_good_first: bool   # was the model right when good was option A
    correct_good_second: bool  # was the model right when good was option B

    @property
    def score(self) -> float:
        return (int(self.correct_good_first) + int(self.correct_good_second)) / 2.0


@dataclass
class ModelResult:
    model_name: str
    items: list[ItemResult] = field(default_factory=list)

    def _subset(self, phenomenon: str | None) -> list[ItemResult]:
        if phenomenon is None:
            return self.items
        return [it for it in self.items if it.phenomenon == phenomenon]

    def accuracy(self, phenomenon: str | None = None) -> float:
        subset = self._subset(phenomenon)
        if not subset:
            return float("nan")
        return sum(it.score for it in subset) / len(subset)

    def position_bias(self) -> float:
        """How much the model's accuracy depends on where the good sentence sits.
        0.0 means no positional preference; higher means more bias."""
        if not self.items:
            return float("nan")
        acc_good_first = sum(it.correct_good_first for it in self.items) / len(self.items)
        acc_good_second = sum(it.correct_good_second for it in self.items) / len(self.items)
        return abs(acc_good_first - acc_good_second)

    @property
    def phenomena(self) -> list[str]:
        return sorted({it.phenomenon for it in self.items})


def evaluate(model: Model, pairs: list[MinimalPair]) -> ModelResult:
    """Run one model over every pair in both orders."""
    results: list[ItemResult] = []
    for pair in pairs:
        # Order 1: good sentence is A. Correct answer is "A".
        good_first = model.choose(pair.good, pair.bad) == "A"
        # Order 2: good sentence is B. Correct answer is "B".
        good_second = model.choose(pair.bad, pair.good) == "B"
        results.append(
            ItemResult(
                id=pair.id,
                phenomenon=pair.phenomenon,
                correct_good_first=good_first,
                correct_good_second=good_second,
            )
        )
    return ModelResult(model_name=model.name, items=results)
