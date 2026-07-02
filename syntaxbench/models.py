"""Model interface, runnable baseline models, and real model adapters.

Every model implements the same tiny contract:

    choose(option_a, option_b) -> "A" or "B"

The model is shown two sentences with no information about which is grammatical,
and returns which one it judges acceptable. The harness handles presenting each
pair in both orders to control for position bias, so models never need to.

The real-model classes import their SDK lazily inside __init__, so this file
still loads (and the baselines still run) even when those packages are not
installed.
"""

import random
from abc import ABC, abstractmethod


class Model(ABC):
    """Base class. A model picks the grammatical sentence given two options."""

    name: str = "unnamed"

    @abstractmethod
    def choose(self, option_a: str, option_b: str) -> str:
        """Return 'A' or 'B'."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Runnable baselines. No API keys. They (1) prove the pipeline runs end to end
# and (2) act as honest reference points the real models must beat.
# ---------------------------------------------------------------------------

class RandomBaseline(Model):
    """Chance performance. Anything that cannot beat this has learned nothing."""

    name = "random-baseline"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def choose(self, option_a: str, option_b: str) -> str:
        return self._rng.choice(["A", "B"])


class AlwaysFirst(Model):
    """Always answers 'A'. Degenerate on purpose: its per-order numbers expose
    why the two-order position-bias control matters."""

    name = "always-first"

    def choose(self, option_a: str, option_b: str) -> str:
        return "A"


class ShorterSentence(Model):
    """Picks the shorter sentence. A trivial heuristic with no syntax in it.
    If it beats chance on a phenomenon, the items for that phenomenon have a
    length confound and are gameable. Use it as a probe, not a contender."""

    name = "shorter-sentence-heuristic"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def choose(self, option_a: str, option_b: str) -> str:
        if len(option_a) < len(option_b):
            return "A"
        if len(option_b) < len(option_a):
            return "B"
        return self._rng.choice(["A", "B"])


# ---------------------------------------------------------------------------
# Prompt and parsing shared by the real models.
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "Which sentence is grammatically acceptable in standard English? "
    "Reply with only the single letter A or B.\n"
    "A: {a}\n"
    "B: {b}"
)


def _parse_choice(reply: str) -> str:
    """Pull an A or B out of a model's free-text reply. Defaults to 'A' if unclear."""
    for char in (reply or "").strip().upper():
        if char in ("A", "B"):
            return char
    return "A"


# ---------------------------------------------------------------------------
# Real models. The leaderboard fills in once you register these in run.py.
# ---------------------------------------------------------------------------

class AnthropicModel(Model):
    """Claude via the Anthropic API. Requires `pip install anthropic` and the
    ANTHROPIC_API_KEY environment variable (the client reads it automatically)."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.name = f"anthropic:{model}"
        self._model = model
        self._client = anthropic.Anthropic()

    def choose(self, option_a: str, option_b: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=5,
            messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(a=option_a, b=option_b)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return _parse_choice(text)


class OpenAIModel(Model):
    """GPT via the OpenAI API. Requires `pip install openai` and the
    OPENAI_API_KEY environment variable."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.name = f"openai:{model}"
        self._model = model
        self._client = OpenAI()

    def choose(self, option_a: str, option_b: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=5,
            messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(a=option_a, b=option_b)}],
        )
        return _parse_choice(resp.choices[0].message.content or "")


class HuggingFaceModel(Model):
    """TODO: an open model run locally. Requires `pip install transformers torch`.
    Fill this in when you want to compare open models against the API ones."""

    def __init__(self, model: str = "meta-llama/Llama-3.2-1B-Instruct"):
        self.name = f"hf:{model}"
        self._model = model

    def choose(self, option_a: str, option_b: str) -> str:
        raise NotImplementedError(
            "Load the model with transformers, format the prompt with its chat "
            "template, generate a few tokens, and pass the decoded text to _parse_choice."
        )
