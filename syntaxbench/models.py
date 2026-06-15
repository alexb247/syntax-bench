"""Model interface, runnable baseline models, and stubs for real models.

Every model implements the same tiny contract:

    choose(option_a, option_b) -> "A" or "B"

The model is shown two sentences with no information about which is grammatical,
and returns which one it judges acceptable. The harness handles presenting each
pair in both orders to control for position bias, so models never need to.
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
# Runnable baselines. These need no API keys and exist to (1) prove the
# pipeline runs end to end and (2) act as honest reference points.
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
# Real-model stubs. Fill these in to populate the leaderboard. Each one only
# needs to turn the two options into a prompt, send it, and parse 'A' or 'B'
# out of the reply. The shared _parse_choice helper does the parsing.
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "Which sentence is grammatically acceptable in standard English? "
    "Reply with only the single letter A or B.\n"
    "A: {a}\n"
    "B: {b}"
)


def _parse_choice(reply: str) -> str:
    """Pull an A or B out of a model's free-text reply. Defaults to 'A' if unclear."""
    for char in reply.strip().upper():
        if char in ("A", "B"):
            return char
    return "A"


class OpenAIModel(Model):
    """TODO: requires `pip install openai` and OPENAI_API_KEY in the environment."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.name = f"openai:{model}"
        self._model = model

    def choose(self, option_a: str, option_b: str) -> str:
        raise NotImplementedError(
            "Plug in the OpenAI client here. Sketch:\n"
            "    from openai import OpenAI\n"
            "    client = OpenAI()\n"
            "    prompt = _PROMPT_TEMPLATE.format(a=option_a, b=option_b)\n"
            "    resp = client.chat.completions.create(model=self._model,\n"
            "        messages=[{'role': 'user', 'content': prompt}], max_tokens=1)\n"
            "    return _parse_choice(resp.choices[0].message.content)"
        )


class AnthropicModel(Model):
    """TODO: requires `pip install anthropic` and ANTHROPIC_API_KEY in the environment."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.name = f"anthropic:{model}"
        self._model = model

    def choose(self, option_a: str, option_b: str) -> str:
        raise NotImplementedError(
            "Plug in the Anthropic client here. Sketch:\n"
            "    import anthropic\n"
            "    client = anthropic.Anthropic()\n"
            "    prompt = _PROMPT_TEMPLATE.format(a=option_a, b=option_b)\n"
            "    resp = client.messages.create(model=self._model, max_tokens=1,\n"
            "        messages=[{'role': 'user', 'content': prompt}])\n"
            "    return _parse_choice(resp.content[0].text)"
        )


class HuggingFaceModel(Model):
    """TODO: requires `pip install transformers torch`. For open models run locally."""

    def __init__(self, model: str = "meta-llama/Llama-3.2-1B-Instruct"):
        self.name = f"hf:{model}"
        self._model = model

    def choose(self, option_a: str, option_b: str) -> str:
        raise NotImplementedError(
            "Load the model with transformers, format the prompt with its chat "
            "template, generate one token, and pass the decoded text to _parse_choice."
        )
