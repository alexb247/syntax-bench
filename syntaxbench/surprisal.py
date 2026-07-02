"""Per-token surprisal from a local causal LM (default: gpt2), CPU only.

This is the only module in the project that touches torch/transformers, and
nothing in the core harness imports it: run.py and multiseed.py stay
stdlib-only. factorial_islands.py imports it lazily, so a missing install
fails there with a pointer to the optional extra:

    pip3 install transformers torch

Surprisal is -log p(token | prefix) in natural log units (nats). A BOS token
is prepended so the first word of the sentence is scored like every other.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SurprisalScorer:
    """Scores sentences with per-token surprisal, summed and averaged."""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._model.to("cpu")
        self._model.eval()
        if self._tokenizer.bos_token_id is None:
            raise ValueError(f"{model_name} has no BOS token; surprisal for the first word would be undefined.")

    @torch.no_grad()
    def score(self, sentence: str) -> dict:
        """Return {"total": nats, "mean": nats/token, "n_tokens": int} for one sentence."""
        ids = self._tokenizer.encode(sentence)
        if not ids:
            raise ValueError("Cannot score an empty sentence.")
        input_ids = torch.tensor([[self._tokenizer.bos_token_id] + ids])
        logits = self._model(input_ids).logits
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
        targets = input_ids[0, 1:]
        surprisals = -log_probs[torch.arange(len(targets)), targets]
        total = float(surprisals.sum())
        return {"total": total, "mean": total / len(targets), "n_tokens": len(targets)}
