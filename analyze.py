import os, json
from syntaxbench.data import load_pairs
from syntaxbench.harness import evaluate
from syntaxbench.models import RandomBaseline, AlwaysFirst, ShorterSentence, Model
from syntaxbench.report import leaderboard_table

PROMPT = ("Which sentence is grammatically acceptable in standard English? "
          "Reply with only the single letter A or B.\nA: {a}\nB: {b}")

def parse(reply):
    for c in (reply or "").strip().upper():
        if c in ("A", "B"):
            return c
    return "A"

class Claude(Model):
    def __init__(self, model):
        import anthropic
        self.name = f"anthropic:{model}"
        self._m = model
        self._c = anthropic.Anthropic()
    def choose(self, a, b):
        r = self._c.messages.create(model=self._m, max_tokens=5,
            messages=[{"role": "user", "content": PROMPT.format(a=a, b=b)}])
        return parse("".join(x.text for x in r.content if getattr(x, "type", None) == "text"))

raw = {}
for line in open("data/minimal_pairs.jsonl"):
    line = line.strip()
    if line:
        rec = json.loads(line)
        raw[rec["id"]] = rec

pairs = load_pairs("data/minimal_pairs.jsonl")
phen = sorted({p.phenomenon for p in pairs})

models = [RandomBaseline(), AlwaysFirst(), ShorterSentence()]
if os.environ.get("ANTHROPIC_API_KEY"):
    models += [Claude("claude-haiku-4-5-20251001"),
               Claude("claude-sonnet-4-6"),
               Claude("claude-opus-4-8")]
else:
    print('No ANTHROPIC_API_KEY set. Run:  export ANTHROPIC_API_KEY="sk-ant-..."  then rerun.\n')

results = [evaluate(m, pairs) for m in models]
print(leaderboard_table(results, phen))

print("\n" + "=" * 64)
print("WHERE THE REAL MODELS MISSED")
print("=" * 64)
detail = {}
for r in results:
    if ":" not in r.model_name:
        continue
    misses = [it for it in r.items if it.score < 1.0]
    detail[r.model_name] = [it.id for it in misses]
    print(f"\n{r.model_name}  ({len(misses)} of {len(r.items)} items imperfect)")
    if not misses:
        print("  perfect, no misses")
    for it in misses:
        rec = raw[it.id]
        sub = f", {rec['subtype']}" if rec.get("subtype") else ""
        kind = "wrong in BOTH orders" if it.score == 0 else "wrong in ONE order (position-sensitive)"
        print(f"  - {it.id} ({it.phenomenon}{sub}): {kind}")
        print(f"      grammatical:   {rec['good']}")
        print(f"      ungrammatical: {rec['bad']}")

with open("item_misses.json", "w") as f:
    json.dump(detail, f, indent=2)
print("\nSaved the miss list to item_misses.json")
