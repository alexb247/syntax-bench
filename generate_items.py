"""Generate the controlled minimal-pair bank from templates with a fixed seed.

Each phenomenon is produced by a template that holds length and surface cues
constant and varies only the structural feature under test. Reproducible:
running this always yields the same bank.
"""
import json, random, itertools

SEED = 42
N_THAT_TRACE = 50
N_NPI = 50
N_ISLAND_PER_SUBTYPE = 13

rng = random.Random(SEED)


def sample_unique(make, want, tries=200000):
    """Call make() until we have `want` unique (good,bad) pairs."""
    seen, out = set(), []
    for _ in range(tries):
        item = make()
        key = (item["good"], item["bad"])
        if key in seen or item["good"] == item["bad"]:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= want:
            break
    return out


# --------------------------------------------------------------------------
# THAT-TRACE: object extraction across 'that' (good) vs subject extraction (bad).
# One NP, one verb; only the gap position moves, so the pair is length-matched.
# --------------------------------------------------------------------------
TT_WH = ["Who", "Which candidate", "Which lawyer", "Which official", "Which manager",
         "Which executive", "Which player", "Which scientist", "Which author",
         "Which suspect", "Which witness", "Which student"]
TT_MATRIX = ["do you", "do they", "do we", "does she", "does he", "did you",
             "did they", "did she", "did he", "did the senator", "did the reporter"]
TT_MVERB = ["believe", "think", "suppose", "imagine", "assume", "claim", "expect",
            "figure", "guess", "decide", "conclude", "argue", "insist"]
TT_NP = ["the board", "the manager", "the committee", "the agency", "the senator",
         "the panel", "the judge", "the coach", "the editor", "the officer",
         "the director", "the jury", "the firm", "the council", "the dean"]
TT_EVERB = ["hired", "promoted", "praised", "criticized", "recruited", "dismissed",
            "selected", "nominated", "blamed", "endorsed", "investigated",
            "interviewed", "rejected", "sued", "fired"]

def make_tt():
    wh = rng.choice(TT_WH); m = rng.choice(TT_MATRIX); v = rng.choice(TT_MVERB)
    np = rng.choice(TT_NP); ev = rng.choice(TT_EVERB)
    return {"phenomenon": "that-trace", "condition": "controlled",
            "good": f"{wh} {m} {v} that {np} {ev}?",
            "bad":  f"{wh} {m} {v} that {ev} {np}?",
            "note": "Object extraction across overt 'that' is grammatical; subject extraction across 'that' is barred (that-trace effect). Same words, so length is held constant."}


# --------------------------------------------------------------------------
# NPI: an NPI in the relative clause (downward-entailing restrictor of 'every',
# good) vs the same NPI in the main predicate (upward-entailing scope, bad).
# --------------------------------------------------------------------------
NPI_N = ["author", "patient", "volunteer", "customer", "student", "witness",
         "employee", "applicant", "contestant", "tenant", "passenger", "vendor",
         "official", "resident", "member", "client", "trainee", "supplier"]
NPI_PRED = ["eventually dropped", "given a warning", "flagged for review",
            "asked to leave", "offered a refund", "contacted again",
            "placed on probation", "removed from the list", "called back",
            "thanked", "reimbursed", "investigated", "escalated", "disqualified",
            "interviewed again", "rewarded"]
NPI_EVER_V = ["lied", "cheated", "complained", "hesitated", "objected",
              "protested", "faltered", "transgressed", "misbehaved", "defaulted"]
NPI_ATALL_V = ["complained", "hesitated", "contributed", "participated", "waited",
               "objected", "struggled", "wavered", "engaged", "responded"]
NPI_SLIGHT_V = ["complained", "hesitated", "wavered", "objected", "faltered"]
NPI_FINGER_INF = ["help", "assist", "contribute", "intervene", "cooperate"]

def make_npi():
    n = rng.choice(NPI_N); pred = rng.choice(NPI_PRED)
    pat = rng.choice(["ever", "atall", "slight", "finger"])
    if pat == "ever":
        clause = f"had ever {rng.choice(NPI_EVER_V)}"
    elif pat == "atall":
        clause = f"had {rng.choice(NPI_ATALL_V)} at all"
    elif pat == "slight":
        clause = f"had {rng.choice(NPI_SLIGHT_V)} in the slightest"
    else:
        clause = f"had lifted a finger to {rng.choice(NPI_FINGER_INF)}"
    return {"phenomenon": "npi", "condition": "controlled",
            "good": f"Every {n} who {clause} was {pred}.",
            "bad":  f"Every {n} who was {pred} {clause}.",
            "note": "The NPI is licensed in the relative clause (the downward-entailing restrictor of 'every') but not in the main predicate (the upward-entailing nuclear scope). Same words; only the NPI's position changes."}


# --------------------------------------------------------------------------
# ISLANDS: four subtypes, each a controlled extraction contrast.
# --------------------------------------------------------------------------
ISL_ADJ_WH = ["Which book", "Which song", "Which proposal", "Which film", "Which report",
              "Which letter", "Which contract", "Which speech", "Which article", "Which memo"]
ISL_ADJ_SUBJ = ["John", "Mary", "the senator", "the manager", "the editor",
                "the artist", "the lawyer", "the student", "the director", "the author"]
ISL_ADJ_BRIDGE = ["say", "claim", "argue", "insist", "mention", "reveal", "admit"]
ISL_ADJ_ESUBJ = ["he", "she", "they", "the staff", "the team", "the committee"]
ISL_ADJ_EVERB = ["reviewed", "revised", "approved", "rejected", "edited",
                 "signed", "leaked", "drafted", "finalized", "released"]
ISL_ADJ_INTRANS = ["resign", "cry", "quit", "leave", "celebrate", "panic", "object", "relax"]
ISL_ADJ_CONJ = ["after", "before", "because", "when", "while"]

def make_isl_adjunct():
    wh = rng.choice(ISL_ADJ_WH); s = rng.choice(ISL_ADJ_SUBJ); es = rng.choice(ISL_ADJ_ESUBJ); ev = rng.choice(ISL_ADJ_EVERB)
    return {"phenomenon": "island", "condition": "controlled", "subtype": "adjunct",
            "good": f"{wh} did {s} {rng.choice(ISL_ADJ_BRIDGE)} that {es} {ev}?",
            "bad":  f"{wh} did {s} {rng.choice(ISL_ADJ_INTRANS)} {rng.choice(ISL_ADJ_CONJ)} {es} {ev}?",
            "note": "Extraction from a 'that' complement is fine; extraction from an adjunct clause is barred (adjunct island)."}

ISL_SUBJ_WH = ["Which candidate", "Which actor", "Which senator", "Which author",
               "Which executive", "Which player", "Which artist", "Which official"]
ISL_SUBJ_SUBJ = ["the reporters", "the magazine", "the paper", "the editors",
                 "the columnist", "the tabloid", "the website", "the bloggers"]
ISL_SUBJ_TVERB = ["publish", "run", "print", "post"]
ISL_SUBJ_NP = [("story", "about"), ("photo", "of"), ("report", "on"),
               ("article", "about"), ("profile", "of"), ("column", "about"), ("piece", "about")]
ISL_SUBJ_TVERB2 = ["upset", "offend", "embarrass", "anger", "surprise", "alarm", "annoy"]

def make_isl_subject():
    wh = rng.choice(ISL_SUBJ_WH); s = rng.choice(ISL_SUBJ_SUBJ); tv = rng.choice(ISL_SUBJ_TVERB)
    n, prep = rng.choice(ISL_SUBJ_NP); tv2 = rng.choice(ISL_SUBJ_TVERB2)
    return {"phenomenon": "island", "condition": "controlled", "subtype": "subject",
            "good": f"{wh} did {s} {tv} a {n} {prep}?",
            "bad":  f"{wh} did a {n} {prep} {tv2} {s}?",
            "note": "The same phrase allows extraction in object position but not as the subject (subject island). Length-matched."}

ISL_CNP_WH = ["Who", "Which suspect", "Which official", "Which witness", "Which donor"]
ISL_CNP_SUBJ = ["the senator", "the analyst", "the reporter", "the detective", "the auditor", "the prosecutor"]
ISL_CNP_VTHAT = ["claim", "suggest", "reveal", "allege", "report", "insist"]
ISL_CNP_NOUN = [("make the claim"), ("reject the suggestion"), ("deny the rumor"),
                ("dismiss the idea"), ("hear the report"), ("spread the rumor"), ("confirm the allegation")]
ISL_CNP_ESUBJ = ["the agency", "the firm", "the company", "the board", "the official", "the bank"]
ISL_CNP_EVERB = ["monitored", "concealed", "paid", "investigated", "protected", "audited", "ignored"]

def make_isl_cnp():
    wh = rng.choice(ISL_CNP_WH); s = rng.choice(ISL_CNP_SUBJ); es = rng.choice(ISL_CNP_ESUBJ); ev = rng.choice(ISL_CNP_EVERB)
    return {"phenomenon": "island", "condition": "controlled", "subtype": "complex_NP",
            "good": f"{wh} did {s} {rng.choice(ISL_CNP_VTHAT)} that {es} {ev}?",
            "bad":  f"{wh} did {s} {rng.choice(ISL_CNP_NOUN)} that {es} {ev}?",
            "note": "A clause under a verb allows extraction; the same clause inside a noun phrase blocks it (complex NP island)."}

ISL_WH_WH = ["Which proposal", "Which document", "Which plan", "Which report",
             "Which contract", "Which book", "Which article", "Which memo", "Which draft", "Which bill"]
ISL_WH_SUBJ = ["she", "he", "the lawyer", "the manager", "they", "the editor"]
ISL_WH_BRIDGE = ["think", "believe", "assume", "say", "claim"]
ISL_WH_COMP = [("wonder whether"), ("ask whether"), ("wonder if"), ("ask if"), ("inquire whether")]
ISL_WH_ESUBJ = ["we", "they", "the clerk", "the team"]
ISL_WH_EVERB = ["should approve", "would discuss", "should release", "would review",
                "should publish", "would reject", "should finalize", "would endorse"]

def make_isl_wh():
    wh = rng.choice(ISL_WH_WH); s = rng.choice(ISL_WH_SUBJ); es = rng.choice(ISL_WH_ESUBJ); ev = rng.choice(ISL_WH_EVERB)
    return {"phenomenon": "island", "condition": "controlled", "subtype": "wh",
            "good": f"{wh} did {s} {rng.choice(ISL_WH_BRIDGE)} that {es} {ev}?",
            "bad":  f"{wh} did {s} {rng.choice(ISL_WH_COMP)} {es} {ev}?",
            "note": "Extraction across 'that' is fine; extraction across 'whether/if' crosses a wh-island. This is a weak, gradient island."}


def build():
    items = []
    items += sample_unique(make_tt, N_THAT_TRACE)
    items += sample_unique(make_npi, N_NPI)
    for maker in (make_isl_adjunct, make_isl_subject, make_isl_cnp, make_isl_wh):
        items += sample_unique(maker, N_ISLAND_PER_SUBTYPE)
    # stable ids per phenomenon
    counters = {}
    prefix = {"that-trace": "tt", "npi": "npi", "island": "isl"}
    for it in items:
        p = it["phenomenon"]; counters[p] = counters.get(p, 0) + 1
        it_id = f"{prefix[p]}-{counters[p]:03d}"
        it_out = {"id": it_id, **it}
        items[items.index(it)] = it_out
    return items


if __name__ == "__main__":
    items = build()
    with open("data/minimal_pairs.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    from collections import Counter
    print("total:", len(items))
    print("by phenomenon:", dict(Counter(i["phenomenon"] for i in items)))
    print("island subtypes:", dict(Counter(i.get("subtype", "-") for i in items if i["phenomenon"] == "island")))
