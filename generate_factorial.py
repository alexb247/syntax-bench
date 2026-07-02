"""Generate the v2 factorial island bank: matched 2x2 quadruples per subtype.

The v1 forced-choice islands are length-confounded (README, Limitations): the
island structure is inherently longer than its control, so a shorter-sentence
heuristic partly games them. The factorial design fixes that. Each item is a
quadruple crossing

    [island vs non-island structure] x [extraction vs no-extraction]

with the lexical content held constant within the quadruple. The extraction
cost and the structure cost each appear on both sides of the interaction, so
the difference in differences isolates the island effect itself.

The no-extraction members are polar questions, keeping interrogative form and
did-support constant across all four sentences; only the structure under test
and the wh-phrase/gap vary. Self-contained on purpose: shares no vocabulary
pools with generate_items.py, so the v1 bank cannot drift. Reproducible:
fixed seed, running this always yields the same bank.
"""
import json
import random

SEED = 42
N_PER_SUBTYPE = 12

rng = random.Random(SEED)


def sample_unique(make, want, tries=200000):
    """Call make() until we have `want` quadruples with unique sentence sets."""
    seen, out = set(), []
    for _ in range(tries):
        item = make()
        sentences = item["sentences"]
        key = tuple(sorted(sentences.values()))
        if key in seen or len(set(sentences.values())) != len(sentences):
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= want:
            break
    return out


# --------------------------------------------------------------------------
# ADJUNCT: 'that' complement of a bridge verb vs the same clause as an
# adjunct of an intransitive verb.
# --------------------------------------------------------------------------
ADJ_WH_N = ["report", "memo", "contract", "proposal", "letter", "speech", "budget", "article"]
ADJ_SUBJ = ["the editor", "the senator", "the manager", "the lawyer", "the director", "the mayor"]
ADJ_ESUBJ = ["the staff", "the committee", "the interns", "the board", "the team"]
ADJ_EVERB = ["reviewed", "revised", "approved", "drafted", "leaked", "signed", "rejected"]
ADJ_BRIDGE = ["say", "claim", "mention", "insist", "admit"]
# Strictly intransitive only: a verb with even an optional object (e.g.
# "celebrate") would let the extraction gap attach to it, giving the island
# member a grammatical parse -- the defect cut in the v1 validation pass.
ADJ_INTRANS = ["resign", "panic", "object", "complain", "sulk"]
ADJ_CONJ = ["after", "before", "because", "while"]

def make_adjunct():
    wh_n = rng.choice(ADJ_WH_N); subj = rng.choice(ADJ_SUBJ)
    esubj = rng.choice(ADJ_ESUBJ); everb = rng.choice(ADJ_EVERB)
    bridge = rng.choice(ADJ_BRIDGE); intrans = rng.choice(ADJ_INTRANS)
    conj = rng.choice(ADJ_CONJ)
    return {
        "subtype": "adjunct",
        "sentences": {
            "nonisland_noextraction": f"Did {subj} {bridge} that {esubj} {everb} the {wh_n}?",
            "nonisland_extraction":   f"Which {wh_n} did {subj} {bridge} that {esubj} {everb}?",
            "island_noextraction":    f"Did {subj} {intrans} {conj} {esubj} {everb} the {wh_n}?",
            "island_extraction":      f"Which {wh_n} did {subj} {intrans} {conj} {esubj} {everb}?",
        },
        "lexicon": {"wh_noun": wh_n, "subject": subj,
                    "embedded_subject": esubj, "embedded_verb": everb},
        "note": "Non-island: 'that' complement of a bridge verb. Island: the same "
                "clause as an adjunct of an intransitive verb (adjunct island). The "
                "matrix verb necessarily differs with the structure; it is constant "
                "across the extraction contrast, so the interaction subtracts it.",
    }


# --------------------------------------------------------------------------
# COMPLEX NP: a clause under a verb vs the same clause inside a noun phrase.
# --------------------------------------------------------------------------
CNP_WH_N = ["official", "donor", "witness", "suspect", "lobbyist", "diplomat", "banker", "journalist"]
CNP_SUBJ = ["the reporter", "the analyst", "the detective", "the auditor", "the prosecutor", "the columnist"]
CNP_VTHAT = ["claim", "suggest", "allege", "insist", "reveal"]
CNP_NPRED = ["spread the rumor", "make the claim", "deny the rumor",
             "reject the suggestion", "hear the report", "dismiss the idea"]
CNP_ESUBJ = ["the agency", "the firm", "the board", "the bank", "the ministry"]
CNP_EVERB = ["monitored", "paid", "protected", "audited", "bribed", "ignored"]

def make_complex_np():
    wh_n = rng.choice(CNP_WH_N); subj = rng.choice(CNP_SUBJ)
    esubj = rng.choice(CNP_ESUBJ); everb = rng.choice(CNP_EVERB)
    vthat = rng.choice(CNP_VTHAT); npred = rng.choice(CNP_NPRED)
    return {
        "subtype": "complex_NP",
        "sentences": {
            "nonisland_noextraction": f"Did {subj} {vthat} that {esubj} {everb} the {wh_n}?",
            "nonisland_extraction":   f"Which {wh_n} did {subj} {vthat} that {esubj} {everb}?",
            "island_noextraction":    f"Did {subj} {npred} that {esubj} {everb} the {wh_n}?",
            "island_extraction":      f"Which {wh_n} did {subj} {npred} that {esubj} {everb}?",
        },
        "lexicon": {"wh_noun": wh_n, "subject": subj,
                    "embedded_subject": esubj, "embedded_verb": everb},
        "note": "Non-island: 'that' clause under a verb. Island: the same clause "
                "inside a noun phrase (complex NP island).",
    }


# --------------------------------------------------------------------------
# WH: extraction across 'that' vs across 'whether/if' (a weak, gradient island).
# --------------------------------------------------------------------------
WH_WH_N = ["contract", "proposal", "document", "plan", "draft", "bill", "manuscript", "petition"]
WH_SUBJ = ["the lawyer", "the editor", "the manager", "the professor", "the consultant", "the architect"]
WH_BRIDGE = ["say", "think", "believe", "claim", "assume"]
WH_COMP = ["wonder whether", "ask whether", "wonder if", "ask if"]
WH_ESUBJ = ["the clerk", "the committee", "the partners", "the trustees", "the council"]
WH_EVERB = ["approved", "reviewed", "endorsed", "finalized", "rejected", "drafted"]

def make_wh():
    wh_n = rng.choice(WH_WH_N); subj = rng.choice(WH_SUBJ)
    esubj = rng.choice(WH_ESUBJ); everb = rng.choice(WH_EVERB)
    bridge = rng.choice(WH_BRIDGE); comp = rng.choice(WH_COMP)
    return {
        "subtype": "wh",
        "sentences": {
            "nonisland_noextraction": f"Did {subj} {bridge} that {esubj} {everb} the {wh_n}?",
            "nonisland_extraction":   f"Which {wh_n} did {subj} {bridge} that {esubj} {everb}?",
            "island_noextraction":    f"Did {subj} {comp} {esubj} {everb} the {wh_n}?",
            "island_extraction":      f"Which {wh_n} did {subj} {comp} {esubj} {everb}?",
        },
        "lexicon": {"wh_noun": wh_n, "subject": subj,
                    "embedded_subject": esubj, "embedded_verb": everb},
        "note": "Non-island: extraction across 'that'. Island: extraction across "
                "'whether/if' (wh-island; weak and gradient).",
    }


# --------------------------------------------------------------------------
# SUBJECT: extraction from an NP in object position vs from the same NP as
# subject.
# --------------------------------------------------------------------------
SBJ_WH_N = ["senator", "actor", "candidate", "author", "executive", "athlete", "singer", "governor"]
SBJ_SUBJ = ["the magazine", "the newspaper", "the tabloid", "the website", "the network", "the broadcaster"]
SBJ_TVERB = ["publish", "run", "print", "post"]
SBJ_NP = [("story", "about"), ("photo", "of"), ("report", "on"),
          ("article", "about"), ("profile", "of"), ("column", "about")]
SBJ_TVERB2 = ["upset", "offend", "embarrass", "anger", "annoy"]

def make_subject():
    wh_n = rng.choice(SBJ_WH_N); subj = rng.choice(SBJ_SUBJ)
    tverb = rng.choice(SBJ_TVERB); noun, prep = rng.choice(SBJ_NP)
    tverb2 = rng.choice(SBJ_TVERB2)
    art = "an" if noun[0] in "aeiou" else "a"
    return {
        "subtype": "subject",
        "sentences": {
            "nonisland_noextraction": f"Did {subj} {tverb} {art} {noun} {prep} the {wh_n}?",
            "nonisland_extraction":   f"Which {wh_n} did {subj} {tverb} {art} {noun} {prep}?",
            "island_noextraction":    f"Did {art} {noun} {prep} the {wh_n} {tverb2} {subj}?",
            "island_extraction":      f"Which {wh_n} did {art} {noun} {prep} {tverb2} {subj}?",
        },
        "lexicon": {"wh_noun": wh_n, "subject": subj,
                    "np_noun": noun, "preposition": prep},
        "note": "Non-island: extraction from an NP in object position. Island: "
                "extraction from the same NP as subject (subject island). The matrix "
                "verb necessarily differs with the NP's role; it is constant across "
                "the extraction contrast, so the interaction subtracts it.",
    }


MAKERS = (
    ("adjunct", make_adjunct),
    ("complex_NP", make_complex_np),
    ("wh", make_wh),
    ("subject", make_subject),
)


def build():
    quads = []
    for subtype, maker in MAKERS:
        for i, item in enumerate(sample_unique(maker, N_PER_SUBTYPE), start=1):
            quads.append({"id": f"fac-{subtype}-{i:03d}", **item})
    return quads


if __name__ == "__main__":
    quads = build()
    with open("data/factorial_islands.jsonl", "w") as f:
        for q in quads:
            f.write(json.dumps(q) + "\n")
    from collections import Counter
    print("total quadruples:", len(quads))
    print("by subtype:", dict(Counter(q["subtype"] for q in quads)))
