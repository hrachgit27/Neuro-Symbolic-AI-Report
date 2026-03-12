"""
Neuro-Symbolic Reasoning — Proper 2-Way Comparison
====================================================
Pipeline A : LLM-Only
    The LLM reads the theory and answers directly.
    Pure neural. No symbolic component.

Pipeline B : Neuro-Symbolic Loop
    A genuine bidirectional integration:
    1. Symbolic engine parses theory into facts + rules
    2. Symbolic engine finds all rules that COULD fire given current facts
    3. LLM selects which rule is most relevant to the question
    4. Symbolic engine applies that rule → new fact added to working memory
    5. Repeat steps 2-4 until the question is answerable or no rules remain
    6. SYMBOLIC ENGINE produces the final verdict (True/False/Unknown)
       — the LLM never decides the answer, only guides the search

    This is genuine neuro-symbolic reasoning:
    - Neural component  → guides which rules to pursue (heuristic search)
    - Symbolic component → executes deductions and decides the answer (correctness)

Dataset : tasksource/proofwriter  —  OWA depth-0 through depth-5
Model   : Ollama Llama3 (local)

Install:
    pip install requests datasets

Run:
    python neuro_symbolic_compare.py            # 10 per depth = 60 total
    python neuro_symbolic_compare.py --n 50     # 50 per depth = 300 total
    python neuro_symbolic_compare.py --model deepseek-r1:7b --n 10

Outputs:
    results.json  — full per-problem data for both pipelines
    summary.json  — accuracy + depth breakdown
"""

import re
import json
import time
import argparse
import requests
from datetime import datetime
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "llama3"
MAX_TOKENS   = 150
TEMPERATURE  = 0
MAX_NS_STEPS = 15   # max rule applications in the neuro-symbolic loop

OWA_CONFIGS  = ["depth-0", "depth-1", "depth-2", "depth-3", "depth-4", "depth-5"]
RESULTS_FILE = "results.json"
SUMMARY_FILE = "summary.json"

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Neuro-Symbolic vs LLM-Only on ProofWriter OWA")
    p.add_argument("--n",     type=int, default=10,  help="Problems per depth level (default: 10)")
    p.add_argument("--split", default="test",         help="Dataset split: train/validation/test")
    p.add_argument("--model", default=None,           help="Override Ollama model name")
    return p.parse_args()

# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_proofwriter_owa(split, n_per_depth):
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Run: pip install datasets")

    print(f"  Loading ProofWriter OWA (split={split}, {n_per_depth} per depth × 6 levels)...")
    ds      = load_dataset("tasksource/proofwriter", split=split, streaming=True)
    buckets = defaultdict(list)

    for row in ds:
        cfg = row.get("config", "")
        if cfg not in OWA_CONFIGS:
            continue
        if len(buckets[cfg]) >= n_per_depth:
            continue
        buckets[cfg].append({
            "id":       row["id"],
            "theory":   row["theory"],
            "question": row["question"],
            "answer":   row["answer"].strip().lower(),
            "depth":    row["QDep"],
            "config":   cfg,
            "nfact":    row.get("NFact", 0),
            "nrule":    row.get("NRule", 0),
        })
        if all(len(buckets[c]) >= n_per_depth for c in OWA_CONFIGS):
            break

    problems = []
    for cfg in OWA_CONFIGS:
        problems.extend(buckets[cfg])
        print(f"    {cfg}: {len(buckets[cfg])} problems loaded")

    print(f"\n  Total: {len(problems)} problems.\n")
    return problems

# ─── Symbolic Parser ──────────────────────────────────────────────────────────

# All relational verbs used in ProofWriter
RELATIONAL_VERBS = (
    r"needs?|chases?|eats?|likes?|sees?|visits?|is friends? with|"
    r"helps?|hides?|holds?|hunts?|knows?|loves?|makes?|owns?|"
    r"plays? with|rounds?|supports?|wants?"
)

def parse_knowledge_base(theory):
    """
    Parse a ProofWriter theory into:
      facts : set of known true statements
      rules : list of inference rules

    Handles:
      - Single and multi-word subjects ("Gary", "the bald eagle")
      - Attribute facts:    "X is [not] Y"
      - Relational facts:   "X needs/chases/eats/... Y"
      - forall rules:       "All X are Y" / "X things are Y"
      - generic_if rules:   "If someone/something/it verb... then ... "
      - specific_if rules:  "If <subj> verb... then <subj> verb..."
      - relational rules:   "If something verb X then it verb Y"
    """
    facts = set()
    rules = []
    sentences = [s.strip() for s in re.split(r"\.\s*", theory) if s.strip()]

    for sent in sentences:
        low = sent.lower()

        # ── Rules — must start with "if" or "all" ─────────────────────
        if low.startswith("if ") or low.startswith("all "):

            # forall: "All X are [not] Y" / "All X things/people are [not] Y"
            # Also handles comma-separated: "All round, cold people are blue"
            m = re.match(
                r"^all ([\w][\w\s,]*?)(?:\s+things?|\s+people|\s+animals?)?\s+are (not )?(.+)$", low
            )
            if m:
                rules.append({
                    "type": "forall",
                    "kind": m.group(1).strip(),
                    "negated": bool(m.group(2)),
                    "property": m.group(3).strip(),
                    "raw": sent
                })
                continue

            # forall shorthand: "X things are [not] Y"
            m = re.match(r"^([\w][\w\s,]*?)\s+things?\s+are (not )?(.+)$", low)
            if m:
                rules.append({
                    "type": "forall",
                    "kind": m.group(1).strip(),
                    "negated": bool(m.group(2)),
                    "property": m.group(3).strip(),
                    "raw": sent
                })
                continue

            # forall implicit: "X, Y people/things are Z" without "All" prefix
            # e.g. "Round, cold people are blue"
            m = re.match(
                r"^([\w][\w\s]*(?:,\s*[\w][\w\s]*)+)\s+(?:people|things?|animals?)\s+are (not )?(.+)$", low
            )
            if m:
                rules.append({
                    "type": "forall",
                    "kind": m.group(1).strip(),
                    "negated": bool(m.group(2)),
                    "property": m.group(3).strip(),
                    "raw": sent
                })
                continue

            # generic_if (attribute → relational): "If something is X then it verb Y"
            # e.g. "If something is round then it eats the cow"
            m = re.match(
                r"^if (?:someone|something|it) is (.+?) then (?:they|it) (" + RELATIONAL_VERBS + r") (.+)$",
                low
            )
            if m:
                rules.append({
                    "type": "generic_if_attr_to_rel",
                    "condition": m.group(1).strip(),
                    "cons_verb": m.group(2).strip(),
                    "cons_obj":  m.group(3).strip(),
                    "raw": sent
                })
                continue

            # generic_if (attribute): "If someone/something is X [and not Y] then they/it are/is [not] Z"
            m = re.match(
                r"^if (?:someone|something|it) is (.+?) then (?:they|it|the \w+) (?:are|is) (not )?(.+)$", low
            )
            if m:
                rules.append({
                    "type": "generic_if_attr",
                    "condition": m.group(1).strip(),
                    "negated": bool(m.group(2)),
                    "consequent": m.group(3).strip(),
                    "raw": sent
                })
                continue

            # generic_if (relational): "If something verb X then it verb Y"
            # e.g. "If something visits the bear then the bear sees the rabbit"
            m = re.match(
                r"^if (?:someone|something|it) (" + RELATIONAL_VERBS + r") (.+?) then (.+?) (" + RELATIONAL_VERBS + r") (.+)$",
                low
            )
            if m:
                rules.append({
                    "type": "generic_if_rel",
                    "cond_verb": m.group(1).strip(),
                    "cond_obj": m.group(2).strip(),
                    "cons_subj": m.group(3).strip(),
                    "cons_verb": m.group(4).strip(),
                    "cons_obj": m.group(5).strip(),
                    "raw": sent
                })
                continue

            # generic_if (relational → attribute): "If something verb X then it is Y"
            m = re.match(
                r"^if (?:someone|something|it) (" + RELATIONAL_VERBS + r") (.+?) then (?:they|it) (?:are|is) (not )?(.+)$",
                low
            )
            if m:
                rules.append({
                    "type": "generic_if_rel_to_attr",
                    "cond_verb": m.group(1).strip(),
                    "cond_obj": m.group(2).strip(),
                    "negated": bool(m.group(3)),
                    "consequent": m.group(4).strip(),
                    "raw": sent
                })
                continue

            # specific_if (attr → rel): "If the squirrel is round then the squirrel visits the cow"
            m = re.match(
                r"^if ([\w][\w\s]*?) is ([\w][\w\s]*?) then ([\w][\w\s]*?) (" + RELATIONAL_VERBS + r") (.+)$",
                low
            )
            if m:
                rules.append({
                    "type":       "specific_if_attr_to_rel",
                    "subj":       m.group(1).strip(),
                    "prop":       m.group(2).strip(),
                    "cons_subj":  m.group(3).strip(),
                    "cons_verb":  m.group(4).strip(),
                    "cons_obj":   m.group(5).strip(),
                    "raw":        sent
                })
                continue

            # specific_if: "If <subj> is X [and <subj> is Y] then <subj> is [not] Z"
            m = re.match(
                r"^if ([\w][\w\s]*?) is ([\w][\w\s]*?)"
                r"(?: and ([\w][\w\s]*?) is ([\w][\w\s]*?))?"
                r" then ([\w][\w\s]*?) is (not )?(.+)$",
                low
            )
            if m:
                rules.append({
                    "type": "specific_if",
                    "subj1": m.group(1).strip(), "prop1": m.group(2).strip(),
                    "subj2": m.group(3).strip() if m.group(3) else None,
                    "prop2": m.group(4).strip() if m.group(4) else None,
                    "cons_subj": m.group(5).strip(),
                    "negated": bool(m.group(6)),
                    "consequent": m.group(7).strip(),
                    "raw": sent
                })
                continue

            # specific_if (relational): "If <subj> verb <obj> [and ...] then <subj> verb <obj>"
            m = re.match(
                r"^if ([\w][\w\s]*?) (" + RELATIONAL_VERBS + r") ([\w][\w\s]*?)"
                r"(?: and ([\w][\w\s]*?) (" + RELATIONAL_VERBS + r") ([\w][\w\s]*?))?"
                r" then ([\w][\w\s]*?) (?:is (not )?([\w][\w\s]*)|(" + RELATIONAL_VERBS + r") ([\w][\w\s]*))$",
                low
            )
            if m:
                rules.append({
                    "type": "specific_if_rel",
                    "subj1": m.group(1).strip(), "verb1": m.group(2).strip(), "obj1": m.group(3).strip(),
                    "subj2": m.group(4).strip() if m.group(4) else None,
                    "verb2": m.group(5).strip() if m.group(5) else None,
                    "obj2":  m.group(6).strip() if m.group(6) else None,
                    "cons_subj": m.group(7).strip(),
                    "cons_neg": bool(m.group(8)),
                    "cons_attr": m.group(9).strip() if m.group(9) else None,
                    "cons_verb": m.group(10).strip() if m.group(10) else None,
                    "cons_obj":  m.group(11).strip() if m.group(11) else None,
                    "raw": sent
                })
                continue

            # Catch-all: store unparsed rules as raw so they don't leak into facts
            rules.append({"type": "unparsed", "raw": sent})
            continue

        # ── Facts — only reached for non-if/non-all sentences ──────────

        # Implicit forall: "X, Y people/things are Z" (no "All" prefix)
        # e.g. "Round, cold people are blue" / "Smart, red people are rough"
        m = re.match(
            r"^([\w][\w\s]*(?:,\s*[\w][\w\s]*)+)\s+(?:people|things?|animals?)\s+are (not )?(.+)$", low
        )
        if m:
            rules.append({
                "type": "forall",
                "kind": m.group(1).strip(),
                "negated": bool(m.group(2)),
                "property": m.group(3).strip(),
                "raw": sent
            })
            continue

        # "X is not Y"
        m = re.match(r"^([\w][\w\s]*?) is not (.+)$", low)
        if m:
            facts.add(f"NOT({m.group(1).strip()} is {m.group(2).strip()})")
            continue

        # "X does not verb Y" — negated relational fact
        m = re.match(r"^([\w][\w\s]*?) does not (" + RELATIONAL_VERBS + r") ([\w][\w\s]*)$", low)
        if m:
            subj = m.group(1).strip()
            verb = m.group(2).strip()
            obj  = m.group(3).strip()
            # Normalize to base verb form for consistent storage
            verb_s = verb if verb.endswith("s") else verb + "s"
            facts.add(f"NOT({subj} {verb_s} {obj})")
            continue

        # "X is Y"
        m = re.match(r"^([\w][\w\s]*?) is ([\w][\w\s]*)$", low)
        if m:
            facts.add(f"{m.group(1).strip()} is {m.group(2).strip()}")
            continue

        # Relational fact: "X verb Y"
        m = re.match(r"^([\w][\w\s]*?) (" + RELATIONAL_VERBS + r") ([\w][\w\s]*)$", low)
        if m:
            facts.add(f"{m.group(1).strip()} {m.group(2).strip()} {m.group(3).strip()}")
            continue

    return facts, rules

# ─── Symbolic Engine ──────────────────────────────────────────────────────────

def get_subjects_with_attr(facts, attr):
    """Return all subjects that have a given attribute in the fact base."""
    subjects = set()
    for fact in facts:
        m = re.match(r"^([\w][\w\s]*?) is (.+)$", fact)
        if m and m.group(2).strip() == attr:
            subjects.add(m.group(1).strip())
    return subjects


def get_subjects_with_rel(facts, verb, obj):
    """Return all subjects that have a given relational fact, handling multi-word subjects."""
    subjects = set()
    # Normalize verb to both base and -s form for matching
    verb_s    = verb if verb.endswith("s") else verb + "s"
    verb_base = verb.rstrip("s")
    for fact in facts:
        # Use RELATIONAL_VERBS to split at the verb boundary
        m = re.match(r"^([\w][\w\s]*?) (" + RELATIONAL_VERBS + r") (.+)$", fact)
        if m:
            f_subj = m.group(1).strip()
            f_verb = m.group(2).strip()
            f_obj  = m.group(3).strip()
            if f_obj == obj and f_verb in (verb, verb_s, verb_base):
                subjects.add(f_subj)
    return subjects


def get_applicable_rules(facts, rules):
    """
    Return all rules that CAN fire given the current fact base
    but haven't yet produced their consequent.
    """
    applicable = []

    def add(rule, new_fact):
        if new_fact not in facts:
            applicable.append({"rule": rule, "would_derive": new_fact})

    for rule in rules:
        rtype = rule["type"]
        if rtype == "unparsed":
            continue

        neg = rule.get("negated", False)

        # ── forall: "All X are Y" ──────────────────────────────
        if rtype == "forall":
            kinds = [k.strip() for k in rule["kind"].split(",")]
            prop  = rule["property"]
            # Find subjects satisfying ALL kinds
            sets = [get_subjects_with_attr(facts, k) |
                    get_subjects_with_attr(facts, k + "s") |
                    get_subjects_with_attr(facts, k.rstrip("s"))
                    for k in kinds]
            eligible = sets[0] if len(sets) == 1 else set.intersection(*sets)
            for subj in eligible:
                new = f"NOT({subj} is {prop})" if neg else f"{subj} is {prop}"
                add(rule, new)

        # ── generic_if_attr_to_rel: "If something is X then it verb Y" ─
        elif rtype == "generic_if_attr_to_rel":
            cond      = rule["condition"]
            cons_verb = rule["cons_verb"]
            cons_obj  = rule["cons_obj"]
            parts     = [c.strip() for c in re.split(r"\band\b", cond)]
            first_pos = next((p for p in parts if not p.startswith("not ")), None)
            if not first_pos:
                continue
            candidates = get_subjects_with_attr(facts, first_pos)
            for subj in candidates:
                ok = all(
                    (f"{subj} is {p[4:].strip()}" not in facts if p.startswith("not ")
                     else f"{subj} is {p}" in facts)
                    for p in parts
                )
                if ok:
                    new = f"{subj} {cons_verb} {cons_obj}"
                    add(rule, new)

        # ── generic_if_attr: "If someone is X then they are Y" ─
        elif rtype == "generic_if_attr":
            parts  = [c.strip() for c in re.split(r"\band\b", rule["condition"])]
            conseq = rule["consequent"]
            # Collect candidate subjects from the first positive condition
            first_pos = next((p for p in parts if not p.startswith("not ")), None)
            if not first_pos:
                continue
            candidates = get_subjects_with_attr(facts, first_pos)
            for subj in candidates:
                ok = all(
                    (f"{subj} is {p[4:].strip()}" not in facts if p.startswith("not ")
                     else f"{subj} is {p}" in facts)
                    for p in parts
                )
                if ok:
                    new = f"NOT({subj} is {conseq})" if neg else f"{subj} is {conseq}"
                    add(rule, new)

        # ── generic_if_rel: "If something verb X then Y verb Z" ─
        elif rtype == "generic_if_rel":
            cond_verb = rule["cond_verb"]
            cond_obj  = rule["cond_obj"]
            cons_subj = rule["cons_subj"]
            cons_verb = rule["cons_verb"]
            cons_obj  = rule["cons_obj"]
            # Find all subjects that satisfy "subj verb cond_obj"
            actors = get_subjects_with_rel(facts, cond_verb, cond_obj)
            for actor in actors:
                # "it" or "they" refers back to the actor
                resolved_subj = actor if cons_subj in ("it", "they") else cons_subj
                # "it" in cons_obj also refers to actor
                resolved_obj = actor if cons_obj in ("it", "they") else cons_obj
                new = f"{resolved_subj} {cons_verb} {resolved_obj}"
                add(rule, new)

        # ── generic_if_rel_to_attr: "If something verb X then it is Y" ─
        elif rtype == "generic_if_rel_to_attr":
            cond_verb = rule["cond_verb"]
            cond_obj  = rule["cond_obj"]
            conseq    = rule["consequent"]
            actors    = get_subjects_with_rel(facts, cond_verb, cond_obj)
            for actor in actors:
                new = f"NOT({actor} is {conseq})" if neg else f"{actor} is {conseq}"
                add(rule, new)

        # ── specific_if_attr_to_rel: "If the squirrel is round then the squirrel visits the cow"
        elif rtype == "specific_if_attr_to_rel":
            subj      = rule["subj"]
            prop      = rule["prop"]
            cons_subj = rule["cons_subj"]
            cons_verb = rule["cons_verb"]
            cons_obj  = rule["cons_obj"]
            if f"{subj} is {prop}" in facts:
                new = f"{cons_subj} {cons_verb} {cons_obj}"
                add(rule, new)

        # ── specific_if: "If Gary is X then Gary is Y" ──────────
        elif rtype == "specific_if":
            s1, p1 = rule["subj1"], rule["prop1"]
            s2, p2 = rule.get("subj2"), rule.get("prop2")
            cs, cq = rule["cons_subj"], rule["consequent"]
            ok = f"{s1} is {p1}" in facts and (not s2 or f"{s2} is {p2}" in facts)
            if ok:
                new = f"NOT({cs} is {cq})" if neg else f"{cs} is {cq}"
                add(rule, new)

        # ── specific_if_rel: "If X verb Y [and A verb B] then C is/verb D" ─
        elif rtype == "specific_if_rel":
            s1   = rule["subj1"]; v1 = rule["verb1"]; o1 = rule["obj1"]
            s2   = rule.get("subj2"); v2 = rule.get("verb2"); o2 = rule.get("obj2")
            cs   = rule["cons_subj"]

            # Check first condition — may use "something" (generic) or specific subject
            if s1 in ("something", "someone", "it"):
                # Find all actors satisfying verb1 obj1
                actors1 = get_subjects_with_rel(facts, v1, o1)
            else:
                actors1 = {s1} if f"{s1} {v1} {o1}" in facts or \
                          any(f"{s1} {v} {o1}" in facts
                              for v in [v1, v1.rstrip("s"), v1+"s"]) else set()

            for actor in actors1:
                # Check second condition if present
                if s2 and v2 and o2:
                    s2r = actor if s2 in ("it", "they", "something") else s2
                    o2r = actor if o2 in ("it", "they") else o2
                    cond2 = any(f"{s2r} {v} {o2r}" in facts
                                for v in [v2, v2.rstrip("s"), v2+"s"])
                    if not cond2:
                        continue

                # Resolve consequent subject
                cs_r = actor if cs in ("it", "they") else cs

                if rule.get("cons_attr"):
                    neg = rule.get("cons_neg", False)
                    attr = rule["cons_attr"]
                    new = f"NOT({cs_r} is {attr})" if neg else f"{cs_r} is {attr}"
                elif rule.get("cons_verb") and rule.get("cons_obj"):
                    cv = rule["cons_verb"]
                    co = rule["cons_obj"]
                    co_r = actor if co in ("it", "they") else co
                    new = f"{cs_r} {cv} {co_r}"
                else:
                    continue
                add(rule, new)

    return applicable


def symbolic_check_answer(question, facts):
    """
    The symbolic engine checks whether the question is definitively
    answered by the current fact base.

    Returns: "true", "false", or None (not yet answerable)
    """
    q = question.strip().lower().rstrip("?.").strip()
    # Normalize "the bear is cold" → subject kept as-is since facts also use "the bear"

    # "X does not verb Y" / "X is not Y"
    m = re.match(r"^([\w][\w\s]*?) (?:does not|do not|is not) (.+)$", q)
    if m:
        subj = m.group(1).strip()
        rest = m.group(2).strip()
        # Try as attribute negation
        if f"NOT({subj} is {rest})" in facts: return "true"
        if f"{subj} is {rest}" in facts:       return "false"
        # Try as relational negation — also try base verb form (visit vs visits)
        rel_m = re.match(r"^([\w]+s?) (.+)$", rest)
        if rel_m:
            verb     = rel_m.group(1).strip()
            verb_s   = verb if verb.endswith("s") else verb + "s"
            verb_base= verb.rstrip("s")
            obj      = rel_m.group(2).strip()
            for v in [verb, verb_s, verb_base]:
                if f"NOT({subj} {v} {obj})" in facts: return "true"
                if f"{subj} {v} {obj}" in facts:       return "false"
        return None

    # "X verb Y" (relational question)
    m = re.match(r"^([\w][\w\s]*?) (" + RELATIONAL_VERBS + r") ([\w][\w\s]*)$", q)
    if m:
        subj = m.group(1).strip()
        verb = m.group(2).strip()
        obj  = m.group(3).strip()
        if f"{subj} {verb} {obj}" in facts:        return "true"
        if f"NOT({subj} {verb} {obj})" in facts:   return "false"
        return None

    # "X is Y" (attribute question)
    m = re.match(r"^([\w][\w\s]*?) is ([\w][\w\s]*)$", q)
    if m:
        subj = m.group(1).strip()
        prop = m.group(2).strip()
        if f"{subj} is {prop}" in facts:        return "true"
        if f"NOT({subj} is {prop})" in facts:   return "false"
        return None

    return None

# ─── Ollama ───────────────────────────────────────────────────────────────────

def detect_model(override=None):
    if override:
        return override
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        for pref in ["llama3.1:latest", "llama3:latest", "llama3.1", "llama3", "llama3:8b"]:
            if pref in models:
                return pref
        llama = [m for m in models if "llama3" in m.lower()]
        if llama: return llama[0]
    except Exception:
        pass
    return MODEL


def call_ollama(prompt, model, max_tokens=None):
    t0 = time.time()
    r  = requests.post(OLLAMA_URL, json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {
            "num_predict": max_tokens or MAX_TOKENS,
            "temperature": TEMPERATURE
        }
    }, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip(), round(time.time() - t0, 2)


def extract_verdict(response):
    low = response.strip().lower()
    for word in re.split(r"[\s,.:;!()\[\]]+", low):
        if word in ("true", "false", "unknown"):
            return word
    if "unknown" in low: return "unknown"
    if "true"    in low: return "true"
    if "false"   in low: return "false"
    return "unclear"

# ─── Pipeline A: LLM-Only ─────────────────────────────────────────────────────

def run_llm_only(theory, question, model):
    prompt = (
        "You are a logical reasoning assistant.\n"
        "Read the theory carefully and answer the question.\n"
        "Respond with ONLY one of: True, False, or Unknown — "
        "then a single sentence explaining why.\n\n"
        f"Theory:\n{theory}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    resp, t = call_ollama(prompt, model)
    return {
        "verdict":      extract_verdict(resp),
        "time_s":       t,
        "raw_response": resp,
    }

# ─── Pipeline B: Neuro-Symbolic Loop ─────────────────────────────────────────

def llm_select_rule(question, facts, applicable, model):
    """
    Ask the LLM: given the current facts and the question we're trying to answer,
    which of these applicable rules should we apply next?
    The LLM returns a number (1-N). The symbolic engine applies that rule.
    """
    facts_str = "\n".join(f"  - {f}" for f in sorted(facts))
    rules_str = "\n".join(
        f"  [{i+1}] Apply \"{a['rule']['raw']}\" → would derive: \"{a['would_derive']}\""
        for i, a in enumerate(applicable)
    )
    prompt = (
        "You are guiding a symbolic reasoning engine.\n"
        "Your job is to select which rule to apply next to make progress "
        "toward answering the question.\n\n"
        f"QUESTION: {question}\n\n"
        f"CURRENT KNOWN FACTS:\n{facts_str}\n\n"
        f"APPLICABLE RULES (pick one):\n{rules_str}\n\n"
        "Reply with ONLY the number of the rule to apply next (e.g. '2').\n"
        "Pick the rule whose derived fact is most directly relevant to the question.\n\n"
        "Rule number:"
    )
    resp, t = call_ollama(prompt, model, max_tokens=10)
    # Extract the number from response
    m = re.search(r"\d+", resp)
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < len(applicable):
            return idx, t
    return 0, t  # fallback to first rule


def run_neuro_symbolic(theory, question, model):
    """
    The genuine neuro-symbolic loop.

    At each step:
      - Symbolic engine finds applicable rules
      - LLM selects which rule to apply (guided search)
      - Symbolic engine applies it (adds new fact)
      - Symbolic engine checks if question is now answerable
      - Repeat until answered or no rules left

    The SYMBOLIC ENGINE produces the final answer.
    The LLM only guides which path to take.
    """
    facts, rules = parse_knowledge_base(theory)
    working_memory = set(facts)  # grows as we derive new facts

    loop_trace  = []  # full record of every step
    total_time  = 0.0
    llm_calls   = 0

    for step in range(MAX_NS_STEPS):

        # ── 1. Symbolic engine checks if question is already answered ──
        verdict = symbolic_check_answer(question, working_memory)
        if verdict is not None:
            loop_trace.append({
                "step":   step,
                "action": "SYMBOLIC_CHECK",
                "result": f"Question answered: {verdict}",
            })
            break

        # ── 2. Symbolic engine finds all applicable rules ──────────────
        applicable = get_applicable_rules(working_memory, rules)

        if not applicable:
            # No more rules can fire — symbolic engine declares Unknown
            loop_trace.append({
                "step":   step,
                "action": "SYMBOLIC_EXHAUSTED",
                "result": "No more rules applicable — declaring Unknown",
            })
            verdict = "unknown"
            break

        # ── 3. LLM selects which rule to apply next ────────────────────
        if len(applicable) == 1:
            # Only one option — no need to ask LLM
            chosen_idx = 0
            llm_time   = 0.0
        else:
            chosen_idx, llm_time = llm_select_rule(
                question, working_memory, applicable, model
            )
            total_time += llm_time
            llm_calls  += 1

        chosen    = applicable[chosen_idx]
        new_fact  = chosen["would_derive"]
        rule_text = chosen["rule"]["raw"]

        # ── 4. Symbolic engine applies the rule ───────────────────────
        working_memory.add(new_fact)

        loop_trace.append({
            "step":         step + 1,
            "action":       "APPLY_RULE",
            "rule":         rule_text,
            "derived":      new_fact,
            "llm_selected": chosen_idx + 1,
            "n_candidates": len(applicable),
            "llm_time_s":   llm_time,
        })

    else:
        # Hit max steps without resolving
        verdict = symbolic_check_answer(question, working_memory)
        if verdict is None:
            verdict = "unknown"

    # Final check in case loop ended without explicit verdict
    if verdict is None:
        verdict = symbolic_check_answer(question, working_memory) or "unknown"

    return {
        "verdict":         verdict,
        "total_time_s":    round(total_time, 2),
        "llm_calls":       llm_calls,
        "steps":           len(loop_trace),
        "loop_trace":      loop_trace,
        "final_facts":     sorted(working_memory),
        "initial_facts":   sorted(facts),
        "rules_parsed":    [r["raw"] for r in rules],
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def make_stats():
    return {"llm": 0, "ns": 0, "total": 0}


def run():
    args = parse_args()

    print("\n" + "="*68)
    print("  Neuro-Symbolic Reasoning — Proper 2-Way Comparison")
    print("  Pipeline A: LLM-Only (pure neural)")
    print("  Pipeline B: Neuro-Symbolic Loop (LLM guides, Symbolic decides)")
    print("="*68)

    try:
        requests.get("http://localhost:11434", timeout=3)
    except Exception:
        print("\nERROR: Ollama not running. Start with:  ollama serve\n")
        return

    model    = detect_model(args.model)
    problems = load_proofwriter_owa(args.split, args.n)

    print(f"  Model   : {model}")
    print(f"  Split   : {args.split}")
    print(f"  N/depth : {args.n}  ({len(problems)} total)\n")

    results      = []
    llm_correct  = ns_correct = 0
    config_stats = defaultdict(make_stats)
    qdep_stats   = defaultdict(make_stats)
    total        = len(problems)

    # For logical consistency: group by problem id (shared theory)
    theory_answers = defaultdict(lambda: {'llm': {}, 'ns': {}})

    # Reasoning error categories for NS pipeline
    reasoning_errors = {
        'parser_missed_rule':      0,  # symbolic exhausted with 0 steps fired
        'max_steps_reached':       0,  # hit MAX_NS_STEPS without answer
        'wrong_verdict':           0,  # returned true/false but incorrect
        'unknown_when_answerable': 0,  # returned unknown, answer was true/false
        'llm_misdirection':        0,  # had rules, chained but got wrong answer
    }

    for i, prob in enumerate(problems, 1):
        theory   = prob["theory"]
        question = prob["question"]
        expected = prob["answer"]
        cfg      = prob["config"]
        qdep     = prob["depth"]

        print(f"[{i:>4}/{total}] {cfg}  QDep={qdep}  {prob['id']}")
        print(f"          Q: {question}")

        # ── Pipeline A: LLM-Only ────────────────────────────────
        llm_result = run_llm_only(theory, question, model)
        llm_ok     = llm_result["verdict"] == expected
        if llm_ok: llm_correct += 1

        # ── Pipeline B: Neuro-Symbolic Loop ─────────────────────
        ns_result = run_neuro_symbolic(theory, question, model)
        ns_ok     = ns_result["verdict"] == expected
        if ns_ok: ns_correct += 1

        for stats in [config_stats[cfg], qdep_stats[qdep]]:
            stats["total"] += 1
            if llm_ok: stats["llm"] += 1
            if ns_ok:  stats["ns"]  += 1

        print(f"          LLM-Only  [{'✓' if llm_ok else '✗'}] → {llm_result['verdict']!r:9s}  ({llm_result['time_s']}s)")
        print(f"          Neuro-Sym [{'✓' if ns_ok  else '✗'}] → {ns_result['verdict']!r:9s}  "
              f"({ns_result['steps']} steps, {ns_result['llm_calls']} LLM calls)")
        print(f"          Expected: {expected!r}\n")

        results.append({
            "id":              prob["id"],
            "config":          cfg,
            "qdep":            qdep,
            "nfact":           prob["nfact"],
            "nrule":           prob["nrule"],
            "theory":          theory,
            "question":        question,
            "expected_answer": expected,

            "llm_only": {
                "verdict":      llm_result["verdict"],
                "correct":      llm_ok,
                "time_s":       llm_result["time_s"],
                "raw_response": llm_result["raw_response"],
            },

            "neuro_symbolic": {
                "verdict":       ns_result["verdict"],
                "correct":       ns_ok,
                "total_time_s":  ns_result["total_time_s"],
                "llm_calls":     ns_result["llm_calls"],
                "steps":         ns_result["steps"],
                "loop_trace":    ns_result["loop_trace"],
                "final_facts":   ns_result["final_facts"],
                "initial_facts": ns_result["initial_facts"],
                "rules_parsed":  ns_result["rules_parsed"],
            }
        })

        # ── Track for logical consistency ────────────────────────
        theory_id = prob["id"].rsplit("-", 1)[0]  # group questions by shared theory
        theory_answers[theory_id]["llm"][question] = llm_result["verdict"]
        theory_answers[theory_id]["ns"][question]  = ns_result["verdict"]
        theory_answers[theory_id].setdefault("expected", {})[question] = expected

        # ── Classify NS reasoning errors ─────────────────────────
        if not ns_ok:
            ns_v   = ns_result["verdict"]
            trace  = ns_result["loop_trace"]
            n_steps_fired = sum(1 for t in trace if t.get("action") == "APPLY_RULE")
            hit_max = ns_result["steps"] >= MAX_NS_STEPS

            if hit_max:
                reasoning_errors["max_steps_reached"] += 1
            elif ns_v == "unknown" and n_steps_fired == 0:
                reasoning_errors["parser_missed_rule"] += 1
            elif ns_v == "unknown" and n_steps_fired > 0:
                reasoning_errors["llm_misdirection"] += 1
            elif ns_v != "unknown" and ns_v != expected:
                reasoning_errors["wrong_verdict"] += 1
            elif ns_v == "unknown" and expected in ("true", "false"):
                reasoning_errors["unknown_when_answerable"] += 1

    # ── Summary ──────────────────────────────────────────────────
    n      = len(problems)
    llm_acc = round(llm_correct / n * 100, 1)
    ns_acc  = round(ns_correct  / n * 100, 1)
    delta   = round(ns_acc - llm_acc, 1)

    def breakdown(stats_dict, keys):
        out = {}
        for k in keys:
            s = stats_dict[k]
            t = s["total"]
            out[str(k)] = {
                "total":        t,
                "llm_correct":  s["llm"],
                "ns_correct":   s["ns"],
                "llm_accuracy": round(s["llm"] / t * 100, 1) if t else 0,
                "ns_accuracy":  round(s["ns"]  / t * 100, 1) if t else 0,
                "delta":        round((s["ns"] - s["llm"]) / t * 100, 1) if t else 0,
            }
        return out

    # ── Logical Consistency ──────────────────────────────────────
    # A pipeline is inconsistent if it gives contradictory answers
    # to a positive/negative question pair on the same theory.
    # e.g. "Gary is blue" → true AND "Gary is not blue" → true = contradiction
    def compute_consistency(pipeline_key):
        total_pairs = consistent_pairs = 0
        contradictions = []
        for tid, data in theory_answers.items():
            answers  = data[pipeline_key]
            expected = data["expected"]
            questions = list(answers.keys())
            # Find pairs where one is the negation of the other
            for j, q1 in enumerate(questions):
                for q2 in questions[j+1:]:
                    q1_clean = q1.strip().lower().rstrip("?.").strip()
                    q2_clean = q2.strip().lower().rstrip("?.").strip()
                    # Check if one is "X is not Y" and other is "X is Y" (or relational equiv)
                    is_pair = False
                    if "not" in q2_clean and q2_clean.replace(" not ", " ") == q1_clean:
                        is_pair = True
                    elif "not" in q1_clean and q1_clean.replace(" not ", " ") == q2_clean:
                        is_pair = True
                    elif "does not" in q2_clean:
                        q2_pos = q2_clean.replace("does not ", "")
                        if q2_pos == q1_clean:
                            is_pair = True
                    elif "does not" in q1_clean:
                        q1_pos = q1_clean.replace("does not ", "")
                        if q1_pos == q2_clean:
                            is_pair = True
                    if is_pair:
                        total_pairs += 1
                        v1, v2 = answers[q1], answers[q2]
                        # Contradiction: both "true", or both "false", or
                        # one says "true" when expected pair should be opposite
                        if v1 == v2 and v1 != "unknown":
                            contradictions.append({
                                "theory_id": tid,
                                "q1": q1, "v1": v1,
                                "q2": q2, "v2": v2,
                            })
                        else:
                            consistent_pairs += 1
        consistency_pct = round(consistent_pairs / total_pairs * 100, 1) if total_pairs else 100.0
        return {
            "total_pairs":       total_pairs,
            "consistent_pairs":  consistent_pairs,
            "contradictions":    len(contradictions),
            "consistency_pct":   consistency_pct,
            "contradiction_examples": contradictions[:3],  # save up to 3 examples
        }

    llm_consistency = compute_consistency("llm")
    ns_consistency  = compute_consistency("ns")

    summary = {
        "timestamp":          datetime.now().isoformat(),
        "model":              model,
        "dataset":            "tasksource/proofwriter",
        "assumption":         "OWA (open world)",
        "split":              args.split,
        "n_per_depth":        args.n,
        "n_total":            n,
        "llm_only":           {"correct": llm_correct, "accuracy_pct": llm_acc},
        "neuro_symbolic":     {"correct": ns_correct,  "accuracy_pct": ns_acc},
        "delta_ns_vs_llm":    delta,
        "by_config":          breakdown(config_stats, OWA_CONFIGS),
        "by_qdep":            breakdown(qdep_stats, sorted(qdep_stats)),
        "logical_consistency": {
            "llm_only":       llm_consistency,
            "neuro_symbolic": ns_consistency,
        },
        "reasoning_errors": reasoning_errors,
    }


    # ── Print ─────────────────────────────────────────────────────
    print("="*68)
    print("  FINAL RESULTS")
    print("="*68)
    print(f"  Pipeline A  LLM-Only      : {llm_correct:>3}/{n}  =  {llm_acc}%")
    print(f"  Pipeline B  Neuro-Symbolic: {ns_correct:>3}/{n}  =  {ns_acc}%")
    sign = "+" if delta >= 0 else ""
    print(f"  Δ Neuro-Sym vs LLM        : {sign}{delta}%")

    print(f"\n  {'Config':<12}  {'LLM':>7}  {'Neuro-Sym':>10}  {'Delta':>7}  {'N':>5}")
    print(f"  {'-'*52}")
    for cfg in OWA_CONFIGS:
        v = summary["by_config"].get(cfg, {})
        if not v or v["total"] == 0:
            print(f"  {cfg:<12}  {'—':>7}  {'—':>10}  {'—':>7}  {'0':>5}")
        else:
            sign = "+" if v["delta"] >= 0 else ""
            print(f"  {cfg:<12}  {v['llm_accuracy']:>6.1f}%  {v['ns_accuracy']:>9.1f}%  "
                  f"{sign}{v['delta']:>5.1f}%  {v['total']:>5}")

    print(f"\n  {'QDep':<8}  {'LLM':>7}  {'Neuro-Sym':>10}  {'Delta':>7}  {'N':>5}")
    print(f"  {'-'*48}")
    for d, v in summary["by_qdep"].items():
        sign = "+" if v["delta"] >= 0 else ""
        print(f"  {d:<8}  {v['llm_accuracy']:>6.1f}%  {v['ns_accuracy']:>9.1f}%  "
              f"{sign}{v['delta']:>5.1f}%  {v['total']:>5}")
    print("="*68)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Detailed results  →  {RESULTS_FILE}")
    print(f"  Summary           →  {SUMMARY_FILE}\n")


if __name__ == "__main__":
    run()