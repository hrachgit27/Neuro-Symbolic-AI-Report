import re
import json
import time
import argparse
import requests
from datetime import datetime
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "llama3"
MAX_TOKENS  = 300
TEMPERATURE = 0

# All OWA depth configs in ProofWriter
OWA_CONFIGS = ["depth-0", "depth-1", "depth-2", "depth-3", "depth-4", "depth-5"]

RESULTS_FILE = "results.json"
SUMMARY_FILE = "summary.json"

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Neuro-Symbolic Comparison — ProofWriter OWA")
    p.add_argument("--n",     type=int, default=200, help="Problems per depth level (default: 200)")
    p.add_argument("--split", default="test",        help="Dataset split: train/validation/test")
    p.add_argument("--model", default=None,          help="Override Ollama model name")
    return p.parse_args()

# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_proofwriter_owa(split, n_per_depth):
    """
    Load n_per_depth problems from each OWA depth config.
    Returns a flat list of problems with their config label attached.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Run: pip install datasets")

    print(f"  Loading ProofWriter OWA (split={split}, {n_per_depth} per depth × 6 levels)...")

    # Stream the full split once and bucket by config
    ds = load_dataset("tasksource/proofwriter", split=split, streaming=True)
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
            "answer":   row["answer"].strip().lower(),   # true / false / unknown
            "depth":    row["QDep"],
            "config":   cfg,
            "nfact":    row.get("NFact", 0),
            "nrule":    row.get("NRule", 0),
        })
        # Stop early once all buckets are full
        if all(len(buckets[c]) >= n_per_depth for c in OWA_CONFIGS):
            break

    # Flatten in depth order
    problems = []
    for cfg in OWA_CONFIGS:
        problems.extend(buckets[cfg])
        print(f"    {cfg}: {len(buckets[cfg])} problems loaded")

    total = len(problems)
    print(f"\n  Total: {total} problems across {len(OWA_CONFIGS)} depth levels.\n")
    return problems

# ─── Symbolic Reasoner ────────────────────────────────────────────────────────

def parse_knowledge_base(theory):
    facts = set()
    rules = []
    sentences = [s.strip() for s in re.split(r"\.\s*", theory) if s.strip()]

    for sent in sentences:
        low = sent.lower()

        # "If someone/something is X [and not Y] then they/it are/is [not] Z"
        m = re.match(
            r"^if (?:someone|something) is (.+?) then (?:they|it) (?:are|is) (not )?(.+)$", low
        )
        if m:
            rules.append({
                "type":      "generic_if",
                "condition": m.group(1).strip(),
                "negated":   bool(m.group(2)),
                "consequent":m.group(3).strip(),
                "raw":       sent
            })
            continue

        # "If <subj> is X [and <subj> is Y] then <subj> is [not] Z"
        m = re.match(
            r"^if (\w[\w\s]*?) is (\w[\w\s]*?)(?: and (\w[\w\s]*?) is (\w[\w\s]*?))?"
            r" then (\w[\w\s]*?) is (not )?(.+)$", low
        )
        if m:
            rules.append({
                "type":      "specific_if",
                "subj1":     m.group(1).strip(), "prop1": m.group(2).strip(),
                "subj2":     m.group(3).strip() if m.group(3) else None,
                "prop2":     m.group(4).strip() if m.group(4) else None,
                "cons_subj": m.group(5).strip(),
                "negated":   bool(m.group(6)),
                "consequent":m.group(7).strip(),
                "raw":       sent
            })
            continue

        # "All [X] things/people are [not] Y" or "All Xs are [not] Y"
        m = re.match(
            r"^all (\w[\w\s,]*?)(?:\s+things?|\s+people|\s+animals?)? are (not )?(.+)$", low
        )
        if m:
            rules.append({
                "type":     "forall",
                "kind":     m.group(1).strip(),
                "negated":  bool(m.group(2)),
                "property": m.group(3).strip(),
                "raw":      sent
            })
            continue

        # "X things are [not] Y" (shorthand forall)
        m = re.match(r"^(\w[\w\s,]*?) things? are (not )?(.+)$", low)
        if m:
            rules.append({
                "type":     "forall",
                "kind":     m.group(1).strip(),
                "negated":  bool(m.group(2)),
                "property": m.group(3).strip(),
                "raw":      sent
            })
            continue

        # Fact: "X is not Y"
        m = re.match(r"^(\w[\w\s]*?) is not (.+)$", low)
        if m:
            facts.add(f"NOT({m.group(1).strip()} is {m.group(2).strip()})")
            continue

        # Fact: "X is Y"
        m = re.match(r"^(\w[\w\s]*?) is (\w[\w\s]*)$", low)
        if m:
            facts.add(f"{m.group(1).strip()} is {m.group(2).strip()}")
            continue

        # Relational fact: "X chases/eats/likes Y"
        m = re.match(r"^(\w[\w\s]*?) (chases?|eats?|likes?|sees?|visits?) (\w[\w\s]*)$", low)
        if m:
            facts.add(f"{m.group(1).strip()} {m.group(2)} {m.group(3).strip()}")

    return facts, rules


def forward_chain(facts, rules, max_iter=20):
    derived = set(facts)
    trace   = []

    for _ in range(max_iter):
        changed = False

        for rule in rules:
            rtype = rule["type"]
            neg   = rule.get("negated", False)

            if rtype == "forall":
                kind  = rule["kind"]
                prop  = rule["property"]
                kinds = [k.strip() for k in kind.split(",")]
                # Map each kind to the set of subjects that satisfy it
                subjects_by_kind = defaultdict(set)
                for fact in list(derived):
                    m = re.match(r"^(\w[\w\s]*?) is (.+)$", fact)
                    if m:
                        subj = m.group(1).strip()
                        attr = m.group(2).strip()
                        for k in kinds:
                            if attr == k or attr.rstrip("s") == k.rstrip("s"):
                                subjects_by_kind[k].add(subj)
                eligible = (
                    subjects_by_kind[kinds[0]] if len(kinds) == 1
                    else set.intersection(*[subjects_by_kind[k] for k in kinds])
                    if all(subjects_by_kind[k] for k in kinds) else set()
                )
                for subj in eligible:
                    new = f"NOT({subj} is {prop})" if neg else f"{subj} is {prop}"
                    if new not in derived:
                        derived.add(new); trace.append({"rule": rule["raw"], "derived": new})
                        changed = True

            elif rtype == "generic_if":
                cond_parts = [c.strip() for c in re.split(r"\band\b", rule["condition"])]
                conseq     = rule["consequent"]
                for fact in list(derived):
                    m = re.match(r"^(\w[\w\s]*?) is (.+)$", fact)
                    if not m:
                        continue
                    subj = m.group(1).strip()
                    ok   = True
                    for part in cond_parts:
                        if part.startswith("not "):
                            if f"{subj} is {part[4:].strip()}" in derived:
                                ok = False; break
                        else:
                            if f"{subj} is {part}" not in derived:
                                ok = False; break
                    if ok:
                        new = f"NOT({subj} is {conseq})" if neg else f"{subj} is {conseq}"
                        if new not in derived:
                            derived.add(new); trace.append({"rule": rule["raw"], "derived": new})
                            changed = True

            elif rtype == "specific_if":
                s1, p1 = rule["subj1"], rule["prop1"]
                s2, p2 = rule.get("subj2"), rule.get("prop2")
                cs, cq = rule["cons_subj"], rule["consequent"]
                ok = f"{s1} is {p1}" in derived
                if ok and s2 and p2:
                    ok = f"{s2} is {p2}" in derived
                if ok:
                    new = f"NOT({cs} is {cq})" if neg else f"{cs} is {cq}"
                    if new not in derived:
                        derived.add(new); trace.append({"rule": rule["raw"], "derived": new})
                        changed = True

        if not changed:
            break

    return derived, derived - facts, trace

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
        if llama:
            return llama[0]
    except Exception:
        pass
    return MODEL


def call_ollama(prompt, model):
    t0 = time.time()
    r  = requests.post(OLLAMA_URL, json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": MAX_TOKENS, "temperature": TEMPERATURE}
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

# ─── Prompts ──────────────────────────────────────────────────────────────────

def llm_only_prompt(theory, question):
    return (
        "You are a logical reasoning assistant.\n"
        "Read the theory carefully and answer the question.\n"
        "Respond with ONLY one of: True, False, or Unknown — "
        "then a single sentence explaining why.\n\n"
        f"Theory:\n{theory}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def hybrid_prompt(theory, question, derived, trace):
    facts_str = "\n".join(f"  - {f}" for f in sorted(derived))
    trace_str = (
        "\n".join(
            f"  [{t['step']}] Rule: \"{t['rule']}\"  =>  derived: \"{t['derived']}\""
            for t in trace
        ) if trace else "  (no new facts derived)"
    )
    return (
        "You are a logical reasoning assistant.\n"
        "A forward-chaining symbolic reasoner has already processed the theory.\n"
        "Use the derived facts and trace below to answer the question.\n"
        "Respond with ONLY one of: True, False, or Unknown — "
        "then a single sentence explaining why.\n\n"
        f"DERIVED FACTS:\n{facts_str}\n\n"
        f"REASONING TRACE:\n{trace_str}\n\n"
        f"Original Theory:\n{theory}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    args = parse_args()

    print("\n" + "="*65)
    print("  Neuro-Symbolic Reasoning — ProofWriter OWA (depth-0 → depth-5)")
    print("="*65)

    try:
        requests.get("http://localhost:11434", timeout=3)
    except Exception:
        print("\nERROR: Ollama not running.  Start it with:  ollama serve\n")
        return

    model = detect_model(args.model)
    print(f"\n  Model   : {model}")
    print(f"  Split   : {args.split}")
    print(f"  N/depth : {args.n}  (up to {args.n * len(OWA_CONFIGS)} total)\n")

    problems = load_proofwriter_owa(args.split, args.n)

    results       = []
    llm_correct   = 0
    hyb_correct   = 0
    # Stats keyed by (config, QDep)
    config_stats  = defaultdict(lambda: {"llm": 0, "hyb": 0, "total": 0})
    qdep_stats    = defaultdict(lambda: {"llm": 0, "hyb": 0, "total": 0})

    total = len(problems)

    for i, prob in enumerate(problems, 1):
        theory   = prob["theory"]
        question = prob["question"]
        expected = prob["answer"]
        cfg      = prob["config"]
        qdep     = prob["depth"]

        print(f"[{i:>4}/{total}] {cfg}  QDep={qdep}  {prob['id']}")
        print(f"          Q: {question}")

        # Symbolic pass
        facts, rules              = parse_knowledge_base(theory)
        derived, new_facts, trace = forward_chain(facts, rules)
        numbered_trace = [
            {"step": j+1, "rule": t["rule"], "derived": t["derived"]}
            for j, t in enumerate(trace)
        ]

        # LLM-Only
        llm_resp, llm_time = call_ollama(llm_only_prompt(theory, question), model)
        llm_verdict        = extract_verdict(llm_resp)
        llm_ok             = llm_verdict == expected
        if llm_ok: llm_correct += 1

        # LLM + Symbolic
        hyb_resp, hyb_time = call_ollama(
            hybrid_prompt(theory, question, derived, numbered_trace), model
        )
        hyb_verdict = extract_verdict(hyb_resp)
        hyb_ok      = hyb_verdict == expected
        if hyb_ok: hyb_correct += 1

        config_stats[cfg]["total"] += 1
        qdep_stats[qdep]["total"]  += 1
        if llm_ok: config_stats[cfg]["llm"] += 1; qdep_stats[qdep]["llm"] += 1
        if hyb_ok: config_stats[cfg]["hyb"] += 1; qdep_stats[qdep]["hyb"] += 1

        print(f"          LLM-Only  [{'✓' if llm_ok else '✗'}] → {llm_verdict!r:9s}  ({llm_time}s)")
        print(f"          Hybrid    [{'✓' if hyb_ok else '✗'}] → {hyb_verdict!r:9s}  ({hyb_time}s)")
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
                "verdict":      llm_verdict,
                "correct":      llm_ok,
                "time_s":       llm_time,
                "raw_response": llm_resp,
            },

            "llm_symbolic": {
                "verdict":      hyb_verdict,
                "correct":      hyb_ok,
                "time_s":       hyb_time,
                "raw_response": hyb_resp,
                "symbolic_reasoning": {
                    "facts_parsed":    sorted(facts),
                    "rules_parsed":    [r["raw"] for r in rules],
                    "facts_derived":   sorted(new_facts),
                    "reasoning_trace": numbered_trace,
                }
            }
        })

    # ── Summary ──────────────────────────────────────────────────
    n       = len(problems)
    llm_acc = round(llm_correct / n * 100, 1)
    hyb_acc = round(hyb_correct / n * 100, 1)
    delta   = round(hyb_acc - llm_acc, 1)

    by_config = {}
    for cfg in OWA_CONFIGS:
        s = config_stats[cfg]
        t = s["total"]
        by_config[cfg] = {
            "total":        t,
            "llm_correct":  s["llm"],
            "hyb_correct":  s["hyb"],
            "llm_accuracy": round(s["llm"] / t * 100, 1) if t else 0,
            "hyb_accuracy": round(s["hyb"] / t * 100, 1) if t else 0,
            "delta":        round((s["hyb"] - s["llm"]) / t * 100, 1) if t else 0,
        }

    by_qdep = {}
    for d in sorted(qdep_stats):
        s = qdep_stats[d]
        t = s["total"]
        by_qdep[f"qdep_{d}"] = {
            "total":        t,
            "llm_correct":  s["llm"],
            "hyb_correct":  s["hyb"],
            "llm_accuracy": round(s["llm"] / t * 100, 1) if t else 0,
            "hyb_accuracy": round(s["hyb"] / t * 100, 1) if t else 0,
            "delta":        round((s["hyb"] - s["llm"]) / t * 100, 1) if t else 0,
        }

    summary = {
        "timestamp":          datetime.now().isoformat(),
        "model":              model,
        "dataset":            "tasksource/proofwriter",
        "assumption":         "OWA (open world)",
        "split":              args.split,
        "configs":            OWA_CONFIGS,
        "n_per_depth":        args.n,
        "n_total":            n,
        "llm_only":           {"correct": llm_correct, "accuracy_pct": llm_acc},
        "llm_symbolic":       {"correct": hyb_correct, "accuracy_pct": hyb_acc},
        "delta_accuracy_pct": delta,
        "by_config":          by_config,
        "by_qdep":            by_qdep,
    }

    # ── Print ─────────────────────────────────────────────────────
    print("="*65)
    print("  FINAL RESULTS — OWA depth-0 through depth-5")
    print("="*65)
    print(f"  LLM-Only  accuracy : {llm_correct}/{n}  =  {llm_acc}%")
    print(f"  Hybrid    accuracy : {hyb_correct}/{n}  =  {hyb_acc}%")
    sign = "+" if delta >= 0 else ""
    print(f"  Delta (Hybrid−LLM) : {sign}{delta}%")

    print("\n  By config (depth level):")
    print(f"  {'Config':<12}  {'LLM':>7}  {'Hybrid':>7}  {'Delta':>7}  {'N':>5}")
    print(f"  {'-'*46}")
    for cfg, v in by_config.items():
        dsign = "+" if v["delta"] >= 0 else ""
        print(f"  {cfg:<12}  {v['llm_accuracy']:>6.1f}%  {v['hyb_accuracy']:>6.1f}%  {dsign}{v['delta']:>5.1f}%  {v['total']:>5}")

    print("\n  By QDep (reasoning hops):")
    print(f"  {'QDep':<8}  {'LLM':>7}  {'Hybrid':>7}  {'Delta':>7}  {'N':>5}")
    print(f"  {'-'*42}")
    for key, v in by_qdep.items():
        dsign = "+" if v["delta"] >= 0 else ""
        print(f"  {key:<8}  {v['llm_accuracy']:>6.1f}%  {v['hyb_accuracy']:>6.1f}%  {dsign}{v['delta']:>5.1f}%  {v['total']:>5}")
    print("="*65)

    # ── Save ─────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Detailed results  →  {RESULTS_FILE}")
    print(f"  Summary           →  {SUMMARY_FILE}\n")


if __name__ == "__main__":
    run()