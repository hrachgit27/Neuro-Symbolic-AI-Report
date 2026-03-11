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

OWA_CONFIGS  = ["depth-0", "depth-1", "depth-2", "depth-3", "depth-4", "depth-5"]
RESULTS_FILE = "results.json"
SUMMARY_FILE = "summary.json"

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="3-Way Neuro-Symbolic Comparison on ProofWriter OWA")
    p.add_argument("--n",     type=int, default=10,   help="Problems per depth level (default: 10)")
    p.add_argument("--split", default="test",          help="Dataset split: train/validation/test")
    p.add_argument("--model", default=None,            help="Override Ollama model name")
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
            "id":        row["id"],
            "theory":    row["theory"],
            "question":  row["question"],
            "answer":    row["answer"].strip().lower(),
            "depth":     row["QDep"],
            "config":    cfg,
            "nfact":     row.get("NFact", 0),
            "nrule":     row.get("NRule", 0),
            "allProofs": row.get("allProofs", ""),
        })
        if all(len(buckets[c]) >= n_per_depth for c in OWA_CONFIGS):
            break

    problems = []
    for cfg in OWA_CONFIGS:
        problems.extend(buckets[cfg])
        print(f"    {cfg}: {len(buckets[cfg])} problems loaded")

    print(f"\n  Total: {len(problems)} problems.\n")
    return problems

# ─── Ground-Truth Proof Parser ────────────────────────────────────────────────

def parse_allproofs(allproofs_str):
    """
    Parse the dataset's allProofs field into a structured list of steps.

    Format example:
      @0: Gary is furry.[(triple1)] Gary is white.[(triple6)]
      @1: Anne is rough.[(((triple2) -> rule1))]

    Returns:
      steps  : list of {"depth": int, "fact": str, "proof": str}
      facts  : flat set of all derived fact strings
    """
    steps = []
    facts = set()

    if not allproofs_str or allproofs_str.strip() == "null":
        return steps, facts

    # Split on @N: markers
    depth_blocks = re.split(r"@(\d+):\s*", allproofs_str.strip())
    # depth_blocks = ['', '0', 'facts...', '1', 'facts...', ...]
    i = 1
    while i < len(depth_blocks) - 1:
        depth_num = int(depth_blocks[i])
        block     = depth_blocks[i + 1].strip()
        i += 2

        # Each fact is: "Subject is property.[(proof)]"
        # Split on pattern: word boundary before capital or "The"
        entries = re.findall(r"([^[]+\.[^[]*\[[^\]]*\])", block)
        for entry in entries:
            # Separate fact text from proof bracket
            m = re.match(r"^(.+?)\.\s*\[(.+)\]$", entry.strip())
            if m:
                fact_text  = m.group(1).strip()
                proof_text = m.group(2).strip()
                steps.append({
                    "depth": depth_num,
                    "fact":  fact_text,
                    "proof": proof_text,
                })
                facts.add(fact_text)

    return steps, facts


def format_gt_trace(steps):
    """Format ground-truth steps as a numbered reasoning trace for the LLM prompt."""
    if not steps:
        return "  (no proof steps — answer may be Unknown)"
    lines = []
    for s in steps:
        lines.append(f"  [depth {s['depth']}] {s['fact']}  (proof: {s['proof']})")
    return "\n".join(lines)

# ─── Regex Symbolic Parser (Pipeline B) ──────────────────────────────────────

def parse_knowledge_base(theory):
    facts = set()
    rules = []
    sentences = [s.strip() for s in re.split(r"\.\s*", theory) if s.strip()]

    for sent in sentences:
        low = sent.lower()

        m = re.match(
            r"^if (?:someone|something) is (.+?) then (?:they|it) (?:are|is) (not )?(.+)$", low
        )
        if m:
            rules.append({
                "type": "generic_if", "condition": m.group(1).strip(),
                "negated": bool(m.group(2)), "consequent": m.group(3).strip(), "raw": sent
            })
            continue

        m = re.match(
            r"^if (\w[\w\s]*?) is (\w[\w\s]*?)(?: and (\w[\w\s]*?) is (\w[\w\s]*?))?"
            r" then (\w[\w\s]*?) is (not )?(.+)$", low
        )
        if m:
            rules.append({
                "type": "specific_if",
                "subj1": m.group(1).strip(), "prop1": m.group(2).strip(),
                "subj2": m.group(3).strip() if m.group(3) else None,
                "prop2": m.group(4).strip() if m.group(4) else None,
                "cons_subj": m.group(5).strip(), "negated": bool(m.group(6)),
                "consequent": m.group(7).strip(), "raw": sent
            })
            continue

        m = re.match(
            r"^all (\w[\w\s,]*?)(?:\s+things?|\s+people|\s+animals?)? are (not )?(.+)$", low
        )
        if m:
            rules.append({
                "type": "forall", "kind": m.group(1).strip(),
                "negated": bool(m.group(2)), "property": m.group(3).strip(), "raw": sent
            })
            continue

        m = re.match(r"^(\w[\w\s,]*?) things? are (not )?(.+)$", low)
        if m:
            rules.append({
                "type": "forall", "kind": m.group(1).strip(),
                "negated": bool(m.group(2)), "property": m.group(3).strip(), "raw": sent
            })
            continue

        m = re.match(r"^(\w[\w\s]*?) is not (.+)$", low)
        if m:
            facts.add(f"NOT({m.group(1).strip()} is {m.group(2).strip()})")
            continue

        m = re.match(r"^(\w[\w\s]*?) is (\w[\w\s]*)$", low)
        if m:
            facts.add(f"{m.group(1).strip()} is {m.group(2).strip()}")
            continue

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
                sbk   = defaultdict(set)
                for fact in list(derived):
                    m = re.match(r"^(\w[\w\s]*?) is (.+)$", fact)
                    if m:
                        subj = m.group(1).strip()
                        attr = m.group(2).strip()
                        for k in kinds:
                            if attr == k or attr.rstrip("s") == k.rstrip("s"):
                                sbk[k].add(subj)
                eligible = (
                    sbk[kinds[0]] if len(kinds) == 1
                    else set.intersection(*[sbk[k] for k in kinds])
                    if all(sbk[k] for k in kinds) else set()
                )
                for subj in eligible:
                    new = f"NOT({subj} is {prop})" if neg else f"{subj} is {prop}"
                    if new not in derived:
                        derived.add(new)
                        trace.append({"rule": rule["raw"], "derived": new})
                        changed = True

            elif rtype == "generic_if":
                parts  = [c.strip() for c in re.split(r"\band\b", rule["condition"])]
                conseq = rule["consequent"]
                for fact in list(derived):
                    m = re.match(r"^(\w[\w\s]*?) is (.+)$", fact)
                    if not m: continue
                    subj = m.group(1).strip()
                    ok   = all(
                        (f"{subj} is {p[4:].strip()}" not in derived if p.startswith("not ")
                         else f"{subj} is {p}" in derived)
                        for p in parts
                    )
                    if ok:
                        new = f"NOT({subj} is {conseq})" if neg else f"{subj} is {conseq}"
                        if new not in derived:
                            derived.add(new)
                            trace.append({"rule": rule["raw"], "derived": new})
                            changed = True

            elif rtype == "specific_if":
                s1, p1 = rule["subj1"], rule["prop1"]
                s2, p2 = rule.get("subj2"), rule.get("prop2")
                cs, cq = rule["cons_subj"], rule["consequent"]
                ok = f"{s1} is {p1}" in derived and (not s2 or f"{s2} is {p2}" in derived)
                if ok:
                    new = f"NOT({cs} is {cq})" if neg else f"{cs} is {cq}"
                    if new not in derived:
                        derived.add(new)
                        trace.append({"rule": rule["raw"], "derived": new})
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
        if llama: return llama[0]
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

ANSWER_INSTRUCTION = (
    "Respond with ONLY one of: True, False, or Unknown — "
    "then a single sentence explaining why."
)

def prompt_llm_only(theory, question):
    return (
        f"You are a logical reasoning assistant.\n"
        f"Read the theory carefully and answer the question.\n"
        f"{ANSWER_INSTRUCTION}\n\n"
        f"Theory:\n{theory}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def prompt_regex_symbolic(theory, question, derived, trace):
    facts_str = "\n".join(f"  - {f}" for f in sorted(derived))
    trace_str = (
        "\n".join(
            f"  [{j+1}] Rule: \"{t['rule']}\"  =>  derived: \"{t['derived']}\""
            for j, t in enumerate(trace)
        ) if trace else "  (no new facts derived)"
    )
    return (
        f"You are a logical reasoning assistant.\n"
        f"A forward-chaining symbolic reasoner has processed the theory.\n"
        f"Use the derived facts and trace to answer the question.\n"
        f"{ANSWER_INSTRUCTION}\n\n"
        f"DERIVED FACTS:\n{facts_str}\n\n"
        f"REASONING TRACE:\n{trace_str}\n\n"
        f"Original Theory:\n{theory}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def prompt_gt_symbolic(theory, question, gt_steps):
    trace_str = format_gt_trace(gt_steps)
    return (
        f"You are a logical reasoning assistant.\n"
        f"The following ground-truth proof steps have been formally verified for this theory.\n"
        f"Use them to answer the question.\n"
        f"{ANSWER_INSTRUCTION}\n\n"
        f"GROUND-TRUTH PROOF STEPS:\n{trace_str}\n\n"
        f"Original Theory:\n{theory}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

# ─── Accuracy Helpers ─────────────────────────────────────────────────────────

def make_stats():
    return {"llm": 0, "regex": 0, "gt": 0, "total": 0}

# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    args = parse_args()

    print("\n" + "="*68)
    print("  Neuro-Symbolic 3-Way Comparison — ProofWriter OWA")
    print("  Pipeline A: LLM-Only")
    print("  Pipeline B: LLM + Regex Parser (forward chaining)")
    print("  Pipeline C: LLM + Ground-Truth Proofs (allProofs)")
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
    llm_correct  = regex_correct = gt_correct = 0
    config_stats = defaultdict(make_stats)
    qdep_stats   = defaultdict(make_stats)
    total        = len(problems)

    for i, prob in enumerate(problems, 1):
        theory   = prob["theory"]
        question = prob["question"]
        expected = prob["answer"]
        cfg      = prob["config"]
        qdep     = prob["depth"]

        print(f"[{i:>4}/{total}] {cfg}  QDep={qdep}  {prob['id']}")
        print(f"          Q: {question}")

        # ── Pipeline B: Regex symbolic ──────────────────────────
        facts, rules              = parse_knowledge_base(theory)
        derived, new_facts, trace = forward_chain(facts, rules)

        # ── Pipeline C: Ground-truth proofs ─────────────────────
        gt_steps, gt_facts = parse_allproofs(prob["allProofs"])

        # ── LLM calls (3 parallel prompts) ──────────────────────
        llm_resp,   llm_time   = call_ollama(prompt_llm_only(theory, question), model)
        regex_resp, regex_time = call_ollama(prompt_regex_symbolic(theory, question, derived, trace), model)
        gt_resp,    gt_time    = call_ollama(prompt_gt_symbolic(theory, question, gt_steps), model)

        llm_verdict   = extract_verdict(llm_resp)
        regex_verdict = extract_verdict(regex_resp)
        gt_verdict    = extract_verdict(gt_resp)

        llm_ok   = llm_verdict   == expected
        regex_ok = regex_verdict == expected
        gt_ok    = gt_verdict    == expected

        if llm_ok:   llm_correct   += 1
        if regex_ok: regex_correct += 1
        if gt_ok:    gt_correct    += 1

        for stats in [config_stats[cfg], qdep_stats[qdep]]:
            stats["total"] += 1
            if llm_ok:   stats["llm"]   += 1
            if regex_ok: stats["regex"] += 1
            if gt_ok:    stats["gt"]    += 1

        print(f"          LLM-Only  [{'✓' if llm_ok   else '✗'}] → {llm_verdict!r:9s}  ({llm_time}s)")
        print(f"          Regex     [{'✓' if regex_ok else '✗'}] → {regex_verdict!r:9s}  ({regex_time}s)")
        print(f"          GT Proof  [{'✓' if gt_ok    else '✗'}] → {gt_verdict!r:9s}  ({gt_time}s)")
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

            # Pipeline A
            "llm_only": {
                "verdict":      llm_verdict,
                "correct":      llm_ok,
                "time_s":       llm_time,
                "raw_response": llm_resp,
            },

            # Pipeline B
            "llm_regex_symbolic": {
                "verdict":      regex_verdict,
                "correct":      regex_ok,
                "time_s":       regex_time,
                "raw_response": regex_resp,
                "symbolic_reasoning": {
                    "facts_parsed":    sorted(facts),
                    "rules_parsed":    [r["raw"] for r in rules],
                    "facts_derived":   sorted(new_facts),
                    "reasoning_trace": [
                        {"step": j+1, "rule": t["rule"], "derived": t["derived"]}
                        for j, t in enumerate(trace)
                    ],
                }
            },

            # Pipeline C
            "llm_gt_symbolic": {
                "verdict":      gt_verdict,
                "correct":      gt_ok,
                "time_s":       gt_time,
                "raw_response": gt_resp,
                "ground_truth_proof": {
                    "steps":       gt_steps,
                    "facts_total": sorted(gt_facts),
                }
            },
        })

    # ── Summary ──────────────────────────────────────────────────
    n         = len(problems)
    llm_acc   = round(llm_correct   / n * 100, 1)
    regex_acc = round(regex_correct / n * 100, 1)
    gt_acc    = round(gt_correct    / n * 100, 1)

    def breakdown(stats_dict, keys):
        out = {}
        for k in keys:
            s = stats_dict[k]
            t = s["total"]
            out[str(k)] = {
                "total":         t,
                "llm_correct":   s["llm"],
                "regex_correct": s["regex"],
                "gt_correct":    s["gt"],
                "llm_accuracy":  round(s["llm"]   / t * 100, 1) if t else 0,
                "regex_accuracy":round(s["regex"] / t * 100, 1) if t else 0,
                "gt_accuracy":   round(s["gt"]    / t * 100, 1) if t else 0,
            }
        return out

    summary = {
        "timestamp":      datetime.now().isoformat(),
        "model":          model,
        "dataset":        "tasksource/proofwriter",
        "assumption":     "OWA (open world)",
        "split":          args.split,
        "configs":        OWA_CONFIGS,
        "n_per_depth":    args.n,
        "n_total":        n,
        "llm_only":       {"correct": llm_correct,   "accuracy_pct": llm_acc},
        "llm_regex":      {"correct": regex_correct, "accuracy_pct": regex_acc},
        "llm_gt_proof":   {"correct": gt_correct,    "accuracy_pct": gt_acc},
        "delta_regex_vs_llm": round(regex_acc - llm_acc, 1),
        "delta_gt_vs_llm":    round(gt_acc    - llm_acc, 1),
        "delta_gt_vs_regex":  round(gt_acc    - regex_acc, 1),
        "by_config": breakdown(config_stats, OWA_CONFIGS),
        "by_qdep":   breakdown(qdep_stats, sorted(qdep_stats)),
    }

    # ── Print ─────────────────────────────────────────────────────
    print("="*68)
    print("  FINAL RESULTS — 3-Way Comparison")
    print("="*68)
    print(f"  Pipeline A  LLM-Only      : {llm_correct:>3}/{n}  =  {llm_acc}%")
    print(f"  Pipeline B  LLM + Regex   : {regex_correct:>3}/{n}  =  {regex_acc}%")
    print(f"  Pipeline C  LLM + GT Proof: {gt_correct:>3}/{n}  =  {gt_acc}%")
    print()
    sign = lambda d: f"+{d}" if d >= 0 else str(d)
    print(f"  Δ Regex  vs LLM   : {sign(round(regex_acc - llm_acc,   1))}%")
    print(f"  Δ GT     vs LLM   : {sign(round(gt_acc    - llm_acc,   1))}%")
    print(f"  Δ GT     vs Regex : {sign(round(gt_acc    - regex_acc, 1))}%")

    print(f"\n  {'Config':<12}  {'LLM':>7}  {'Regex':>7}  {'GT Proof':>9}  {'N':>5}")
    print(f"  {'-'*50}")
    for cfg in OWA_CONFIGS:
        v = summary["by_config"].get(cfg, {})
        if not v or v["total"] == 0:
            print(f"  {cfg:<12}  {'—':>7}  {'—':>7}  {'—':>9}  {'0':>5}")
        else:
            print(f"  {cfg:<12}  {v['llm_accuracy']:>6.1f}%  {v['regex_accuracy']:>6.1f}%  {v['gt_accuracy']:>8.1f}%  {v['total']:>5}")

    print(f"\n  {'QDep':<8}  {'LLM':>7}  {'Regex':>7}  {'GT Proof':>9}  {'N':>5}")
    print(f"  {'-'*46}")
    for d, v in summary["by_qdep"].items():
        print(f"  {d:<8}  {v['llm_accuracy']:>6.1f}%  {v['regex_accuracy']:>6.1f}%  {v['gt_accuracy']:>8.1f}%  {v['total']:>5}")
    print("="*68)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Detailed results  →  {RESULTS_FILE}")
    print(f"  Summary           →  {SUMMARY_FILE}\n")


if __name__ == "__main__":
    run()